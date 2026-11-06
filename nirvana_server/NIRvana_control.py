"""LightField adapter, also importable for offline tests without pythonnet.

One instance is shared by both TCP ports. Mutating operations never queue behind
another acquisition/movement. LightField's own experiment is also checked busy.
"""
import csv
import io
import math
from pathlib import Path
import threading
import time
from contextlib import contextmanager

PROTOCOL = "nirvana-control-v1"


class LightFieldController:
    def __init__(self, experiment, spectrometer_settings, camera_settings,
                 experiment_settings, export_dir, double=float, integer=int):
        self.experiment = experiment
        self.spectrometer_settings = spectrometer_settings
        self.camera_settings = camera_settings
        self.experiment_settings = experiment_settings
        self.double, self.integer = double, integer
        self.csv_path = Path(export_dir) / "tempdat.csv"
        self.lock = threading.Lock()
        self.preview_token = None
        self.preview_touched = 0
        self.preview_data = None
        self.preview_error = None
        self.preview_sequence = 0
        self.completed = threading.Event()
        self.preview_available = False
        self.completion_available = False
        try:
            self.experiment.ImageDataSetReceived += self._on_frame
            self.preview_available = True
        except Exception as exc:
            print("Live preview unavailable:", exc)
        try:
            self.experiment.ExperimentCompleted += self._on_completed
            self.completion_available = True
        except Exception as exc:
            print("Completion event unavailable; using IsRunning:", exc)

    @contextmanager
    def exclusive(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("LightField is busy; wait for the current operation to finish.")
        try:
            yield
        finally:
            self.lock.release()

    def _setting(self, group, *names):
        for name in names:
            key = getattr(group, name, None)
            if key is not None and self.experiment.Exists(key):
                return key
        return None

    def _require_idle(self):
        if self.preview_token or bool(self.experiment.IsRunning):
            raise RuntimeError("LightField is acquiring/previewing. Stop it before changing settings.")

    def _set(self, key, value):
        if key is None:
            raise RuntimeError("This setting is unavailable in the loaded LightField experiment.")
        if not self.experiment.IsValid(key, value):
            raise ValueError("LightField rejected the requested value: {!r}".format(value))
        self.experiment.SetValue(key, value)

    def _gratings(self, key):
        if key is None:
            return []
        return list(self.experiment.GetCurrentCapabilities(key))

    def _status(self):
        grating_key = self._setting(self.spectrometer_settings, "GratingSelected", "Grating")
        wl_key = self._setting(self.spectrometer_settings, "GratingCenterWavelength")
        choices_error = None
        try:
            gratings = [str(value) for value in self._gratings(grating_key)]
        except Exception as exc:
            gratings, choices_error = [], str(exc)
        status = {
            "status_source": "lightfield", "grating": None, "wl": None,
            "gratings": gratings, "grating_choices_error": choices_error,
            "capabilities": {"grating": bool(gratings), "wavelength": wl_key is not None,
                             "preview": self.preview_available},
            "busy": bool(self.experiment.IsRunning) or self.preview_token is not None,
        }
        if grating_key is not None:
            status["grating"] = str(self.experiment.GetValue(grating_key))
        if wl_key is not None:
            status["wl"] = float(self.experiment.GetValue(wl_key))
        for name, field in (("SensorTemperatureReading", "temp"),
                            ("SensorTemperatureSetPoint", "target_temp")):
            key = self._setting(self.camera_settings, name)
            status[field] = float(self.experiment.GetValue(key)) if key else None
        return status

    def command(self, request):
        with self.exclusive():
            command = request.get("command")
            if command == "get_status":
                return self._status()
            if command in ("set_grating", "set_wavelength"):
                self._require_idle()
                if command == "set_grating":
                    key = self._setting(self.spectrometer_settings, "GratingSelected", "Grating")
                    # Keep the original .NET value/type, not a guessed index or string.
                    matches = [g for g in self._gratings(key) if str(g) == str(request.get("value"))]
                    if len(matches) != 1:
                        raise ValueError("Select an exact grating reported by LightField.")
                    value = matches[0]
                else:
                    key = self._setting(self.spectrometer_settings, "GratingCenterWavelength")
                    number = float(request["value"])
                    if not math.isfinite(number) or number <= 0:
                        raise ValueError("Wavelength must be finite and positive (nm).")
                    value = self.double(number)
                self.preview_data = None
                self._set(key, value)
                # This is LightField readback, not an independent mechanical encoder.
                actual = self.experiment.GetValue(key)
                matches = (str(actual) == str(value) if command == "set_grating"
                           else abs(float(actual) - float(value)) <= 0.01)
                if not matches:
                    raise RuntimeError("LightField readback does not match the request; check the device PC.")
                return self._status()
            if command == "start_preview":
                self._require_idle()
                if not self.preview_available:
                    raise RuntimeError("LightField image events unavailable; live preview cannot start.")
                token = request.get("token")
                if not isinstance(token, str) or not token:
                    raise ValueError("A preview session token is required.")
                self._set_exposure(request["exposure_time"])
                self.preview_data, self.preview_error = None, None
                self.preview_token = token
                self.preview_touched = time.monotonic()
                try:
                    self.experiment.Preview()
                except Exception:
                    self.preview_token = None
                    raise
                return {"preview": True}
            if command in ("get_preview", "stop_preview"):
                if request.get("token") != self.preview_token or not self.preview_token:
                    raise RuntimeError("Preview session is absent or belongs to another client.")
                if command == "stop_preview":
                    self._stop_preview()
                    return {"preview": False}
                self.preview_touched = time.monotonic()
                if self.preview_error:
                    raise RuntimeError(self.preview_error)
                return {"frame": self.preview_data}
            raise ValueError("Unknown command: {!r}".format(command))

    def _set_exposure(self, milliseconds):
        value = float(milliseconds)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Exposure must be finite and positive (milliseconds).")
        key = self._setting(self.camera_settings, "ShutterTimingExposureTime")
        self._set(key, self.double(value))

    def _on_completed(self, sender, args):
        self.completed.set()

    def _on_frame(self, sender, args):
        # Do not acquire the command lock: LightField can call back during Preview/Stop.
        if self.preview_token is None:
            return
        try:
            dataset = args.ImageDataSet
            if dataset.Regions.Length != 1:
                raise ValueError("Live preview requires one LightField region of interest.")
            frame = dataset.GetFrame(0, int(dataset.Frames) - 1)
            width, height = int(frame.Width), int(frame.Height)
            raw = [float(value) for value in frame.GetData()]
            calibration = self.experiment.SystemColumnCalibration
            wavelengths = [float(calibration.Get(i)) for i in range(calibration.Length)]
            if width != len(wavelengths) or len(raw) != width * height or not width or not height:
                raise ValueError("Live frame dimensions do not match LightField wavelength calibration.")
            # A full camera image is vertically summed for the alignment spectrum.
            counts = [sum(raw[column::width]) for column in range(width)]
            if not all(math.isfinite(v) for v in wavelengths + counts) or min(wavelengths) <= 0:
                raise ValueError("Invalid live spectrum/calibration.")
            self.preview_sequence += 1
            self.preview_data = {"sequence": self.preview_sequence, "wavelengths": wavelengths,
                                 "intensities": counts, "rows_summed": height}
        except Exception as exc:
            self.preview_error = str(exc)

    def _stop_preview(self):
        self.experiment.Stop()
        self.preview_token = None
        self.preview_data = None

    def expire_preview(self):
        # Recover if a dashboard disappears without sending Stop.
        if not self.lock.acquire(blocking=False):
            return
        try:
            if self.preview_token and time.monotonic() - self.preview_touched > 15:
                self._stop_preview()
        finally:
            self.lock.release()

    def acquire_csv(self, request, send):
        """Legacy wire protocol: one JSON request, CSV response terminated by EOF."""
        if request.get("command") != "start_acquisition":
            raise ValueError("Unknown acquisition command; no files changed.")
        with self.exclusive():
            self._require_idle()
            frames = int(request.get("num_frames", 1))
            if frames <= 0 or frames != float(request.get("num_frames", 1)):
                raise ValueError("num_frames must be a positive integer.")
            exposure = float(request.get("exposure_time", 1.0))
            self._set_exposure(exposure)
            key = self._setting(self.experiment_settings, "AcquisitionFramesToStore", "FrameSettingsFramesToStore")
            self._set(key, self.integer(frames))
            # Only an accepted acquisition may remove the previous temporary CSV.
            if self.csv_path.exists():
                self.csv_path.unlink()
            self.completed.clear()
            self.experiment.Acquire()
            deadline = time.monotonic() + max(140, exposure * frames / 1000 + 140)
            previous, stable_since = None, time.monotonic()
            while time.monotonic() < deadline:
                done = self.completed.is_set() if self.completion_available else not self.experiment.IsRunning
                if done and self.csv_path.exists():
                    data = self.csv_path.read_bytes()
                    if data != previous:
                        previous, stable_since = data, time.monotonic()
                    elif data and time.monotonic() - stable_since >= 0.4:
                        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
                        if not rows or not {"Frame", "Wavelength", "Intensity"}.issubset(rows[0]):
                            raise ValueError("CSV export needs Frame, Wavelength and Intensity columns.")
                        if {int(row["Frame"]) for row in rows} == set(range(1, frames + 1)):
                            send(data)
                            self.csv_path.unlink()  # Retain CSV if transmission failed.
                            return
                time.sleep(0.1)
            raise TimeoutError("LightField acquisition/CSV export did not complete; check the device PC.")
