import Pyro4
import numpy as np
import threading
import time
import uuid
from drivers.nirvana_control_client import NIRvanaControlClient
from datetime import datetime
from qcodes.instrument import Instrument
from qcodes.parameters import ArrayParameter

def remote_get(proxy, property_name, method_name=None):
    """
    Get a property from a Pyro proxy.
    If a direct property/attribute exists and is not callable, return it.
    Otherwise, fall back to calling the getter method.
    """
    try:
        val = getattr(proxy, property_name)
        if not callable(val):
            return val
    except AttributeError:
        pass

    m_name = method_name or f"get_{property_name}"
    method = getattr(proxy, m_name)
    return method()

def remote_set(proxy, property_name, val, method_name=None):
    """
    Set a property on a Pyro proxy.
    If a setter method exists, call it. Otherwise, set the attribute directly.
    """
    m_name = method_name or f"set_{property_name}"
    try:
        method = getattr(proxy, m_name)
        method(val)
        return
    except AttributeError:
        pass

    setattr(proxy, property_name, val)

class SimulatedLightFieldWrapper:
    def __init__(self, camera_inst):
        self.camera_inst = camera_inst
        self.grating = "[1500 g/mm][0][0]"
        self.target_temperature = -70.0
        self.temperature = -70.0
        self.exposure_time = 0.1
        self.wavelength = 760.0
        self.shutter_control = 1
        self.num_frames = 1
        self._cached_wavelengths = None

    def get_status(self):
        return {
            'success': True,
            'temp': self.temperature,
            'target_temp': self.target_temperature,
            'status': 1,
            'gain': 1.0,
            'grating': self.grating,
            'exposure': self.exposure_time,
            'wl': self.wavelength,
            'shutter': self.shutter_control,
            'num_frames': self.num_frames
        }

    def start_preview(self):
        pass

    def stop_preview(self):
        pass

    def get_latest_preview_data(self):
        res = self.get_spectrum()
        if res is not None:
            return res[0][0]
        return None

    def get_wavelengths_cache(self):
        if self._cached_wavelengths is not None:
            return self._cached_wavelengths
        self._cached_wavelengths = np.linspace(self.wavelength - 60, self.wavelength + 60, 512).tolist()
        return self._cached_wavelengths

    def abort_acquisition(self):
        pass

    def set_grating(self, val):
        self.grating = val

    def set_shutter_control(self, val):
        self.shutter_control = int(val)

    def set_target_temperature(self, val):
        self.target_temperature = float(val)

    def get_spectrum(self, *args, **kwargs):
        wlens = self.get_wavelengths_cache()
        num_pixels = len(wlens)
        v1 = 0.0
        v2 = 0.0
        b_field = 0.0
        try:
            smu = self.camera_inst.find_instrument('smu')
            v1 = smu.ch1.volt() if hasattr(smu.ch1, 'volt') else 0.0
            v2 = smu.ch2.volt() if hasattr(smu.ch2, 'volt') else 0.0
        except Exception:
            pass
        try:
            magnet = self.camera_inst.find_instrument('magnet')
            b_field = magnet.field() if hasattr(magnet, 'field') else 0.0
        except Exception:
            pass
        
        # Peak position: center 760nm, shifts with V1
        peak_pos = 760.0 + v1 * 20.0
        # Peak width: proportional to V2
        peak_width = max(2.0, 10.0 + v2 * 0.5)
        # Height: dependent on B field
        peak_height = max(100.0, 5000.0 * (1.0 + np.sin(b_field)) + 1000.0)
        
        wlens_arr = np.array(wlens)
        gaussian = peak_height * np.exp(-((wlens_arr - peak_pos) / peak_width)**2)
        noise = np.random.normal(200.0, 15.0, num_pixels)
        spectrum = np.clip(gaussian + noise, 0, 65535)
        frames = [spectrum.tolist()] * self.num_frames
        return frames, wlens

class PyroProxyWrapper:
    def __init__(self, proxy):
        object.__setattr__(self, '_proxy', proxy)
        object.__setattr__(self, '_cached_wavelengths', None)

    def __getattr__(self, name):
        if name == 'get_status':
            return self.get_status
        elif name == 'start_preview':
            return self.start_preview
        elif name == 'stop_preview':
            return self.stop_preview
        elif name == 'get_latest_preview_data':
            return self.get_latest_preview_data
        elif name == 'get_wavelengths_cache':
            return self.get_wavelengths_cache
        elif name == 'abort_acquisition':
            return self.abort_acquisition
        elif name == 'get_spectrum':
            return self.get_spectrum
        elif name == 'set_grating':
            return self.set_grating
        elif name == 'set_shutter_control':
            return self.set_shutter_control
        elif name == 'set_target_temperature':
            return self.set_target_temperature
        elif name == 'set_exposure_time':
            return self.set_exposure_time
        elif name == 'set_num_frames':
            return self.set_num_frames
        elif name == 'set_wavelength':
            return self.set_wavelength
        
        # Forward everything else to the real proxy
        return getattr(object.__getattribute__(self, '_proxy'), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, '_proxy'), name, value)

    def __dir__(self):
        real_proxy = object.__getattribute__(self, '_proxy')
        try:
            attrs = dir(real_proxy)
        except Exception:
            attrs = []
        return attrs + [
            'get_status', 'start_preview', 'stop_preview',
            'get_latest_preview_data', 'get_wavelengths_cache',
            'abort_acquisition', 'get_spectrum', 'set_grating',
            'set_shutter_control', 'set_target_temperature',
            'set_exposure_time', 'set_num_frames', 'set_wavelength'
        ]

    def __enter__(self):
        object.__getattribute__(self, '_proxy').__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return object.__getattribute__(self, '_proxy').__exit__(exc_type, exc_val, exc_tb)

    def set_grating(self, val):
        proxy = object.__getattribute__(self, '_proxy')
        if isinstance(val, str):
            known = {"[hol,1500][0][0]": 3, "[750  nm,1200][1][0]": 1,
                     "[1.0  \u00e6m,300][2][0]": 2}
            try:
                g_val = known[val] if val in known else int(val)
            except ValueError:
                raise ValueError(f"Unknown WinSpec grating: {val!r}") from None
        else:
            g_val = int(val)
        if g_val not in (1, 2, 3):
            raise ValueError(f"Unknown WinSpec grating: {val!r}")
        proxy.grating = g_val

    def set_shutter_control(self, val):
        proxy = object.__getattribute__(self, '_proxy')
        proxy.shutter_control = int(val)

    def set_target_temperature(self, val):
        proxy = object.__getattribute__(self, '_proxy')
        proxy.target_temperature = float(val)

    def set_exposure_time(self, val):
        proxy = object.__getattribute__(self, '_proxy')
        proxy.exposure_time = float(val)

    def set_num_frames(self, val):
        proxy = object.__getattribute__(self, '_proxy')
        proxy.num_frames = int(val)

    def set_wavelength(self, val):
        proxy = object.__getattribute__(self, '_proxy')
        proxy.wavelength = float(val)

    def get_status(self):
        proxy = object.__getattribute__(self, '_proxy')
        try:
            # Safely fetch properties with default fallbacks
            temp = getattr(proxy, 'temperature', None)
            target_temp = getattr(proxy, 'target_temperature', None)
            grating_num = getattr(proxy, 'grating', None)
            exposure = getattr(proxy, 'exposure_time', None)
            wl = getattr(proxy, 'wavelength', None)
            shutter = getattr(proxy, 'shutter_control', None)
            num_frames = getattr(proxy, 'num_frames', None)

            # Map WinSpec grating number back to LightField-style string representation
            grating = None
            if grating_num == 1:
                grating = "[750  nm,1200][1][0]"
            elif grating_num == 2:
                grating = "[1.0  \u00e6m,300][2][0]"
            elif grating_num == 3:
                grating = "[hol,1500][0][0]"
            else:
                grating = grating_num

            # Map to expected structure
            data = {
                'success': True,
                'temp': temp,
                'target_temp': target_temp,
                'status': 1 if (temp is not None and target_temp is not None and abs(temp - target_temp) < 1.0) else 2,
                'gain': 1.0,
                'grating': grating,
                'exposure': exposure,
                'wl': wl,
                'shutter': shutter,
                'num_frames': num_frames
            }
            return data
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def start_preview(self):
        proxy = object.__getattribute__(self, '_proxy')
        if hasattr(proxy, "start_focus"):
            try:
                proxy.start_focus()
                return
            except Exception as e:
                print(f"Wrapper: start_focus failed, falling back: {e}")
        pass

    def stop_preview(self):
        proxy = object.__getattribute__(self, '_proxy')
        if hasattr(proxy, "stop_focus"):
            try:
                proxy.stop_focus()
                return
            except Exception as e:
                print(f"Wrapper: stop_focus failed, falling back: {e}")
        pass

    def get_latest_preview_data(self):
        proxy = object.__getattribute__(self, '_proxy')
        if hasattr(proxy, "get_focus_data"):
            try:
                return proxy.get_focus_data()
            except Exception as e:
                print(f"Wrapper: get_focus_data failed: {e}")
                return None
                
        try:
            # We call get_spectrum(wlen=False)
            spectra = proxy.get_spectrum(wlen=False)
            spectra_arr = np.array(spectra)
            if spectra_arr.ndim == 3:
                # Sum over ROI axis (axis 0) to get (frames, pixels), then take first frame
                spectra_arr = np.sum(spectra_arr, axis=0)
            if spectra_arr.ndim == 2:
                return spectra_arr[0].tolist()
            return spectra_arr.tolist()
        except Exception as e:
            print(f"Wrapper: failed to get latest preview data: {e}")
            return None

    def get_wavelengths_cache(self):
        # Return local cache if available
        cached = object.__getattribute__(self, '_cached_wavelengths')
        if cached is not None:
            return cached
            
        proxy = object.__getattribute__(self, '_proxy')
        try:
            # Try to get polynomial calibration first as it is much faster (<1s vs ~20s)
            if hasattr(proxy, 'get_calibration_poly'):
                wlens = proxy.get_calibration_poly()
            else:
                wlens = proxy.get_calibration()
            if wlens is not None:
                object.__setattr__(self, '_cached_wavelengths', wlens)
            return wlens
        except Exception:
            return None

    def abort_acquisition(self):
        proxy = object.__getattribute__(self, '_proxy')
        # Force-close the socket at the OS level first to instantly break the blocked read
        try:
            conn = getattr(proxy, "_pyroConnection", None)
            if conn:
                sock = getattr(conn, "sock", None)
                if sock:
                    import socket
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
                    try:
                        sock.close()
                    except Exception:
                        pass
        except Exception:
            pass
        # Finally release the proxy connection in Pyro
        try:
            proxy._pyroRelease()
        except Exception:
            pass

    def get_spectrum(self, *args, **kwargs):
        # Force wlenpoly=True unless explicitly overridden
        if 'wlen' not in kwargs and 'wlenpoly' not in kwargs and not (len(args) > 0):
            kwargs['wlenpoly'] = True
            
        proxy = object.__getattribute__(self, '_proxy')
        res = proxy.get_spectrum(*args, **kwargs)
        
        # If it returned a tuple of (spectra, wlens)
        if isinstance(res, tuple) and len(res) == 2:
            spectra, wlens = res
            spectra_arr = np.array(spectra)
            if spectra_arr.ndim == 3:
                # Sum over ROI axis (axis 0) to get (frames, pixels)
                spectra_arr = np.sum(spectra_arr, axis=0)
                
            # Cache the wavelengths for subsequent get_wavelengths_cache calls
            if wlens is not None:
                object.__setattr__(self, '_cached_wavelengths', wlens)
                
            return spectra_arr.tolist(), wlens
        # If it returned just spectra (wlen=False)
        spectra_arr = np.array(res)
        if spectra_arr.ndim == 3:
            # Sum over ROI axis (axis 0) to get (frames, pixels)
            spectra_arr = np.sum(spectra_arr, axis=0)
        return spectra_arr.tolist()

class NIRvanaSocketWrapper:
    def __init__(self, server_ip, port, *, enable_controls=False):
        self.server_ip = server_ip
        self.port = port
        self._exposure_time = 0.1  # seconds
        self._num_frames = 1
        self._cached_wavelengths = None
        self._active_socket = None
        self._acquisition_lock = threading.Lock()
        self.controls = NIRvanaControlClient(server_ip, int(port) + 1)
        self.controls_enabled = enable_controls
        self._preview_mode = None
        self._preview_token = None
        self._preview_sequence = None
        self._last_preview_frame = 0
        self.acquisition_health = {
            "state": "unverified", "last_success": None,
            "details": "No spectrum received in this session.",
        }

    def get_exposure_time(self):
        return self._exposure_time

    def set_exposure_time(self, val):
        self._exposure_time = float(val)

    def get_num_frames(self):
        return self._num_frames

    def set_num_frames(self, val):
        self._num_frames = int(val)

    def get_readout_mode(self):
        return "Full Frame"

    def set_readout_mode(self, val):
        pass

    def get_acquisition_mode(self):
        return "Single"

    def set_acquisition_mode(self, val):
        pass

    def get_status(self):
        status = {
            'success': True, 'status_source': 'local_settings',
            'temp': None, 'target_temp': None, 'status': None, 'gain': None,
            'grating': None, 'wl': None, 'shutter': None, 'capabilities': {},
        }
        if self.controls_enabled:
            try:
                status.update(self.controls.get_status())
            except RuntimeError as exc:
                status['control_error'] = str(exc)
        else:
            status['capabilities'] = {'preview': True, 'grating': False, 'wavelength': False}
            status['preview_mode'] = 'acquisition'
        # These are the settings for the NEXT acquisition, not the previous file.
        status.update(exposure=self._exposure_time, num_frames=self._num_frames)
        return status

    def _set_spectrometer(self, method, value):
        if not self.controls_enabled:
            raise RuntimeError("Set the grating/wavelength in LightField; this server provides acquisition only.")
        if not self._acquisition_lock.acquire(blocking=False):
            raise RuntimeError("NIRvana is busy acquiring/previewing; stop it before changing settings.")
        try:
            self._cached_wavelengths = None
            return getattr(self.controls, method)(value)
        finally:
            self._acquisition_lock.release()

    def set_grating(self, value):
        return self._set_spectrometer('set_grating', value)

    def set_wavelength(self, value):
        return self._set_spectrometer('set_wavelength', value)

    def start_preview(self):
        if not self._acquisition_lock.acquire(blocking=False):
            raise RuntimeError("NIRvana is busy. Finish its current operation first.")
        self._cached_wavelengths = None
        if not self.controls_enabled:
            # The unchanged server supports ordinary acquisitions only. Hold the
            # camera lock for the live session and request one frame at a time.
            self._preview_mode = 'acquisition'
            return
        self._preview_mode = 'native'
        self._preview_token = uuid.uuid4().hex
        self._preview_sequence = None
        self._cached_wavelengths = None
        self._last_preview_frame = time.monotonic()
        try:
            self.controls.request('start_preview', token=self._preview_token,
                                  exposure_time=self._exposure_time * 1000, timeout=5)
        except Exception:
            # Best effort cleanup also covers a lost reply after Preview succeeded.
            try:
                self.stop_preview()
            except Exception:
                pass  # Preserve the startup error; the server also expires abandoned previews.
            raise

    def stop_preview(self):
        if self._preview_mode == 'acquisition':
            self._preview_mode = None
            self._acquisition_lock.release()
            return
        if self._preview_token is None:
            return
        try:
            self.controls.request('stop_preview', token=self._preview_token, timeout=5)
        finally:
            self._preview_token = None
            self._preview_mode = None
            self._acquisition_lock.release()

    def get_latest_preview_data(self):
        if self._preview_mode == 'acquisition':
            intensities, _ = self._acquire({
                'command': 'start_acquisition',
                'exposure_time': self._exposure_time * 1000,
                'num_frames': 1,
            })
            return intensities[0]
        if self._preview_token is None:
            raise RuntimeError("Start NIRvana live preview first.")
        response = self.controls.request('get_preview', token=self._preview_token)
        frame = response.get('frame')
        if frame is None or frame['sequence'] == self._preview_sequence:
            if time.monotonic() - self._last_preview_frame > max(30, 3 * self._exposure_time):
                raise RuntimeError("No new LightField preview frame received; check the device PC.")
            return None
        wavelengths = np.asarray(frame['wavelengths'], dtype=float)
        intensities = np.asarray(frame['intensities'], dtype=float)
        if (wavelengths.ndim != 1 or intensities.shape != wavelengths.shape
                or not wavelengths.size or not np.all(np.isfinite(wavelengths))
                or not np.all(np.isfinite(intensities)) or np.any(wavelengths <= 0)):
            raise ValueError("Invalid NIRvana live frame/calibration.")
        self._preview_sequence = frame['sequence']
        self._last_preview_frame = time.monotonic()
        self._cached_wavelengths = wavelengths.tolist()
        return intensities.tolist()

    def get_wavelengths_cache(self):
        if self._cached_wavelengths is not None:
            return self._cached_wavelengths
        return None  # Calibration is only known after receiving a spectrum.

    def abort_acquisition(self):
        sock = self._active_socket
        if sock:
            import socket
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass

    def get_spectrum(self, *args, **kwargs):
        # The GUI uses seconds; Kilian's server requires milliseconds.
        spec_params = {
            "command": "start_acquisition",
            "exposure_time": self._exposure_time * 1e3,
            "num_frames": self._num_frames
        }
        if not self._acquisition_lock.acquire(blocking=False):
            raise RuntimeError("NIRvana is already acquiring or being verified. Wait for it to finish.")
        try:
            return self._acquire(spec_params)
        finally:
            self._acquisition_lock.release()

    def _set_health(self, state, details, **values):
        # Replace the snapshot atomically; the GUI only reads it, with no network polling.
        self.acquisition_health = {
            **self.acquisition_health, "state": state, "details": details, **values,
        }

    def verify_acquisition(self):
        """Check two complete request/acquire/export/receive cycles while idle.

        Test requests leave the user's local exposure/frame settings unchanged.
        No extra command, heartbeat, or retry is sent to the lab server.
        """
        if not self._acquisition_lock.acquire(blocking=False):
            raise RuntimeError("NIRvana is busy. Finish the measurement before checking it.")
        try:
            previous_wavelengths = None
            for check in (1, 2):
                self._set_health("checking", f"Acquiring test spectrum {check}/2 (50 ms, one frame).")
                _, wavelengths = self._acquire(
                    {"command": "start_acquisition", "exposure_time": 50.0, "num_frames": 1},
                    checking=True,
                )
                if previous_wavelengths is not None and wavelengths != previous_wavelengths:
                    raise ValueError("Wavelength calibration changed between the two test spectra.")
                previous_wavelengths = wavelengths
            details = (
                f"Two consecutive test spectra received from {self.server_ip}:{self.port}.\n"
                f"Each: 50 ms, one frame, {len(wavelengths)} pixels.\n"
                f"Wavelength range: {wavelengths[0]:.3f} to {wavelengths[-1]:.3f} nm.\n"
                "CSV structure, numeric values, frame count and wavelength grids validated.\n"
                "Test spectra are not saved as measurement data. Local measurement settings unchanged."
            )
            self._set_health("verified", details)
            return details
        except Exception as exc:
            self._set_health("failed", str(exc))
            raise
        finally:
            self._acquisition_lock.release()

    def _acquire(self, spec_params, *, checking=False):
        import NIRvana_send_trigger as trig

        if not checking:
            self._set_health("acquiring", "Waiting for a complete spectrum from NIRvana.")

        try:
            # The server can spend 140 s waiting for CSV export. A startup check
            # gets 180 s; normal acquisitions retain the lab client's 600 s timeout.
            timeout = {"acquisition_timeout_s": 180} if checking else {}
            wavelengths, intensities = trig.trigger(
                spec_params,
                device_ip=self.server_ip,
                port=self.port,
                on_socket=lambda sock: setattr(self, "_active_socket", sock),
                **timeout,
            )
            self._cached_wavelengths = wavelengths.tolist()
            self._set_health(
                "checking" if checking else "verified",
                f"Received {len(intensities)} frame(s), {len(wavelengths)} pixels.",
                last_success=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
            # The measurement routines expect (intensities, wavelengths).
            return intensities.tolist(), wavelengths.tolist()
        except Exception as exc:
            self._set_health("failed", str(exc))
            raise
        finally:
            self._active_socket = None

class NIRvanaMonoWrapper:
    def __init__(self, server_ip=None, port=65432, *, enable_controls=False):
        self.controls_enabled = enable_controls
        self.controls = NIRvanaControlClient(server_ip, int(port) + 1) if server_ip else None

    def get_wavelength(self):
        if not self.controls_enabled:
            raise RuntimeError("Wavelength is set in LightField; the acquisition server has no remote wavelength readback.")
        if self.controls is None:
            raise RuntimeError("NIRvana control endpoint is not configured.")
        return self.controls.get_wavelength()

    def set_wavelength(self, val):
        if not self.controls_enabled:
            raise RuntimeError("Set wavelength in LightField; the acquisition server has no remote wavelength control.")
        if self.controls is None:
            raise RuntimeError("NIRvana control endpoint is not configured.")
        return self.controls.set_wavelength(val)

class RemoteCCDImage(ArrayParameter):
    def __init__(self, name, instrument, shape=(512, 512), **kwargs):
        setpoints = (np.arange(shape[0]), np.arange(shape[1]))
        super().__init__(name=name, shape=shape, instrument=instrument, unit='counts', setpoints=setpoints, **kwargs)
        
    def get_raw(self) -> np.ndarray:
        return self.instrument.capture_image()

class ClientLightFieldCamera(Instrument):
    """
    QCoDeS client driver for the Princeton Instruments Camera 
    exposed by the unified LightField Pyro4 server.
    """
    def __init__(self, name, server_ip="DESKTOP-LSQKIT5.dhcp-int-phys.ethz.ch", port=50920, pyro_uri=None, spectrometer_type="lightfieldG15", simulated=False, nirvana_controls=False, **kwargs):
        super().__init__(name, **kwargs)
        self.spectrometer_type = spectrometer_type
        self.simulated = simulated
        
        if self.simulated:
            self._exposure_time = 0.1
            self._num_frames = 1
            self.remote_lf = SimulatedLightFieldWrapper(self)
            print(f"Initialized ClientLightFieldCamera '{name}' in Simulation Mode.")
        else:
            if spectrometer_type == 'lightfieldG11':
                self.remote_lf = NIRvanaSocketWrapper(server_ip, port, enable_controls=nirvana_controls)
            else:
                Pyro4.config.SERIALIZER = 'pickle'
                Pyro4.config.PICKLE_PROTOCOL_VERSION = 2
                if pyro_uri is None:
                    pyro_uri = f"PYRO:LightField@{server_ip}:{port}"
                proxy = Pyro4.Proxy(pyro_uri)
                if spectrometer_type in ['wingspecG15', 'winspecG11', 'wingspecG11', 'winspecG15']:
                    self.remote_lf = PyroProxyWrapper(proxy)
                else:
                    self.remote_lf = proxy
        
        self.add_parameter('exposure_time',
                           label='Exposure Time',
                           unit='s',
                           get_cmd=self._get_exposure_time,
                           set_cmd=self._set_exposure_time)

        self.add_parameter('num_frames',
                           label='Number of Frames',
                           get_cmd=self._get_num_frames,
                           set_cmd=self._set_num_frames)

        self.add_parameter('readout_mode',
                           label='Readout Mode',
                           get_cmd=lambda: "Full Frame",
                           set_cmd=lambda val: None if self.simulated else remote_set(self.remote_lf, 'readout_mode', val, 'set_readout_mode'))
                           
        self.add_parameter('image',
                           parameter_class=RemoteCCDImage,
                           shape=(512, 512))

    def _get_exposure_time(self):
        if self.simulated:
            return self._exposure_time
        return remote_get(self.remote_lf, 'exposure_time', 'get_exposure_time')

    def _set_exposure_time(self, val):
        if self.simulated:
            self._exposure_time = float(val)
            self.remote_lf.exposure_time = self._exposure_time
        else:
            remote_set(self.remote_lf, 'exposure_time', val, 'set_exposure_time')

    def _get_num_frames(self):
        if self.simulated:
            return self._num_frames
        return remote_get(self.remote_lf, 'num_frames', 'get_num_frames')

    def _set_num_frames(self, val):
        if self.simulated:
            self._num_frames = int(val)
            self.remote_lf.num_frames = self._num_frames
        else:
            remote_set(self.remote_lf, 'num_frames', val, 'set_num_frames')

    def capture_image(self) -> np.ndarray:
        if self.simulated:
            res = self.remote_lf.get_spectrum()
            intensities_arr = np.array(res[0][0])
            if len(intensities_arr) != 512:
                x_old = np.linspace(0, 1, len(intensities_arr))
                x_new = np.linspace(0, 1, 512)
                intensities_512 = np.interp(x_new, x_old, intensities_arr)
            else:
                intensities_512 = intensities_arr
        else:
            # Fetch the spectrum (Note: get_spectrum returns (intensities, wavelengths))
            if self.spectrometer_type in ['wingspecG15', 'winspecG11', 'wingspecG11', 'winspecG15']:
                res = self.remote_lf.get_spectrum(wlen=True)
            else:
                res = self.remote_lf.get_spectrum()
            if res is None:
                print(f"ClientLightFieldCamera Warning: Remote spectrometer '{self.spectrometer_type}' returned None. Verify camera hardware and server script on {self.remote_lf.server_ip if hasattr(self.remote_lf, 'server_ip') else 'remote host'}.")
                intensities_512 = np.zeros(512)
            else:
                intensities, wavelengths = res
                intensities_arr = np.array(intensities)
                if intensities_arr.ndim == 3:
                    # WinSpec returns 3D array (ROI, frames, pixels)
                    intensities_arr = np.mean(intensities_arr[0], axis=0)
                elif intensities_arr.ndim == 2:
                    intensities_arr = np.mean(intensities_arr, axis=0)
                
                # Interpolate intensities to 512 points to match the GUI shape
                if len(intensities_arr) != 512:
                    x_old = np.linspace(0, 1, len(intensities_arr))
                    x_new = np.linspace(0, 1, 512)
                    intensities_512 = np.interp(x_new, x_old, intensities_arr)
                else:
                    intensities_512 = intensities_arr
            
        # Return a 2D image matrix of shape (512, 512) where each row is intensities_512 / 512.
        # When GUI does np.sum(img, axis=0), it gets exactly intensities_512.
        img_row = intensities_512 / 512.0
        return np.tile(img_row, (512, 1))

    def get_idn(self) -> dict:
        return {
            'vendor': 'Princeton Instruments',
            'model': 'Remote LightField Camera Client',
            'serial': 'simulated' if self.simulated else 'remote',
            'firmware': None
        }

    def abort_acquisition(self) -> None:
        """Abort the current remote spectrum acquisition."""
        if self.simulated:
            return
        try:
            if self.spectrometer_type == 'lightfieldG11':
                # Closes the client receive socket; the protocol has no server abort command.
                self.remote_lf.abort_acquisition()
            elif self.spectrometer_type in ['wingspecG15', 'winspecG11', 'wingspecG11', 'winspecG15']:
                # Release the main proxy connection to close the active socket and free the thread
                if hasattr(self.remote_lf, "abort_acquisition"):
                    self.remote_lf.abort_acquisition()
                print("ClientLightFieldCamera: Released WinSpec main proxy connection to abort scan.")
            else:
                uri = getattr(self.remote_lf, '_pyroUri', None)
                if uri:
                    import Pyro4
                    with Pyro4.Proxy(uri) as abort_proxy:
                        abort_proxy._pyroTimeout = 2.0
                        abort_proxy.abort_acquisition()
                    print("ClientLightFieldCamera: Sent remote abort command via temporary proxy.")
        except Exception as e:
            print(f"ClientLightFieldCamera: Failed to send abort command: {e}")

class ClientLightFieldMonochromator(Instrument):
    """
    QCoDeS client driver for the Acton SpectraPro Spectrometer
    exposed by the unified LightField Pyro4 server.
    """
    def __init__(self, name, server_ip="DESKTOP-LSQKIT5.dhcp-int-phys.ethz.ch", port=50920, pyro_uri=None, spectrometer_type="lightfieldG15", simulated=False, nirvana_controls=False, **kwargs):
        super().__init__(name, **kwargs)
        self.simulated = simulated
        
        if self.simulated:
            self._wavelength = 1100.0
            print(f"Initialized ClientLightFieldMonochromator '{name}' in Simulation Mode.")
        else:
            if spectrometer_type == 'lightfieldG11':
                self.remote_lf = NIRvanaMonoWrapper(server_ip, port, enable_controls=nirvana_controls)
            else:
                Pyro4.config.SERIALIZER = 'pickle'
                Pyro4.config.PICKLE_PROTOCOL_VERSION = 2
                if pyro_uri is None:
                    pyro_uri = f"PYRO:LightField@{server_ip}:{port}"
                proxy = Pyro4.Proxy(pyro_uri)
                if spectrometer_type in ['wingspecG15', 'winspecG11', 'wingspecG11', 'winspecG15']:
                    self.remote_lf = PyroProxyWrapper(proxy)
                else:
                    self.remote_lf = proxy
        
        self.add_parameter('wavelength',
                           label='Wavelength',
                           unit='nm',
                           get_cmd=self._get_wavelength,
                           set_cmd=self._set_wavelength)

    def _get_wavelength(self):
        if self.simulated:
            return self._wavelength
        return remote_get(self.remote_lf, 'wavelength', 'get_wavelength')

    def _set_wavelength(self, val):
        if self.simulated:
            self._wavelength = float(val)
        else:
            remote_set(self.remote_lf, 'wavelength', val, 'set_wavelength')

    def get_idn(self) -> dict:
        return {
            'vendor': 'Acton',
            'model': 'Remote LightField Spectrometer Client',
            'serial': 'simulated' if self.simulated else 'remote',
            'firmware': None
        }
