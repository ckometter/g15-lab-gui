"""Incremental station initialization for the desktop device manager.

The legacy ``station_setup.initialize_station`` entry point remains available
for standalone measurement scripts. The dashboard uses explicit device checks
and reports when an unavailable physical device is replaced by simulation.
"""

import os
from contextlib import contextmanager

from qcodes.dataset import initialise_or_create_database_at
from qcodes.instrument import Instrument
from qcodes.station import Station

import station_setup
from drivers.remote_client import ClientLightFieldCamera, ClientLightFieldMonochromator, NIRvanaSocketWrapper


DEVICE_NAMES = (
    "smu",
    "monochromator",
    "rotator1",
    "rotator2",
    "rotator3",
    "camera",
    "piezo150",
    "piezo300",
    "magnet",
    "powermeter",
    "lockin",
    "lockin2",
    "awg",
    "dac",
)

DEVICE_LABELS = {
    "smu": "Keysight B2902B SMU",
    "monochromator": "Spectrometer wavelength control",
    "rotator1": "Rotator 1 (excitation)",
    "rotator2": "Rotator 2 (collection)",
    "rotator3": "Rotator 3",
    "camera": "Spectrometer camera",
    "piezo150": "Attocube ANC150",
    "piezo300": "Attocube ANC300",
    "magnet": "Oxford magnet",
    "powermeter": "Thorlabs PM100D power meter",
    "lockin": "SR830 lock-in 1 (bridge / transport current)",
    "lockin2": "SR830 lock-in 2 (transport voltage)",
    "awg": "Siglent SDG2042X AWG (bridge excitation)",
    "dac": "QPG uVoltDAQ DAC/ADC (gate voltages)",
}


def create_station(db_path):
    """Create the shared QCoDeS station once, before any device is opened."""
    Instrument.close_all()
    initialise_or_create_database_at(os.path.abspath(db_path))
    return Station()


def _spectrometer_config(spectrometer_type):
    resolved = {"wingspecG11": "winspecG11", "winspecG15": "wingspecG15"}.get(
        spectrometer_type, spectrometer_type
    )
    cfg = station_setup.SPECTROMETER_CONFIGS[resolved]
    return resolved, cfg.get("server_ip") or "DESKTOP-LSQKIT5.dhcp-int-phys.ethz.ch", cfg.get("port") or 50920, cfg.get("pyro_uri")


def _physical_powermeter():
    import pyvisa

    fallback = "USB0::0x1313::0x8078::P0013460::0::INSTR"
    manager = pyvisa.ResourceManager()
    # PyVISA reuses one manager per backend. Closing it here also closes the
    # ANC300, SMU and every other already-open VISA resource in this process.
    # Discovery borrows the manager; PyVISA owns its process-exit cleanup.
    addresses = [
        address for address in manager.list_resources()
        if "1313" in address and "USB" in address
    ]
    if fallback not in addresses:
        addresses.append(fallback)
    errors = []
    for address in addresses:
        try:
            return station_setup.ThorlabsPM100D("powermeter", address=address)
        except Exception as exc:
            errors.append(f"{address}: {exc}")
    raise RuntimeError("Power meter connection failed. " + "; ".join(errors))


def _physical_awg():
    """Prefer any Siglent USB generator (vendor 0xF4EC), then the configured address."""
    import pyvisa

    configured = station_setup.AWG_ADDRESS
    # Borrow the shared manager without closing it (see _physical_powermeter).
    manager = pyvisa.ResourceManager()
    try:
        found = [address for address in manager.list_resources()
                 if "USB" in address.upper() and "F4EC" in address.upper()]
    except Exception as exc:
        print(f"VISA enumeration failed ({exc}); trying the configured AWG address.")
        found = []
    addresses = list(dict.fromkeys(found + [configured]))
    errors = []
    for address in addresses:
        try:
            return station_setup.SiglentSDG2042X("awg", address=address)
        except Exception as exc:
            errors.append(f"{address}: {exc}")
    raise RuntimeError("Siglent AWG connection failed. " + "; ".join(errors))


def initialize_device(station, name, *, simulated, spectrometer_type=None):
    """Open one device and add it to the shared station; never silently fall back."""
    if name not in DEVICE_NAMES:
        raise ValueError(f"Unknown device: {name}")
    if name in station.components:
        raise RuntimeError(f"{name} is already initialized; restart the dashboard to change its mode.")

    spectrometer_type = spectrometer_type or station_setup.SPECTROMETER_TYPE
    resolved, spec_ip, spec_port, spec_uri = _spectrometer_config(spectrometer_type)
    component = None
    verification_status = "VERIFIED (readback)"
    try:
        if name == "smu":
            address = "simulated" if simulated else "USB0::0x2A8D::0x9201::MY61390757::INSTR"
            component = station_setup.KeysightB2902B(name, address=address, configure_defaults=False)
        elif name == "monochromator":
            component = ClientLightFieldMonochromator(
                name, server_ip=spec_ip, port=spec_port, pyro_uri=spec_uri,
                spectrometer_type=resolved, simulated=simulated,
            )
        elif name in ("rotator1", "rotator2", "rotator3"):
            address = {"rotator1": "0", "rotator2": "1", "rotator3": "2"}[name]
            component = station_setup.ThorlabsELL14(
                name, port="simulated" if simulated else "COM7", address=address
            )
        elif name == "camera":
            component = ClientLightFieldCamera(
                name, server_ip=spec_ip, port=spec_port, pyro_uri=spec_uri,
                spectrometer_type=resolved, simulated=simulated,
            )
        elif name == "piezo150":
            from device_discovery import find_attocube_address, occupied_serial_ports
            address = "simulated" if simulated else find_attocube_address(
                "ANC150", "COM10", skip_ports=occupied_serial_ports(station))
            component = station_setup.AttocubeANC150(
                name, address=address
            )
        elif name == "piezo300":
            from device_discovery import find_attocube_address, occupied_serial_ports
            address = "simulated" if simulated else find_attocube_address(
                "ANC300", "COM3", skip_ports=occupied_serial_ports(station))
            component = station_setup.AttocubeANC300(
                name, address=address
            )
        elif name == "magnet":
            component = station_setup.OxfordMagnet(
                name, address="simulated" if simulated else "GPIB0::25::INSTR"
            )
        elif name == "powermeter":
            component = (
                station_setup.ThorlabsPM100D(name, address="simulated")
                if simulated else _physical_powermeter()
            )
        elif name in ("lockin", "lockin2"):
            address = station_setup.LOCKIN_ADDRESS if name == "lockin" else station_setup.LOCKIN2_ADDRESS
            component = station_setup.StanfordSR830(name, address="simulated" if simulated else address)
        elif name == "awg":
            component = (station_setup.SiglentSDG2042X(name, address="simulated")
                         if simulated else _physical_awg())
        elif name == "dac":
            # Connecting never sends INITIALIZE; existing DAC voltages are retained.
            component = station_setup.QPGuVoltDAQ(
                name, address="simulated" if simulated else station_setup.DAC_ADDRESS)
        if name in ("piezo150", "piezo300") and not simulated:
            print(f"Identified {name} at {address}; existing voltages/modes are retained.")
        if name == "camera" and not simulated and isinstance(component.remote_lf, NIRvanaSocketWrapper):
            print(component.remote_lf.verify_acquisition())
        if not simulated:
            if getattr(component, "simulated", False):
                raise RuntimeError("Physical driver is unavailable; only simulation was created.")
            if name != "camera" or not isinstance(component.remote_lf, NIRvanaSocketWrapper):
                verification_status = verify_device(component, name)
        station.add_component(component, update_snapshot=False)
    except Exception:
        if component is not None:
            try:
                component.close()
            except Exception:
                pass
        else:
            try:
                Instrument.find_instrument(name).close()
            except Exception:
                pass
        raise

    if simulated:
        return "SIMULATED"
    if resolved == "lightfieldG11" and name == "camera":
        return "VERIFIED (2/2 spectra)"
    return verification_status


@contextmanager
def _readback_timeout(remote):
    """Bound connection checks without shortening subsequent acquisitions."""
    previous = remote._pyroTimeout
    remote._pyroTimeout = 5.0
    try:
        yield
    finally:
        remote._pyroTimeout = previous


def verify_device(component, name):
    """Read real device evidence without changing outputs or moving anything."""
    if getattr(component, "simulated", False):
        return "SIMULATED"
    if name == "camera":
        remote = component.remote_lf
        if isinstance(remote, NIRvanaSocketWrapper):
            print(remote.verify_acquisition())
            return "VERIFIED (2/2 spectra)"
        with _readback_timeout(remote):
            status = remote.get_status()
        if not status.get("success"):
            raise RuntimeError(status.get("error", "Camera status query failed"))
    elif name == "monochromator":
        from drivers.remote_client import NIRvanaMonoWrapper
        if isinstance(component.remote_lf, NIRvanaMonoWrapper):
            if not component.remote_lf.controls_enabled:
                print("Set wavelength/grating in LightField. The current server provides camera acquisition; remote wavelength control is not supported. Camera/live spectra are checked separately.")
                return "MANUAL (set in LightField)"
            component.wavelength()  # Read-only check on the separate control port.
        else:
            with _readback_timeout(component.remote_lf):
                component.wavelength()
    elif name.startswith("rotator"):
        # The public angle getter catches errors and returns zero; probe directly.
        with component.lock:
            float(component.device.get_angle())
    elif name == "piezo150":
        response = component.ask("getv 1")
        if "not in computer control mode" in response.lower():
            print("ANC150 is connected, but its front-panel computer-control mode is not enabled. No mode or voltage was changed.")
            return "MANUAL (enable computer control)"
        if "error" in response.lower() or "voltage" not in response.lower():
            raise RuntimeError(f"ANC150 did not return a voltage: {response!r}")
    elif name == "piezo300":
        for axis in (1, 2):
            state = component.get_axis_status(axis)
            print(f"ANC300 axis {axis}: offset {state['offset']:.3f} V, output {state['output']:.3f} V, mode {state['mode']}. Readback only; no voltage changed.")
    elif name == "magnet":
        component.field()
    elif name == "powermeter":
        component.power()
    elif name in ("lockin", "lockin2"):
        identity = component.IDN()
        print(f"{name}: {identity.get('vendor')} {identity.get('model')}, input "
              f"{component.input_config()}, reference {component.reference_source()} at "
              f"{component.frequency():.6g} Hz, sensitivity {component.sensitivity():g} "
              f"{component.sensitivity.unit}. Readback only; no settings changed.")
    elif name == "awg":
        identity = component.IDN()
        states = [f"CH{n}: {'ON' if s['enabled'] else 'OFF'} {s['wave_type']} {s['frequency']:.6g} Hz "
                  f"{s['amplitude']:g} Vpp phase {s['phase']:g} deg"
                  for n, s in ((n, getattr(component, f'ch{n}').state()) for n in (1, 2))]
        print(f"AWG: {identity.get('vendor')} {identity.get('model')}. " + "; ".join(states)
              + ". Readback only; outputs unchanged.")
    elif name == "dac":
        component.IDN()
        voltages = ", ".join(f"DAC{ch} {component.dac(ch).voltage():.6g} V" for ch in range(4))
        print(f"uVoltDAQ setpoints: {voltages}. Readback only; no voltage changed.")
    else:
        component.IDN()
    return "VERIFIED (readback)"


def remove_device(station, name):
    component = station.components.get(name)
    if component is not None:
        station.remove_component(name)
        component.close()


def ensure_simulated_devices(station):
    """Fill missing devices so every dashboard tab can open immediately."""
    for name in DEVICE_NAMES:
        if name not in station.components:
            initialize_device(station, name, simulated=True)


def configure_device(station, name, mode="Auto"):
    """Reconnect one idle device, with an explicit, traceable Auto fallback."""
    existing = station.components.get(name)
    simulated = mode == "Simulation"
    try:
        if existing is not None and bool(existing.simulated) == simulated:
            try:
                status = verify_device(existing, name)
            except Exception:
                if simulated:
                    raise
                # A USB device can reappear with a different port/handle. Replace
                # only this device and try physical discovery before Auto fallback.
                print(f"{name}: old connection failed; reconnecting this device.")
                remove_device(station, name)
                status = initialize_device(station, name, simulated=False)
        else:
            remove_device(station, name)
            status = initialize_device(station, name, simulated=simulated)
        return status
    except Exception as exc:
        if mode != "Auto":
            raise
        print(f"Physical connection failed: {type(exc).__name__}: {exc}")
        remove_device(station, name)
        initialize_device(station, name, simulated=True)
        print("Using simulation. Retry this device to reconnect the physical hardware.")
        return "SIMULATED (physical failed)"
