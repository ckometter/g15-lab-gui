"""Siglent SDG2000X (SDG2042X) basic-wave control for the capacitance bridge.

Uses the same SCPI commands as the qcodes-qpg ``SiglentSDG2042X`` driver
(``Cn:OUTP``, ``Cn:BSWV``), reduced to the controls a bridge needs: output
state, load, wave type, frequency, amplitude (Vpp), offset and phase. Queries
parse the instrument's key/value responses, e.g.
``C1:BSWV WVTP,SINE,FRQ,7777.7HZ,...,AMP,0.1V,...,OFST,0V,...,PHSE,0``.
"""
import re
import threading
import time

from qcodes.instrument import Instrument, InstrumentChannel, VisaInstrument
from qcodes.validators import Enum, Numbers

FREQUENCY_RANGE_HZ = (1e-6, 40e6)
AMPLITUDE_RANGE_VPP = (2e-3, 20.0)
OFFSET_RANGE_V = (-10.0, 10.0)
WAVE_TYPES = ("SINE", "SQUARE", "RAMP", "PULSE", "NOISE", "ARB", "DC", "PRBS", "IQ")

_NUMBER = re.compile(r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def parse_fields(response):
    """Split ``C1:BSWV KEY,VALUE,...`` into (standalone_first, {KEY: VALUE}).

    ``OUTP?`` replies begin with a standalone state (``ON``/``OFF``) before the
    key/value pairs; ``BSWV?`` replies are pairs only.
    """
    text = str(response).strip()
    _, _, body = text.partition(" ")
    items = [item.strip() for item in body.split(",")] if body else []
    first = None
    if len(items) % 2 == 1:
        first, items = items[0], items[1:]
    return first, {items[i].upper(): items[i + 1] for i in range(0, len(items) - 1, 2)}


def parse_number(value):
    """``7777.7HZ`` -> 7777.7, ``0.1V`` -> 0.1, ``-2.5e-3V`` -> -0.0025."""
    match = _NUMBER.match(str(value))
    if not match:
        raise ValueError(f"Siglent returned a non-numeric value: {value!r}")
    return float(match.group(1))


class SiglentSDGChannel(InstrumentChannel):
    def __init__(self, parent, name, channel_number):
        super().__init__(parent, name)
        self.channel_number = channel_number
        self._prefix = f"C{channel_number}:"
        if parent.simulated:
            self._sim = {"enabled": False, "load": "HZ", "wave_type": "SINE",
                         "frequency": 1000.0, "amplitude": 0.1, "offset": 0.0, "phase": 0.0}

        def bswv_parameter(name, key, unit, vals, label):
            self.add_parameter(
                name, label=label, unit=unit, vals=vals,
                get_cmd=lambda: self._get_bswv(key, name),
                set_cmd=lambda value: self._set_bswv(key, name, value))

        self.add_parameter("enabled", label="Output enabled",
                           get_cmd=self._get_enabled, set_cmd=self._set_enabled,
                           vals=Enum(True, False, 0, 1))
        self.add_parameter("load", label="Output load", unit="Ohm",
                           get_cmd=self._get_load, set_cmd=self._set_load)
        self.add_parameter("wave_type", label="Wave type", get_cmd=lambda: self._get_bswv("WVTP", "wave_type"),
                           set_cmd=lambda value: self._set_bswv("WVTP", "wave_type", str(value).upper()),
                           vals=Enum(*WAVE_TYPES, *(w.lower() for w in WAVE_TYPES)))
        bswv_parameter("frequency", "FRQ", "Hz", Numbers(*FREQUENCY_RANGE_HZ), "Frequency")
        bswv_parameter("amplitude", "AMP", "V", Numbers(*AMPLITUDE_RANGE_VPP), "Amplitude (peak-to-peak)")
        bswv_parameter("offset", "OFST", "V", Numbers(*OFFSET_RANGE_V), "DC offset")
        bswv_parameter("phase", "PHSE", "deg", Numbers(0.0, 360.0), "Phase")

    # -- SCPI ------------------------------------------------------------------
    def _query(self, command):
        return self.root_instrument.ask(self._prefix + command)

    def _write(self, command):
        self.root_instrument.write(self._prefix + command)

    def _get_enabled(self):
        if self.root_instrument.simulated:
            return bool(self._sim["enabled"])
        state, _ = parse_fields(self._query("OUTP?"))
        if state not in ("ON", "OFF"):
            raise RuntimeError(f"Unexpected Siglent output state: {state!r}")
        return state == "ON"

    def _set_enabled(self, value):
        if self.root_instrument.simulated:
            self._sim["enabled"] = bool(value)
        else:
            self._write(f"OUTP {'ON' if value else 'OFF'}")

    def _get_load(self):
        if self.root_instrument.simulated:
            return self._sim["load"]
        _, fields = parse_fields(self._query("OUTP?"))
        load = fields.get("LOAD", "")
        return load if load.upper() == "HZ" else parse_number(load)

    def _set_load(self, value):
        value = "HZ" if str(value).upper() in ("HZ", "HIGHZ", "INF") else int(value)
        if self.root_instrument.simulated:
            self._sim["load"] = value
        else:
            self._write(f"OUTP LOAD,{value}")

    def _get_bswv(self, key, name):
        if self.root_instrument.simulated:
            return self._sim[name]
        _, fields = parse_fields(self._query("BSWV?"))
        if key not in fields:
            raise RuntimeError(f"Siglent BSWV? reply has no {key} field.")
        return fields[key].upper() if key == "WVTP" else parse_number(fields[key])

    def _set_bswv(self, key, name, value):
        if self.root_instrument.simulated:
            self._sim[name] = value if key == "WVTP" else float(value)
        else:
            formatted = value if key == "WVTP" else format(float(value), ".10g")
            self._write(f"BSWV {key},{formatted}")

    def state(self):
        """Output and basic-wave settings with two queries (no writes)."""
        if self.root_instrument.simulated:
            return dict(self._sim)
        state, outp = parse_fields(self._query("OUTP?"))
        _, bswv = parse_fields(self._query("BSWV?"))
        load = outp.get("LOAD", "")
        return {
            "enabled": state == "ON",
            "load": load if load.upper() == "HZ" else parse_number(load),
            "wave_type": bswv.get("WVTP", "").upper(),
            "frequency": parse_number(bswv["FRQ"]),
            "amplitude": parse_number(bswv["AMP"]),
            "offset": parse_number(bswv.get("OFST", "0")),
            "phase": parse_number(bswv.get("PHSE", "0")),
        }


class SiglentSDG2042X(VisaInstrument):
    """Two-channel Siglent arbitrary waveform generator; ``address='simulated'`` for offline use."""

    def __init__(self, name, address, *, write_delay=0.05, **kwargs):
        self._lock = threading.RLock()
        self.simulated = address == "simulated"
        self.write_delay = float(write_delay)
        if self.simulated:
            Instrument.__init__(self, name)
        else:
            kwargs.setdefault("terminator", "\n")
            kwargs.setdefault("timeout", 5)
            super().__init__(name, address, **kwargs)
        for number in (1, 2):
            self.add_submodule(f"ch{number}", SiglentSDGChannel(self, f"ch{number}", number))
        if self.simulated:
            print(f"Initialized Siglent SDG2042X '{name}' in Simulation Mode.")
        else:
            self.connect_message()

    def write(self, cmd: str) -> None:
        if self.simulated:
            return
        with self._lock:
            super().write(cmd)
            # The SDG executes commands sequentially; back-to-back writes can be dropped.
            if self.write_delay > 0:
                time.sleep(self.write_delay)

    def ask(self, cmd: str) -> str:
        if self.simulated:
            raise RuntimeError("Raw SCPI queries are unavailable for the simulated AWG.")
        with self._lock:
            return super().ask(cmd)

    def excitation_state(self):
        """Cached state for the bridge model and metadata; never queries hardware."""
        state = {}
        for number in (1, 2):
            channel = getattr(self, f"ch{number}")
            for name in ("enabled", "frequency", "amplitude", "phase"):
                if self.simulated:
                    value = channel._sim[name]
                else:
                    value = channel.parameters[name].cache.get(get_if_invalid=False)
                state[f"ch{number}_{name}"] = value
        return state

    def get_idn(self) -> dict:
        if self.simulated:
            return {"vendor": "Siglent Technologies", "model": "SDG2042X",
                    "serial": "simulated", "firmware": "sim"}
        return super().get_idn()

    def snapshot_base(self, update=False, params_to_skip_update=None):
        if self.simulated:
            return Instrument.snapshot_base(self, update=update, params_to_skip_update=params_to_skip_update)
        return super().snapshot_base(update=update, params_to_skip_update=params_to_skip_update)

    def close(self) -> None:
        if getattr(self, "simulated", False):
            Instrument.close(self)
        else:
            super().close()
