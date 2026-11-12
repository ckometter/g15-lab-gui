"""A stand-in for the lab's ANC150 serial port (firmware 1.4), for offline tests.

It answers like the real controller: the echoed command, a result line, OK or
ERROR, then the "> " prompt, after about 25 ms per command (measured on the lab
PC). Counted steps take steps / frequency seconds; a step command that arrives
while an axis is still stepping is counted in ``overlaps``.
"""
import threading
import time


class FakeANC150Port:
    def __init__(self, frequencies=(102, 43, 52), voltages=(58, 56, 60), latency=0.025):
        self.latency = latency
        self.mode = {1: "gnd", 2: "gnd", 3: "gnd"}
        self.frequency = dict(zip((1, 2, 3), frequencies))
        self.voltage = dict(zip((1, 2, 3), voltages))
        self.steps = {1: 0, 2: 0, 3: 0}
        self.busy_until = {1: 0.0, 2: 0.0, 3: 0.0}
        self.overlaps = 0
        self.commands = []
        self._output = bytearray()
        self._lock = threading.Lock()

    def write(self, command):
        time.sleep(self.latency)
        with self._lock:
            self.commands.append(command)
            words, body, ok = command.split(), "", True
            try:
                name, axis = words[0], int(words[1])
                if name == "getm":
                    body = f"mode = {self.mode[axis]}\r\n"
                elif name == "getf":
                    body = f"frequency = {self.frequency[axis]} Hz\r\n"
                elif name == "getv":
                    body = f"voltage = {self.voltage[axis]} V\r\n"
                elif name == "setm" and words[2] in ("gnd", "stp", "cap"):
                    self.mode[axis] = words[2]
                elif name in ("stepu", "stepd") and self.mode[axis] == "stp":
                    count = int(words[2])
                    now = time.monotonic()
                    if now < self.busy_until[axis]:
                        self.overlaps += 1
                    self.steps[axis] += count if name == "stepu" else -count
                    self.busy_until[axis] = now + count / self.frequency[axis]
                elif name == "stop":
                    self.busy_until[axis] = 0.0
                else:
                    ok = False
            except (IndexError, KeyError, ValueError):
                ok = False
            self._output += f"{command}\r\n{body}{'OK' if ok else 'ERROR'}\r\n> ".encode("ascii")

    def read_bytes(self, count):
        with self._lock:
            data = bytes(self._output[:count])
            del self._output[:count]
        return data


def connect_fake(stepper, port):
    """Make a simulated AttocubeANC150 talk to ``port`` through the real serial code paths."""
    stepper.simulated = False
    stepper._visa_handle = port  # qcodes 0.59: VisaInstrument.visa_handle returns _visa_handle
    return stepper
