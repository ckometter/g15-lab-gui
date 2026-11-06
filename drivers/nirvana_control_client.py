"""Short-lived JSON control connections; never probe the legacy acquisition port."""
import json
import socket

PROTOCOL = "nirvana-control-v1"


class NIRvanaControlClient:
    def __init__(self, server_ip, port=65433):
        self.server_ip, self.port = server_ip, int(port)

    def request(self, command, *, timeout=3.0, **values):
        sent = False
        try:
            with socket.create_connection((self.server_ip, self.port), timeout=timeout) as conn:
                conn.settimeout(timeout)
                conn.sendall(json.dumps({"command": command, **values}, allow_nan=False).encode("utf-8") + b"\n")
                sent = True
                chunks = bytearray()
                while True:
                    data = conn.recv(65536)
                    if not data:
                        break
                    chunks.extend(data)
                    if len(chunks) > 16 * 1024 * 1024:
                        raise ValueError("Oversized control response")
                    if chunks.endswith(b"\n"):
                        break
            response = json.loads(chunks.decode("utf-8"))
            if not isinstance(response, dict) or response.get("protocol") != PROTOCOL:
                raise ValueError("Unrecognized NIRvana control protocol")
            if not response.get("success"):
                raise RuntimeError(response.get("error", "LightField command failed"))
            return response
        except (OSError, ValueError) as exc:
            uncertain = (" The command may have reached LightField; check its readback before retrying."
                         if sent and command.startswith("set_") else "")
            raise RuntimeError("NIRvana controls at {}:{}: {}.{}".format(
                self.server_ip, self.port, exc, uncertain)) from exc

    def get_status(self):
        return self.request("get_status")

    def get_wavelength(self):
        status = self.get_status()
        if status.get("wl") is None:
            raise RuntimeError("No wavelength readback in the loaded LightField experiment.")
        return float(status["wl"])

    def set_wavelength(self, value):
        return self.request("set_wavelength", value=float(value), timeout=120)

    def set_grating(self, value):
        return self.request("set_grating", value=str(value), timeout=120)
