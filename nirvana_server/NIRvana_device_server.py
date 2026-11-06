"""Run on the NIRvana PC alongside the EXISTING basi_automation_test.py.

65432: original acquisition protocol; 65433: versioned controls/live preview.
Importing this module never initializes hardware.
"""
import argparse
import json
from pathlib import Path
import socketserver
import threading
import time

try:
    from .NIRvana_control import LightFieldController, PROTOCOL
except ImportError:
    from NIRvana_control import LightFieldController, PROTOCOL


def read_request(connection):
    connection.settimeout(5)
    data = b""
    while len(data) <= 65536:
        chunk = connection.recv(4096)
        if not chunk:
            break
        data += chunk
        try:
            value = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(value, dict):
            raise ValueError("Request must be a JSON object.")
        return value
    raise ValueError("Incomplete or oversized JSON request.")


class Server(socketserver.ThreadingTCPServer):
    # Keep Windows' exclusive binding; two automation servers must not share a port.
    daemon_threads = False

    def __init__(self, address, controller, controls=False):
        self.controller, self.controls = controller, controls
        super().__init__(address, Handler)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            request = read_request(self.request)
            self.request.settimeout(15)
            if self.server.controls:
                result = self.server.controller.command(request)
                self.send_json({**result, "success": True, "protocol": PROTOCOL})
            else:
                self.server.controller.acquire_csv(request, self.request.sendall)
        except Exception as exc:
            print("Request from {} failed: {}".format(self.client_address, exc), flush=True)
            try:
                if self.server.controls:
                    self.send_json({"success": False, "protocol": PROTOCOL, "error": str(exc)})
                else:
                    self.request.sendall(("ERROR: " + str(exc)).encode("utf-8"))
            except OSError:
                pass

    def send_json(self, value):
        self.request.sendall(json.dumps(value, allow_nan=False).encode("utf-8") + b"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=65432)
    parser.add_argument("--control-port", type=int, default=65433)
    parser.add_argument("--export-dir", default=str(Path(__file__).resolve().parent / "Data"))
    args = parser.parse_args()
    # Bind BOTH ports before creating automation, so a running old server is detected.
    with Server((args.host, args.port), None) as acquisition, Server(
            (args.host, args.control_port), None, controls=True) as controls:
        import basi_automation_test as automation
        from System import Double, Int64
        from PrincetonInstruments.LightField.AddIns import SpectrometerSettings, CameraSettings, ExperimentSettings
        threads = []
        try:
            automation.initialize_lightfield(export_dir=args.export_dir)
            controller = LightFieldController(automation.experiment, SpectrometerSettings,
                                             CameraSettings, ExperimentSettings, args.export_dir,
                                             double=Double, integer=Int64)
            acquisition.controller = controls.controller = controller
            for server in (acquisition, controls):
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                threads.append((server, thread))
            print("NIRvana acquisition on {}; controls on {}. Ctrl+C to stop.".format(
                args.port, args.control_port), flush=True)
            while True:
                controller.expire_preview()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Stopping server; waiting for active requests to finish.")
        finally:
            for server, thread in threads:
                server.shutdown()
                thread.join()
                server.server_close()
            if threads and controller.preview_token:
                controller._stop_preview()
            automation.dispose_lightfield()


if __name__ == "__main__":
    main()
