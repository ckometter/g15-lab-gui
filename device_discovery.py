"""Read-only serial identity discovery. No output, mode or movement commands."""
import re
import time

import serial
from serial.tools import list_ports

_known_ports = {}


def serial_inventory():
    return [{'port': port.device, 'description': port.description,
             'hwid': port.hwid, 'usb': port.vid is not None}
            for port in sorted(list_ports.comports(), key=lambda port: port.device)]


def identify_attocube(port, timeout=0.7):
    connection = serial.Serial(port=None, baudrate=38400, timeout=0.05, write_timeout=0.5)
    connection.dtr = False
    connection.rts = False
    connection.port = port
    try:
        connection.open()  # An in-use port fails; never close another application's handle.
        connection.write(b'ver\r')
        response = bytearray()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and len(response) < 16384:
            response.extend(connection.read(connection.in_waiting or 1))
            if b'OK\r\n' in response or b'ERROR\r\n' in response:
                break
        text = response.decode('ascii', errors='replace')
        # ANC300 also prints "ANC150 compatibility console". Match its identity,
        # not an arbitrary occurrence of ANC150 in the firmware banner.
        match = re.search(r'attocube\s+(ANC(?:150|300))\b', text, flags=re.I)
        if match:
            return {'port': port, 'model': match[1].upper(), 'identity': text.strip()}
        if re.search(r'attocube controller version 1\.4\b', text, re.I) and re.search(r'serial number \d+', text, re.I):
            # The legacy ANC150 firmware reports a generic controller banner.
            return {'port': port, 'model': 'ANC150', 'identity': text.strip(), 'legacy_banner': True}
        return None
    finally:
        connection.close()


def scan_attocubes(*, skip_ports=(), identify=True):
    inventory = serial_inventory()
    found, errors = [], {}
    if identify:
        for item in inventory:
            port = item['port']
            if port in skip_ports or not item['usb']:
                continue
            try:
                result = identify_attocube(port)
                if result:
                    found.append(result)
                    _known_ports[result['model']] = port
            except (OSError, serial.SerialException) as exc:
                errors[port] = str(exc)
    return {'ports': inventory, 'controllers': found, 'errors': errors,
            'identity_probed': identify}


def find_attocube_address(model, preferred_port, *, skip_ports=()):
    inventory = serial_inventory()
    available = {item['port']: item for item in inventory}
    candidates = list(dict.fromkeys([_known_ports.get(model), preferred_port]
                                    + [item['port'] for item in inventory if item['usb']]))
    errors = []
    for port in candidates:
        if not port or port not in available or port in skip_ports:
            continue
        try:
            identity = identify_attocube(port)
            if identity and identity['model'] == model:
                _known_ports[model] = port
                return 'ASRL{}::INSTR'.format(int(port.removeprefix('COM')))
        except (OSError, serial.SerialException) as exc:
            errors.append(f'{port}: {exc}')
    raise RuntimeError(f'{model} was not identified on the available serial ports. '
                       + ('; '.join(errors) or 'Check its USB cable and power.'))


def occupied_serial_ports(station):
    ports = set()
    for component in station.components.values():
        if getattr(component, 'simulated', False):
            continue
        handle = getattr(component, 'visa_handle', None)
        resource = str(getattr(handle, 'resource_name', ''))
        match = re.match(r'ASRL(\d+)::', resource, flags=re.I)
        if match:
            ports.add('COM' + match[1])
    return ports
