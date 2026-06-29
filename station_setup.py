import os
import sys
import warnings

# Suppress harmless PyVISA attribute state warnings (e.g. VI_WARN_NSUP_ATTR_STATE)
try:
    from pyvisa.errors import VisaIOWarning
    warnings.filterwarnings("ignore", category=VisaIOWarning)
except ImportError:
    pass

import qcodes as qc
from qcodes.dataset import initialise_or_create_database_at
from qcodes.instrument import Instrument

# Add current folder to sys.path to ensure local drivers package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import custom drivers from the local drivers package
from drivers.keysight_b2902b import KeysightB2902B
from drivers.acton_spectrapro import ActonSpectraPro750
from drivers.thorlabs_ell14 import ThorlabsELL14
from drivers.princeton_pylon import PrincetonPylon
from drivers.attocube_anc150 import AttocubeANC150
from drivers.attocube_anc300 import AttocubeANC300
from drivers.oxford_magnet import OxfordMagnet
from drivers.thorlabs_pm100d import ThorlabsPM100D

# Spectrometer configuration switch
# Options: 'lightfieldG15', 'wingspecG15', 'lightfieldG11'
SPECTROMETER_TYPE = 'wingspecG15'


# Configuration details for each spectrometer
SPECTROMETER_CONFIGS = {
    'lightfieldG15': {
        'server_ip': 'G15-Pylon.dhcp-int.phys.ethz.ch',
        'port': 50920,
        'pyro_uri': None  # None means construct from server_ip & port
    },
    'wingspecG15': {
        'server_ip': None,
        'port': None,
        'pyro_uri': 'PYRO:WinSpec@phd-exile-phys.ethz.ch:9091'
    },
    'lightfieldG11': {
        'server_ip': "10.236.10.143",  # this is the Nirvana computer --- might need to update if device PC reboots
        'port': 65432,
        'pyro_uri': None
    }
}

# Default dictionary of simulated devices
simulated_devices = {
    'smu': False,
    'monochromator': False,
    'rotator1': False,
    'rotator2': True,
    'rotator3': True,
    'camera': False,
    'piezo150': True,
    'piezo300': True,
    'magnet': True,
    'powermeter': False
}

def initialize_station(simulated: bool = False, simulated_devices: dict = None, db_path: str = "my_measurement_data.db", spectrometer_type: str = SPECTROMETER_TYPE) -> qc.Station:
    """
    Close existing instrument instances to avoid name conflict errors, connect to the database,
    and initialize all station components (both physical and simulated).
    """
    # Close any existing instrument instances in memory to avoid name conflict errors (e.g. KeyError for 'smu')
    Instrument.close_all()
    
    db_abs_path = os.path.abspath(db_path)
    initialise_or_create_database_at(db_abs_path)
    print(f"QCoDeS database set to: {db_abs_path}")

    station = qc.Station()

    if simulated_devices is None:
        simulated_devices = globals()['simulated_devices'].copy()

    def should_simulate(device_name: str) -> bool:
        if simulated:
            return True
        return simulated_devices.get(device_name, True)

    # 1. Add Keysight B2902B Source
    if not should_simulate('smu'):
        try:
            smu_addr = 'USB0::0x2A8D::0x9201::MY61390757::INSTR'
            smu = KeysightB2902B('smu', address=smu_addr)
        except Exception as e:
            print(f"Warning: Could not connect to physical Keysight SMU at {smu_addr} ({e}). Falling back to simulation.")
            smu = KeysightB2902B('smu', address='simulated')
    else:
        smu = KeysightB2902B('smu', address='simulated')
    station.add_component(smu)

    # Retrieve configuration for selected spectrometer
    spec_cfg = SPECTROMETER_CONFIGS.get(spectrometer_type, SPECTROMETER_CONFIGS['lightfieldG15'])
    spec_ip = spec_cfg.get('server_ip') or 'G15-Pylon.dhcp-int.phys.ethz.ch'
    spec_port = spec_cfg.get('port') or 50920
    spec_uri = spec_cfg.get('pyro_uri')

    # 2. Add Acton SpectraPro 750 Monochromator (Strictly Physical Remote Connection)
    from drivers.remote_client import ClientLightFieldMonochromator
    monochromator = ClientLightFieldMonochromator('monochromator', server_ip=spec_ip, port=spec_port, pyro_uri=spec_uri, spectrometer_type=spectrometer_type)
    station.add_component(monochromator)

    # 3. Add Thorlabs ELL14 Rotators
    # Rotator 1 (address '0')
    if not should_simulate('rotator1'):
        try:
            ell_port1 = 'COM7'
            rotator1 = ThorlabsELL14('rotator1', port=ell_port1, address='0')
        except Exception as e:
            print(f"Warning: Could not connect to physical Rotator 1 at {ell_port1} address '0' ({e}). Falling back to simulation.")
            rotator1 = ThorlabsELL14('rotator1', port='simulated', address='0')
    else:
        rotator1 = ThorlabsELL14('rotator1', port='simulated', address='0')
    station.add_component(rotator1)

    # Rotator 2 (address '1')
    if not should_simulate('rotator2'):
        try:
            ell_port2 = 'COM7'
            rotator2 = ThorlabsELL14('rotator2', port=ell_port2, address='1')
        except Exception as e:
            print(f"Warning: Could not connect to physical Rotator 2 at {ell_port2} address '1' ({e}). Falling back to simulation.")
            rotator2 = ThorlabsELL14('rotator2', port='simulated', address='1')
    else:
        rotator2 = ThorlabsELL14('rotator2', port='simulated', address='1')
    station.add_component(rotator2)

    # Rotator 3 (address '2')
    if not should_simulate('rotator3'):
        try:
            ell_port3 = 'COM7'
            rotator3 = ThorlabsELL14('rotator3', port=ell_port3, address='2')
        except Exception as e:
            print(f"Warning: Could not connect to physical Rotator 3 at {ell_port3} address '2' ({e}). Falling back to simulation.")
            rotator3 = ThorlabsELL14('rotator3', port='simulated', address='2')
    else:
        rotator3 = ThorlabsELL14('rotator3', port='simulated', address='2')
    station.add_component(rotator3)

    # 4. Add Princeton Pylon CCD Camera (Strictly Physical Remote Connection)
    from drivers.remote_client import ClientLightFieldCamera
    camera = ClientLightFieldCamera('camera', server_ip=spec_ip, port=spec_port, pyro_uri=spec_uri, spectrometer_type=spectrometer_type)
    station.add_component(camera)

    # 5. Add Attocube ANC150 (adjustment-only stage)
    if not should_simulate('piezo150'):
        try:
            anc150_addr = 'ASRL10::INSTR'
            piezo150 = AttocubeANC150('piezo150', address=anc150_addr)
        except Exception as e:
            print(f"Warning: Could not connect to physical Attocube ANC150 at {anc150_addr} ({e}). Falling back to simulation.")
            piezo150 = AttocubeANC150('piezo150', address='simulated')
    else:
        piezo150 = AttocubeANC150('piezo150', address='simulated')
    station.add_component(piezo150)

    # 6. Add Attocube ANC300 (piezo sweep controller)
    if not should_simulate('piezo300'):
        try:
            piezo300 = AttocubeANC300('piezo300', address='ASRL3::INSTR')
        except Exception as e:
            print(f"Warning: Could not connect to physical Attocube ANC300 at ASRL3::INSTR ({e}). Falling back to simulation.")
            piezo300 = AttocubeANC300('piezo300', address='simulated')
    else:
        piezo300 = AttocubeANC300('piezo300', address='simulated')
    station.add_component(piezo300)

    # 7. Add OxfordMagnet
    if not should_simulate('magnet'):
        try:
            magnet = OxfordMagnet('magnet', address='GPIB0::25::INSTR')
        except Exception as e:
            print(f"Warning: Could not connect to physical OxfordMagnet at GPIB0::25::INSTR ({e}). Falling back to simulation.")
            magnet = OxfordMagnet('magnet', address='simulated')
    else:
        magnet = OxfordMagnet('magnet', address='simulated')
    station.add_component(magnet)

    # 8. Add Thorlabs PM100D Powermeter
    if not should_simulate('powermeter'):
        try:
            pm_addr = 'USB0::0x1313::0x8078::P0013460::INSTR'
            powermeter = ThorlabsPM100D('powermeter', address=pm_addr)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"Warning: Could not connect to physical Thorlabs PM100D at {pm_addr} ({e}). Falling back to simulation.")
            powermeter = ThorlabsPM100D('powermeter', address='simulated')
    else:
        powermeter = ThorlabsPM100D('powermeter', address='simulated')
    station.add_component(powermeter)

    return station
