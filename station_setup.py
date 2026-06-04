import os
import sys
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

def initialize_station(simulated: bool = True, simulated_devices: dict = None, db_path: str = "my_measurement_data.db") -> qc.Station:
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
        simulated_devices = {
            'smu': True,
            'monochromator': True,
            'rotator1': True,
            'rotator2': True,
            'camera': True,
            'piezo150': True,
            'piezo300': True
        }

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

    # 2. Add Acton SpectraPro 750 Monochromator
    if not should_simulate('monochromator'):
        try:
            mono_addr = 'ASRL3::INSTR'
            monochromator = ActonSpectraPro750('monochromator', address=mono_addr)
        except Exception as e:
            print(f"Warning: Could not connect to physical Monochromator at {mono_addr} ({e}). Falling back to simulation.")
            monochromator = ActonSpectraPro750('monochromator', address='simulated')
    else:
        monochromator = ActonSpectraPro750('monochromator', address='simulated')
    station.add_component(monochromator)

    # 3. Add Thorlabs ELL14 Rotators
    # Rotator 1 (address '0')
    if not should_simulate('rotator1'):
        try:
            ell_port1 = 'COM4'
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
            ell_port2 = 'COM4'
            rotator2 = ThorlabsELL14('rotator2', port=ell_port2, address='1')
        except Exception as e:
            print(f"Warning: Could not connect to physical Rotator 2 at {ell_port2} address '1' ({e}). Falling back to simulation.")
            rotator2 = ThorlabsELL14('rotator2', port='simulated', address='1')
    else:
        rotator2 = ThorlabsELL14('rotator2', port='simulated', address='1')
    station.add_component(rotator2)

    # 4. Add Princeton Pylon CCD Camera
    if not should_simulate('camera'):
        try:
            camera = PrincetonPylon('camera', camera_id=0)
        except Exception as e:
            print(f"Warning: Could not connect to physical Princeton Pylon CCD Camera ({e}). Falling back to simulation.")
            camera = PrincetonPylon('camera', camera_id='simulated')
    else:
        camera = PrincetonPylon('camera', camera_id='simulated')
    station.add_component(camera)

    # 5. Add Attocube ANC150 (adjustment-only stage)
    if not should_simulate('piezo150'):
        try:
            anc150_addr = 'ASRL5::INSTR'
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
            piezo300 = AttocubeANC300('piezo300', address='ASRL6::INSTR')
        except Exception as e:
            print(f"Warning: Could not connect to physical Attocube ANC300 at ASRL6::INSTR ({e}). Falling back to simulation.")
            piezo300 = AttocubeANC300('piezo300', address='simulated')
    else:
        piezo300 = AttocubeANC300('piezo300', address='simulated')
    station.add_component(piezo300)

    return station
