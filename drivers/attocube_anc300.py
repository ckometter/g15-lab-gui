from qcodes.instrument import Instrument

try:
    from qcodes_contrib_drivers.drivers.Attocube.ANC300 import ANC300 as PhysicalANC300
except ImportError:
    PhysicalANC300 = None

if PhysicalANC300 is not None:
    class AttocubeANC300(PhysicalANC300):
        """
        QCoDeS driver for the Attocube ANC300 Piezo Controller.
        Integrates physical driver (from qcodes_contrib_drivers) and simulation fallback.
        """
        def __init__(self, name, address, **kwargs):
            # Close any existing instrument with the same name to avoid collision
            try:
                instr = Instrument.find_instrument(name)
                instr.close()
            except KeyError:
                pass
            except Exception:
                try:
                    Instrument.remove_instance(instr)
                except Exception:
                    pass

            self.simulated = (address == 'simulated')
            if self.simulated:
                # Bypass physical VisaInstrument initialization
                Instrument.__init__(self, name, **kwargs)
                self._voltages = {1: 0.0, 2: 0.0, 3: 0.0}
                for axis in [1, 2, 3]:
                    self.add_parameter(f'voltage_ch{axis}',
                                       label=f'Axis {axis} Sweep Voltage',
                                       unit='V',
                                       get_cmd=lambda a=axis: self.get_voltage(a),
                                       set_cmd=lambda val, a=axis: self.set_voltage(a, val))
                print(f"Initialized Attocube ANC300 '{name}' in Simulation Mode.")
            else:
                super().__init__(name, address, **kwargs)
                # Disable snapshot queries for trigger parameters to avoid hardware warning errors
                for ax_name in ['axis1', 'axis2', 'axis3']:
                    ax = getattr(self, ax_name, None)
                    if not ax and ax_name in self.submodules:
                        ax = self.submodules[ax_name]
                    if ax and hasattr(ax, 'parameters'):
                        for p_name in ['triggerUp', 'triggerDown']:
                            if p_name in ax.parameters:
                                ax.parameters[p_name].snapshot_get = False
                                ax.parameters[p_name]._snapshot_get = False
                for submod in self.submodules.values():
                    if hasattr(submod, 'parameters'):
                        for p_name in ['triggerUp', 'triggerDown']:
                            if p_name in submod.parameters:
                                submod.parameters[p_name].snapshot_get = False
                                submod.parameters[p_name]._snapshot_get = False

        def _open_resource(self, address: str, visalib: str | None):
            resource, visabackend, resource_manager = super()._open_resource(address, visalib)
            if 'ASRL' in address:
                resource.baud_rate = 38400
            return resource, visabackend, resource_manager

        def get_voltage(self, axis: int) -> float:
            """Get fine scanning DC offset voltage for the given axis (1, 2, or 3)."""
            if self.simulated:
                return self._voltages.get(axis, 0.0)
            else:
                # physical mode: check if native driver axis voltage parameter exists (geto)
                axis_obj = getattr(self, f'axis{axis}', None)
                if axis_obj and hasattr(axis_obj, 'voltage'):
                    return float(axis_obj.voltage())
                # fallback parameter
                param_name = f'voltage_ch{axis}'
                if hasattr(self, param_name):
                    return float(getattr(self, param_name)())
                return 0.0

        def set_voltage(self, axis: int, val: float) -> None:
            """Set fine scanning DC offset voltage for the given axis (1, 2, or 3)."""
            if self.simulated:
                self._voltages[axis] = val
            else:
                # physical mode: check if native driver axis offset parameter exists
                axis_obj = getattr(self, f'axis{axis}', None)
                if axis_obj and hasattr(axis_obj, 'offset'):
                    axis_obj.offset(val)
                else:
                    param_name = f'voltage_ch{axis}'
                    if hasattr(self, param_name):
                        getattr(self, param_name)(val)

        def _set_voltage(self, val: float) -> None:
            # Backward compatibility method (acts on channel 1)
            self.set_voltage(1, val)

        def close(self) -> None:
            if getattr(self, 'simulated', False):
                Instrument.close(self)
            else:
                try:
                    super().close()
                except Exception:
                    self.remove_instance(self)

        def get_idn(self) -> dict:
            if getattr(self, 'simulated', False):
                return {
                    'vendor': 'Attocube',
                    'model': 'ANC300',
                    'serial': 'simulated',
                    'firmware': None
                }
            try:
                return super().get_idn()
            except Exception:
                return {
                    'vendor': 'Attocube',
                    'model': 'ANC300',
                    'serial': 'unknown',
                    'firmware': None
                }
else:
    class AttocubeANC300(Instrument):
        """
        Simulated mockup driver fallback for the Attocube ANC300 Piezo Controller.
        """
        def __init__(self, name, address=None, **kwargs):
            # Close any existing instrument with the same name to avoid collision
            try:
                instr = Instrument.find_instrument(name)
                instr.close()
            except KeyError:
                pass
            except Exception:
                try:
                    Instrument.remove_instance(instr)
                except Exception:
                    pass

            super().__init__(name, **kwargs)
            self.simulated = True
            self._voltages = {1: 0.0, 2: 0.0, 3: 0.0}
            for axis in [1, 2, 3]:
                self.add_parameter(f'voltage_ch{axis}',
                                   label=f'Axis {axis} Sweep Voltage',
                                   unit='V',
                                   get_cmd=lambda a=axis: self.get_voltage(a),
                                   set_cmd=lambda val, a=axis: self.set_voltage(a, val))
            print(f"Initialized Attocube ANC300 '{name}' in Simulation Mode (fallback).")

        def get_voltage(self, axis: int) -> float:
            """Get fine scanning DC offset voltage for the given axis (1, 2, or 3)."""
            return self._voltages.get(axis, 0.0)

        def set_voltage(self, axis: int, val: float) -> None:
            """Set fine scanning DC offset voltage for the given axis (1, 2, or 3)."""
            self._voltages[axis] = val

        def _set_voltage(self, val: float) -> None: #internal helper function
            # Backward compatibility method (acts on channel 1)
            self.set_voltage(1, val)

        def close(self) -> None:
            if getattr(self, 'simulated', False):
                Instrument.close(self)
            else:
                try:
                    super().close()
                except Exception:
                    self.remove_instance(self)

        def get_idn(self) -> dict:
            return {
                'vendor': 'Attocube',
                'model': 'ANC300',
                'serial': 'simulated',
                'firmware': None
            }

