from qcodes.instrument import VisaInstrument, Instrument

class ActonSpectraPro750(VisaInstrument):
    """
    QCoDeS driver for the Acton SpectraPro 750 Monochromator/Spectrograph.
    Communicates via RS-232 serial.
    """
    def __init__(self, name, address, **kwargs):
        # Remove and close any existing instrument with the same name to avoid collision
        try:
            instr = Instrument.find_instrument(name)
            try:
                Instrument.remove_instance(instr)
            except Exception:
                pass
            try:
                instr.close()
            except Exception:
                pass
        except KeyError:
            pass

        self.simulated = (address == 'simulated')
        
        if self.simulated:
            Instrument.__init__(self, name, **kwargs)
            self._wavelength = 500.0
        else:
            # Acton monochromators typically use:
            # Baud rate: 9600, Data bits: 8, Stop bits: 1, Parity: None
            # Terminator is carriage return '\r'
            super().__init__(name, address, terminator='\r', **kwargs)
            
        self.add_parameter('wavelength',
                           label='Wavelength',
                           unit='nm',
                           get_cmd=self._get_wavelength,
                           set_cmd=self._set_wavelength)
        
        if self.simulated:
            print(f"Initialized Acton SpectraPro 750 '{name}' in Simulation Mode.")
        else:
            self.connect_message()

    def snapshot_base(self, update = False, params_to_skip_update = None) -> dict: #    snap = station.monochromator.snapshot(update=True), print(json.dumps(snap, indent=2))
        if self.simulated:
            return Instrument.snapshot_base(self, update=update, params_to_skip_update=params_to_skip_update)
        return super().snapshot_base(update=update, params_to_skip_update=params_to_skip_update)

    def get_idn(self) -> dict:
        return {
            'vendor': 'Acton',
            'model': 'SpectraPro 750',
            'serial': 'simulated' if self.simulated else 'unknown',
            'firmware': None
        }

    def close(self) -> None:
        if getattr(self, 'simulated', False):
            Instrument.close(self)
        else:
            try:
                super().close()
            except Exception:
                self.remove_instance(self)


    def _get_wavelength(self) -> float:
        if self.simulated:
            return self._wavelength
        
        res = self.ask('?NM')
        # Parse output like '  500.00 nm' or '500.00\rok' or just '500.00'
        clean_res = res.replace('nm', '').replace('ok', '').strip()
        parts = clean_res.split()
        if parts:
            try:
                return float(parts[0])
            except ValueError:
                pass
        raise ValueError(f"Could not parse wavelength response: {res}")

    def _set_wavelength(self, val: float) -> None:
        if self.simulated:
            self._wavelength = val
        else:
            # Acton command to change wavelength is '<value> GOTO'
            self.write(f"{val:.3f} GOTO")
