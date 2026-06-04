import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import qcodes as qc
from qcodes.dataset import Measurement, new_experiment, plot_dataset
from qcodes.parameters import Parameter
from qcodes.instrument import Instrument

# Add parent folder to path to import station_setup
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from station_setup import initialize_station

class Lambda0Intensity(Parameter):
    """
    Custom QCoDeS parameter that captures a 2D frame from the Princeton camera
    and extracts the intensity at target_lambda using the general lookup logic.
    """
    def __init__(self, camera, monochromator=None, target_lambda=None, name="intensity", **kwargs):
        super().__init__(name=name,
                         label="Intensity at lambda_0",
                         unit="counts",
                         **kwargs)
        self.camera = camera
        self.monochromator = monochromator
        self.target_lambda = target_lambda

    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            self.target_lambda = args[0]
            return self.get()
        elif len(args) > 1:
            raise ValueError("Lambda0Intensity parameter call only accepts at most 1 positional argument (target_lambda).")
        return self.get()

    def get_raw(self) -> float:
        # Retrieve monochromator setting dynamically if not provided
        mono = self.monochromator
        if mono is None:
            try:
                mono = Instrument.find_instrument('monochromator')
            except Exception:
                pass
                
        t_lambda = self.target_lambda
        if t_lambda is None:
            if mono is not None:
                t_lambda = mono.wavelength()
            else:
                t_lambda = 500.0
                
        if mono is not None:
            lambda_mono = mono.wavelength()
        else:
            lambda_mono = t_lambda
            
        # Capture the image
        img = self.camera.image()
        spectrum_1d = np.sum(img, axis=0)
        
        # Calibrate wavelength axis
        dispersion = 0.05
        center_pixel = len(spectrum_1d) // 2
        pixels = np.arange(len(spectrum_1d))
        wavelengths = lambda_mono + (pixels - center_pixel) * dispersion
        
        # Retrieve intensity using the general lookup logic
        idx = np.argmin(np.abs(wavelengths - t_lambda))
        return float(spectrum_1d[idx])


def run_double_gate_sweep(station, lambda_0: float = 500.0, sample_name: str = "gate_sweep_sample"):
    """
    Runs a 2D electrostatic gate sweep experiment:
    1. Sets the monochromator wavelength to lambda_0.
    2. Sweeps Top Gate (smu.ch1.volt) and Bottom Gate (smu.ch2.volt).
    3. Measures the CCD intensity at lambda_0 at each voltage step.
    4. Saves results to the database and plots the 2D map.
    """
    print(f"\n--- Starting 2D Gate Sweep (lambda_0 = {lambda_0} nm) ---")
    
    monochromator = station.monochromator
    smu = station.smu
    camera = station.camera
    
    # Set the monochromator wavelength
    print(f"Setting monochromator to lambda_0 = {lambda_0} nm...")
    monochromator.wavelength(lambda_0)
    if hasattr(camera, 'source_wavelength'):
        camera.source_wavelength(lambda_0)
    
    intensity = Lambda0Intensity(camera=camera, monochromator=monochromator, target_lambda=lambda_0)
    
    # Set up QCoDeS dataset
    exp = new_experiment(name="gate_characterization", sample_name=sample_name)
    meas = Measurement(exp=exp)
    
    # Register parameters
    meas.register_parameter(smu.ch1.volt)  # Top Gate
    meas.register_parameter(smu.ch2.volt)  # Bottom Gate
    meas.register_parameter(intensity, setpoints=(smu.ch1.volt, smu.ch2.volt))
    
    smu.ch1.output(True)
    smu.ch2.output(True)
    
    # Define sweep ranges: 5mV to -5mV in 21 steps for both gates
    top_gate_sweep = np.linspace(0.005, -0.005, 21)
    bottom_gate_sweep = np.linspace(0.005, -0.005, 21)
    
    # Execute the 2D sweep loop
    with meas.run() as datasaver:
        for v_top in top_gate_sweep:
            smu.ch1.volt(v_top)  # Apply voltage to Top Gate
            
            for v_bot in bottom_gate_sweep:
                smu.ch2.volt(v_bot)
                
                val = intensity()
                
                datasaver.add_result(
                    (smu.ch1.volt, v_top),
                    (smu.ch2.volt, v_bot),
                    (intensity, val)
                )
                
    smu.ch1.output(False)
    smu.ch2.output(False)
    
    print(f"Sweep complete. Saved as Run ID: {datasaver.run_id}")
    
    # Plot the results
    print("Generating 2D plot...")
    plot_dataset(datasaver.dataset)
    
    # Save a copy of the plot if run directly as a script
    if __name__ == '__main__':
        output_dir = os.path.dirname(os.path.abspath(__file__))
        plot_file = os.path.join(output_dir, "double_gate_sweep.png")
        plt.savefig(plot_file)
        print(f"Saved plot to {plot_file}")
        
    plt.show()
    return datasaver.run_id

if __name__ == '__main__':
    print("=== Running double_gate_sweep in Simulated Mode ===")
    station = initialize_station(simulated=True)
    run_double_gate_sweep(station, lambda_0=500.0, sample_name="simulated_gate_sweep")
