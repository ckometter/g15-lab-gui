import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import qcodes as qc

# Add parent folder to path to import station_setup
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from station_setup import initialize_station

def plot_spectrum(station, target_wavelength: float = 500.0, exposure_time: float = 0.1, plotyn: bool = True):
    """
    Sets the monochromator wavelength to target_wavelength, camera exposure to exposure_time,
    captures a 2D image from the camera, sums it vertically to obtain the 1D spectrum, and plots it.
    """
    mono = station.monochromator
    cam = station.camera
    
    # 1. Update exposure and wavelength only if they actually changed (prevents slow hardware writes in loops)
    if abs(cam.exposure_time() - exposure_time) > 0.001:
        cam.exposure_time(exposure_time)
    if abs(mono.wavelength() - target_wavelength) > 0.001:
        mono.wavelength(target_wavelength)
        
    # 2. Capture 2D image and sum vertically
    img_2d = cam.image()
    spectrum_1d = np.sum(img_2d, axis=0)
    
    # 3. Calibrate wavelength axis
    dispersion = 0.05
    center_pixel = len(spectrum_1d) // 2
    pixels = np.arange(len(spectrum_1d))
    wavelengths = target_wavelength + (pixels - center_pixel) * dispersion
    
    # 4. Plot the spectrum if plotyn is True
    if plotyn:
        is_simulated = getattr(cam, 'simulated', False) or getattr(mono, 'simulated', False)
        plt.figure(figsize=(8, 4))
        plt.plot(wavelengths, spectrum_1d, color='#e377c2' if is_simulated else '#1f77b4', lw=2)
        plt.title(f"1D Spectrum (Monochromator: {target_wavelength} nm)")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (Counts)")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save a copy of the plot if run directly as a script
        if __name__ == '__main__':
            output_dir = os.path.dirname(os.path.abspath(__file__))
            plot_file = os.path.join(output_dir, "spectrum_plot.png")
            plt.savefig(plot_file)
            print(f"Saved plot to {plot_file}")
            
        plt.show()

    return wavelengths, spectrum_1d

if __name__ == '__main__':
    print("=== Running plot_spectrum in Simulated Mode ===")
    station = initialize_station(simulated=True)
    plot_spectrum(station, target_wavelength=500.0, exposure_time=0.1, plotyn=True)
