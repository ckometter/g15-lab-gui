"""Validate an acquired spectrum before a sweep marks a point as measured."""

import numpy as np


def acquire_sweep_spectrum(camera, *, pixels, frames, average, wavelengths=None):
    """Keep existing sum/average behavior; never fill missing data or resample it."""
    if camera is None:
        raise RuntimeError("No spectrometer camera is initialized.")
    result = camera.remote_lf.get_spectrum()
    if result is None:
        raise RuntimeError("Spectrometer returned no spectrum. Measurement stopped.")
    intensities, received_wavelengths = result
    raw = np.asarray(intensities, dtype=float)
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    if raw.shape != (frames, pixels) or not np.isfinite(raw).all():
        raise ValueError(
            f"Invalid spectrum: expected {frames} frame(s) of {pixels} pixels, "
            f"received shape {raw.shape}. Recheck the spectrometer before restarting."
        )
    axis = np.asarray(received_wavelengths, dtype=float)
    if axis.shape != (pixels,) or not np.isfinite(axis).all() or np.any(axis <= 0):
        raise ValueError("Spectrometer returned an invalid wavelength calibration.")
    if wavelengths is not None and not np.allclose(axis, wavelengths, rtol=0, atol=1e-8):
        raise ValueError("Spectrometer wavelength calibration changed. Restart the sweep with the new calibration.")
    return (np.mean(raw, axis=0) if average else np.sum(raw, axis=0)), raw
