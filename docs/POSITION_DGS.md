# Position dual-gate scans

Open **Position DGS** in the dashboard's left tab bar. This panel uses the same
gate transformation, sweeps, spectra, IV plots, and reference/background options
as Dual-Gate Sweep. Click **Use current scanner offsets** to read the ANC300
without moving it, then choose X/Y start, size and voltage step. A zero size
holds that scanner axis fixed. Sizes must be exact multiples of their steps;
the endpoints are included. Coordinates are scanner **volts**, not calibrated
physical distances.

**Size / span** is the total voltage distance from the start to the end.
**Step** is the spacing between measured positions. For example, start 9 V,
size 0.2 V and step 0.1 V measures 9.0, 9.1 and 9.2 V: three positions.
The preview below the controls shows both endpoints and the total position count.

The offset button copies fresh controller X/Y offsets into **Start** without
changing Size, Step or any hardware setting. It only reads offsets; it does not
require the controller to be ready for movement. Readable offsets above 60 V
are displayed, but scanning from them remains blocked by the safety check.
Read failures keep your previous Start values and display an error. A SIMULATED
message means these values belong to the simulated ANC300; reconnect the physical
ANC300 through the separate **Device Manager** to read its real offsets.

Set the gate ranges in this panel. The number of camera acquisitions and
sum/average settings come from **Single Spectra**, as in the ordinary DGS.
**Begin position scan** performs a complete DGS at each scanner pixel.
The loop order is **Y line → X pixel → full standalone DGS**: hold one gate
fixed while sweeping the other, repeat for every value of the held gate, and
only then move to the next X pixel. The sweep-mode selector and serpentine
checkbox affect the gates within that pixel; they do not change the scanner
raster. Scanner X/Y voltages are separate from the Keysight gate voltages and
never enter the gate transformation. At the next pixel, the same complete gate
sequence starts again. The status shows the acquiring scanner pixel and gate
point count, while the selected-pixel label identifies the DGS being displayed.
An unexpectedly incomplete DGS stops the scan before another scanner move.
The small Viridis map shows acquisition completion. Clicking a pixel displays
its DGS without moving the scanner; checking **Follow current scan pixel**
resumes following acquisition. **Open saved position scan** also works offline
with simulated devices and restores the file's reference/background spectra.

## Motion and limit

The PSDGS safety constant is **`SCANNER_MAX_VOLTAGE = 60.0` in
[`position_dgs.py`](../position_dgs.py)**. The minimum is 0 V. This is the limit
for the new position scan, independent of the gate voltage ranges and the
existing manual scanner controls. Change this one constant and restart the
application if a different PSDGS limit is required.

Both scanner axes and the complete requested grid are checked before movement.
The worker checks again before starting and validates every ramp target and
readback. An out-of-range request produces an error; it is never silently clipped.
An unavailable readback or a controller outside DC offset mode blocks the scan.
The software does not change the controller mode to force movement.

Each line scans X from low to high. X then ramps back to the line start **before**
Y advances. The last line also returns X to its start; Y stays at the final line.
The default forward/Y ramp rate is 0.5 V/s and return rate is 0.2 V/s, with
increments at most 0.05 V. Both rates and the position settling time are editable.
The return rate cannot exceed the forward rate. Initial positioning also ramps
from freshly read offsets, rather than assuming the scanner is at zero.

**Interrupt** pauses; **Resume** continues. **Stop scan** stops after the current
operation, keeps acquired data, and does not command a return to the origin.
An acquisition or motion error likewise retains partial data and stops motion.
Device replacement and other measurement/movement tabs are locked during a scan,
including while paused. The device manager remains available for its safe checks.

## Files and filenames

Each PSDGS streams to one recovery HDF5 file in the measurement's dated
`data/YYYY-MM-DD/autosave` folder. **Save Dataset** copies the *entire position
scan*, including incomplete positions, to the chosen filename. It does not save
just the displayed pixel. Data is written one spectrum at a time; the full
position cube is not held in memory.

Ordinary DGS manual saves now propose names such as:

```
DGS_RC_0T_0p2s_1a_1090nm_Vg_m10-10V_101x1_.h5
DGS_RC_0T_0p2s_1a_1090nm_Vmu_m10-10V_VE_m2-2V_101x21_.h5
PSDGS_RC_0T_0p2s_1a_1090nm_Vg_m10-10V_101x1_X9-9p2V_Y32-32p1V_3x2_.h5
```

`m` means minus, `p` means decimal point. `RC` is the requested default filename
label; raw spectra remain available regardless of the selected display
processing. Gate names and voltage ranges are included only for axes containing
more than one point; both gate point counts remain in the name. PSDGS appends X/Y
voltage bounds and spatial grid dimensions. Exposure, acquisition count, field
and gate axes are captured from the run, so editing the next scan does not rename
previous data using the new scan's settings.

**Filename wavelength** defaults to 1090 nm and is manually editable. It changes
filename metadata only: it never moves the monochromator or replaces the actual
camera wavelength calibration in the file. The save dialog also lets you edit
the complete suggested filename. Ordinary DGS recovery filenames are unchanged.

## HDF5 axes and plotting

| Dataset | Dimensions / meaning |
| --- | --- |
| `spectra` | `[y, x, gate2, gate1, energy]`, sum/average of raw acquisitions |
| `raw_intensities` | `[y, x, gate2, gate1, acquisition, energy]` |
| `measured_mask` | `[y, x, gate2, gate1]`; true only after a point was saved |
| `current_ch1`, `current_ch2`, `measured_volt_ch1`, `measured_volt_ch2`, `measured_b` | `[y, x, gate2, gate1]` |
| `x_voltages`, `y_voltages` | Requested scanner coordinates in volts |
| `position_readback_V` | `[y, x, axis]`; X/Y offsets read before each DGS |
| `c1_coords`, `c2_coords`, `v1_grid`, `v2_grid` | Virtual gates and corresponding physical gate voltages |
| `energy_axis`, `wavelength_axis` | Matching calibrated pixel coordinates, sorted by increasing energy |
| `position_complete` | `[y, x]`; every gate point acquired at that position |
| `position_signal` | `[y, x]`; mean integrated raw signal across measured gates |

Datasets have labelled HDF5 dimensions and units. Unmeasured floating-point
values are NaN. Metadata includes simulation/physical device modes, acquisition
settings, gate transform, scanner rates, limit, run start, completion/error status,
and available reference/background spectra. An interrupted/error file is usable;
apply `measured_mask` when analysing it. The pixel selector supports up to
1,000,000 positions (and at most 100001 coordinates per scanner axis).

Example: plot integrated intensity in an energy window versus scanner position,
at a chosen gate point, reading one position at a time:

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

gate1, gate2 = 0, 0
energy_min, energy_max = 1.10, 1.20  # eV; choose for the recorded spectrum
with h5py.File("my_PSDGS.h5", "r") as f:
    x, y = f["x_voltages"][:], f["y_voltages"][:]
    energy = f["energy_axis"][:]
    band = (energy >= energy_min) & (energy <= energy_max)
    if not band.any():
        raise ValueError("The requested band is outside this spectrum.")
    feature = np.full((len(y), len(x)), np.nan)
    for iy in range(len(y)):
        for ix in range(len(x)):
            if f["measured_mask"][iy, ix, gate2, gate1]:
                spectrum = f["spectra"][iy, ix, gate2, gate1]
                feature[iy, ix] = spectrum[band].sum()

plt.pcolormesh(x, y, feature, shading="nearest", cmap="viridis")
plt.xlabel("Scanner X offset (V)")
plt.ylabel("Scanner Y offset (V)")
plt.colorbar(label="Integrated intensity (counts)")
plt.show()
```
