# Starting the applications

Run `lab_gui.py` (or `start_lab_gui.pyw`) to open the Device Manager. Running
`device_manager.py` opens the same shared application. Use **Open Measurement
Dashboard** to show the GUI. Both windows share one station in one process.

Select **Physical** and click **Connect / Retry** to activate an instrument.
The existing dashboard and measurement data are retained. Device changes
require idle measurements and motion; polling is paused during replacement.
**Auto** may fall back to clearly labelled simulation when hardware is absent.

The separate-process manager was reverted because it did not initialize
reliably alongside the GUI. Close the old processes and relaunch this version
once to restore the shared manager/dashboard workflow.

# Voltage ramps and IV plots

During a manual DC voltage ramp, switch to Single Spectra and back to the
source panel to observe spectra or stop the ramp. Other sweep panels remain
locked. The additional 100 V software cap was removed; the Keysight SMU's
supported range remains -210 to +210 V. The 100 nA current protection and
0.05 V maximum ramp increment remain in effect. Scanner limits are separate.

IV slice curves show only measured, finite points. The horizontal axis uses
the selected sweep coordinate (or magnetic field for vertical B/V slices),
and the vertical axis is current in amperes, with appropriate SI prefixes.
Current ranges fit small or constant signals rather than defaulting to 1 A.
Manual zoom is retained until the plot is reset.

# Measurement filenames

Autosaves and manually chosen save filenames begin with the measurement's
start date: `yyyy_mm_dd_rest`, for example
`2026_10_03_bv_sweep_153025.h5`. Saving after midnight retains the start date.
The date is also retained in measurement metadata and the existing dated data
directory. A custom name chosen in the save dialog receives the same prefix.
Programmatic saves to an explicitly supplied path keep that exact path.
