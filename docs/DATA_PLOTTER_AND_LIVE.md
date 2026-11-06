# Measurement / Data plotter

Use the two-position slider at the top of the dashboard while it is idle.
In Data plotter mode, only Spectrum, Dual-Gate Sweep and Position DGS remain
visible. The existing acquisition, source and movement forms are hidden.
Device polling is stopped after any in-flight read finishes; loaded files do
not become hardware commands or overwrite the measurement settings.

Choose Browse folder or Open file in the Saved measurements panel, or
double-click an HDF5 file in the browser. Its schema selects the matching tab.
DGS uses the exact recorded physical voltage grids, measured mask and saved
background/reference. Position DGS reads a single position's DGS on selection;
click a scanner pixel to inspect its spectrum and gate maps. The existing
analysis controls (Signal, Signal - Background, contrast, regions and slices)
remain usable. Unmeasured gate cells are displayed as missing data, including
when selecting positions or autoscaling. Invalid files show an error and retain the previous display.

Switching back restores the measurement arrays, settings and plot views held
before entry, and resumes normal polling. Mode switching is blocked during
measurements, ramps, movement, device reconnects and active file loading.
The separate Device Manager remains available for explicitly requested
connection diagnostics; Connect/Retry cannot replace devices in plotter mode.

# Live spectra

Live off retains the latest completed spectrum. If the legacy NIRvana server
is finishing an exposure, the live button remains in its stopping state until
that frame returns; it then becomes the retained frame. Live operation does
not auto-save spectra from the GUI. The existing NIRvana server may still
write its temporary acquisition/export files; its software was not changed.

Live plots auto-scale unless the user has zoomed or panned. Auto resumes
scaling. A bounded latest-frame mailbox and a 33 ms display timer avoid a queue
of old frames; a slow preview query runs outside the GUI thread. The former
100 ms post-acquisition delay was removed. Camera exposure, transfer and file
export times still bound the achievable frame rate.

A simulated 512-point render benchmark measured median replot work falling
from 0.467 ms to 0.075 ms. This is a software plotting benchmark, not a claim
about physical camera frame rate. Test details and logs are in
`data/maintenance/live-render-benchmark.json` and the regression logs.

# To do

- Relaunch the app at a suitable idle point to load the new source code.
- Review `GATE_COORDINATE_PROPOSAL.md` before any coordinate-model migration.

# Potentially dangerous

Data plotter disables application acquisition and source actions; it does not
turn off physical outputs or change instrument setpoints. A separate program
or instrument front panel can still operate the hardware. Returning to
Measurement restores the previous setup; the ordinary gate preflight checks
run again when starting a measurement. See `GATE_CURRENT_PROTECTION.md` for
limits and remaining physical risks.
