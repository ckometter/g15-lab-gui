# Data plotter (a separate program)

The Data plotter shows the saved files of every measurement tab: Spectrum,
Dual-Gate Sweep, B/V Sweep, Vector Map, Scalar Map, Position DGS, Capacitance
and Transport. The Dashboard and Polarization control tabs save no measurement
files, and files of other programs (the level meter's, for example) are
refused. It is a separate program, so it can be used while the dashboard
measures. Open it with **Open Data plotter…** in the dashboard's toolbar, which
is enabled at all times, or from a console in the repository folder:

    python data_plotter.py [FILE or FOLDER]
    python data_plotter.py --check FILE...

`--check` prints what the Data plotter would show for each file, without a
window. Several Data plotters can be open at once, and closing the dashboard
does not close them. Started from the dashboard, it has no console; its output
goes to `g15_data_plotter.log` in the Windows temporary folder (`%TEMP%`).

Choose Browse folder or Open file, or double-click an HDF5 file in the
browser (Enter opens the selected file). Its contents select the matching tab,
which shows it with that tab's own plots and analysis controls (Signal,
Signal - Background, contrast, regions, slices, cuts); the acquisition controls
are hidden. A file chosen while another one loads opens next. Invalid files
show an error and retain the previous display. **Reload** reads the displayed
file again and keeps the selected scanner pixel.

- **Dual-Gate Sweep, Position DGS:** the exact recorded physical voltage grids,
  measured mask and saved background/reference. Unmeasured gate cells are
  displayed as missing data, including when selecting positions or
  autoscaling. Position DGS reads a single position's DGS on selection; click a
  scanner pixel to inspect its spectrum and gate maps.
- **B/V Sweep, Vector Map, Scalar Map:** the recorded axes (for the maps, the X
  and Y parameters recorded with the scan) and the file's own background and
  reference. Unmeasured points show the value 0 that these tabs save for them;
  the panel gives the measured count.
- **Capacitance, Transport:** shown as the tab's own Open shows a file, with
  its maps, cuts and log.

## Why it cannot slow a measurement

- It runs in its own process. Loading and drawing never use the dashboard's
  GUI thread. They also never wait for h5py's lock, which lets only one HDF5
  read or write run at a time within a process. In a test on the lab PC, a read
  in a thread of the measuring process delayed a Position DGS point write from
  6 ms to 195 ms. An error in the Data plotter cannot stop a measurement.
- It reads files in threads with the lowest disk and CPU priority, in a
  process with below-normal priority.
- It opens each file read-only, without an HDF5 file lock, and only while it
  reads. On Windows, a reader that holds the lock makes the writer's next open
  fail; for Position DGS that stops the scan.
- It writes nothing: no data and no `value_storage`. Its analysis windows keep
  their settings in a private temporary copy. It has simulated devices only
  and refuses instrument (VISA), serial-port, spectrometer-server and network
  connections.

The dashboard itself no longer has a Data plotter mode or slider, so its
measurement tabs and device polling are never taken over for analysis.

## Files that are being written

Position DGS writes its recovery file (`data/<date>/autosave/*PSDGS*.h5`)
after every gate point. DGS writes its autosave once, when the sweep ends.
While a writer has the file open, Windows refuses reads of it; the Data plotter
then retries for about 5 s before it reports that the file is busy. A read is
used when the file did not change while it ran. If the file keeps changing,
the Data plotter shows the last complete read and says so. The info panel shows
the recorded scan status and when the file was read; Reload shows newer points.

The writer is unaffected. In the test suite, a Position DGS writer in another
process writes points while the Data plotter reads the same file continuously,
and no write may fail.

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

- Relaunch the dashboard at a suitable idle point to load the new source code.
  The Data plotter can be started before that: it does not need the dashboard.
- Review `GATE_COORDINATE_PROPOSAL.md` before any coordinate-model migration.

# Potentially dangerous

The Data plotter cannot operate devices. It does not turn off physical outputs
or change instrument setpoints. A separate program or instrument front panel
can still operate the hardware. See `GATE_CURRENT_PROTECTION.md` for limits
and remaining physical risks.

Other programs that open a running Position DGS file with HDF5's default lock
make the scan's next write fail, which stops the scan. Examples are HDFView
and a Python session that calls `h5py.File(path)`. Open such files with
`h5py.File(path, 'r', locking=False)` and close them quickly, or use the Data
plotter.
