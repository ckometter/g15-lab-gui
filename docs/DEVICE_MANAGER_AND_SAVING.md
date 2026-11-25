# Starting the applications

Run `lab_gui.py` (or `start_lab_gui.pyw`) to open the Device Manager. Running
`device_manager.py` opens the same shared application. Use **Open Measurement
Dashboard** to show the GUI. Both windows share one station in one process.

Select **Physical** and click **Connect / Retry** to activate an instrument.
The existing dashboard and measurement data are retained. Device changes
require idle measurements and motion; polling is paused during replacement.
**Auto** may fall back to clearly labelled simulation when hardware is absent.

The uVoltDAQ row also has **Initialize**, for after the DAC has been powered on:
it sends `INITIALIZE` (outputs from tri-state to normal mode, all four at 0 V)
after asking, and can ramp nonzero outputs to 0 V first. See
[GATE_CURRENT_PROTECTION.md](GATE_CURRENT_PROTECTION.md#initializing-the-uvoltdaq-after-power-on).

## When serial devices hang or fail

The ANC150, ANC300 and uVoltDAQ are opened on serial ports through NI-VISA.
NI-VISA guards every serial-port search and open with one lock, shared by all
programs on the PC. If the program holding it gets stuck, every serial search
and open waits forever, in any program, including `rm.list_resources()` in a
Python console. GPIB, USB and network devices are not affected. On 17 Nov 2026
the stuck holder was NI Configuration Manager (`nimxs.exe`), the service behind
NI MAX.

- Before opening one of these three devices, the Device Manager reads the
  lock's state; it never takes the lock or opens a port for this. If the lock
  stays taken for 5 s, the device fails at once and its details give the fix
  below. In Auto mode it falls back to simulation. Further serial devices in
  the same check fail without waiting again.
- The AWG and the power meter search only USB resources, so they never wait on
  this lock.
- **To clear it**, open PowerShell with **Run as administrator** and run
  `Stop-Process -Name nimxs -Force`, then
  `Start-Service mxssvr, nidevldu, NITaggerService`. Restarting the PC also
  works. `Restart-Service mxssvr -Force` is not enough on its own: it stops NI
  Device Loader and NI Variable Engine, then waits for the stuck service, and
  until all three run again the level meter's NI-DAQ cannot be read.
- To check the lock without the GUI, run
  `python -c "import visa_serial_lock as v; print(v.serial_lock_owned())"` in
  the repository folder. `True` several times in a row means it is stuck;
  `False` or `None` means serial devices can be opened.

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

A Dual-Gate autosave never replaces an earlier one: a run that ends in the same
second as another gets `_2`. A run started by the measurement queue is also saved
under the name Save Dataset proposes plus its start time, for example
`..._21x41_221930.h5`. See [MEASUREMENT_QUEUE.md](MEASUREMENT_QUEUE.md#files-and-the-record).
