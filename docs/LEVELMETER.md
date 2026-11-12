# Helium level on the dashboard

The dashboard shows the cryostat's liquid-helium level in two places:

- **Status bar, on every tab:** for example `LHe 550 mm · -2.4 mm/h · 5 min ago`.
  Hovering shows the details; clicking opens the plot.
- **Dashboard tab, right-hand panel:** the last 14 days with the 230 mm
  low-level line, and the last 24 hours with a linear fit and its slope.

## Where the data comes from

The level meter is a **separate program** in
`C:\Users\QPG-User\Desktop\Python_Script\Levelmeter`. It reads the NI-DAQ every
10 minutes into `LevelData.nc`, keeps running when the dashboard is closed, and
is controlled there:

| Double-click | Does |
|---|---|
| `Levelmeter.bat` | switches it on (never a second copy) |
| `Levelmeter-Stop.bat` | switches it off on purpose; it stays off |
| `Levelmeter-Status.bat` | shows what runs, how many copies, the last reading |

A Windows task, "Level meter watchdog", restarts it at log-on and every 30
minutes unless it was switched off on purpose. The folder's `README_LAB.txt`
explains this.

The dashboard never talks to the instrument. A background thread, at low
priority, checks the folder every 30 s. When `LevelData.nc` has changed, it
copies the file and reads the copy with h5py, so it never locks the file the
level meter writes to. Measurements never wait for it. To read another folder,
set the environment variable `G15_LEVELMETER_DIR`.

## Forecast

The dewar is not a straight tube. In the October 2026 data, for the same
boil-off, the level fell about **5× faster in the neck** (above ~645 mm) and
about **3× faster below ~430 mm** than in between. A straight line through the
last 24 hours of millimetres was therefore wrong in both directions: too
pessimistic just after a refill, then about 2× too optimistic from roughly
560 mm down. Tested on the October refills, its median error was 29 h.

The forecast instead converts the level into a helium amount using that shape.
It measures the boil-off from the last 24 hours below the neck (neck readings
are unreliable just after a refill), falling back to the rate from before the
refill and then to a typical 4 mm/h. Its median error on the same refills was
5 h. The status bar shows the present mm/h. The 24-hour plot still shows a
plain straight-line fit of what was measured. The Slack alerts use the same
model (`levelmeter_slack.py` in the level-meter folder; keep the constants in
step).

## Colours

| Shown | Meaning |
|---|---|
| green | normal |
| orange | below 230 mm, or the 24-hour trend reaches 230 mm within a day |
| **red** | no new reading for 30 minutes (the level meter stopped), or about 0 mm |
| grey | switched off on purpose (`LEVELMETER_STOPPED.txt`), or no data found |

About 0 mm (−1 mm, 0 V) means the level meter is switched off, or the dewar is
empty; the reading alone cannot tell which. Outages longer than an hour are
drawn as breaks in the 14-day plot.

The code is in [`levelmeter_monitor.py`](../levelmeter_monitor.py) (reading and
states, no Qt) and
[`measurements/levelmeter_panel.py`](../measurements/levelmeter_panel.py)
(thread, status label and panel). The tests use synthetic files only:
`tests/test_levelmeter.py`.
