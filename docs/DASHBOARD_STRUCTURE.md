# Dashboard tabs: structure and shared code

| Index | Tab | Module (`measurements/`) | Measures |
|---|---|---|---|
| 0 | Single Spectra | `single_spectra_gui.py` | spectra, live view |
| 1 | Dual-Gate Sweep | `Dual_gate_sweep_gui.py` | spectra vs two gates, Keysight or uVoltDAQ outputs |
| 2 | B/V Sweep | `B_V_sweep_gui.py` | spectra vs field and gate |
| 3 | Vector Map | `photoluminescence_map_gui.py` | spectra vs scanner position |
| 4 | Scalar Map | `scalar_map_gui.py` | power meter vs scanner position |
| 5 | Dashboard | `position_tab.py` | positioners and magnet; helium-level plot ([LEVELMETER.md](LEVELMETER.md)) |
| 6 | Polarization control | `polarization_control_gui.py` | rotators |
| 7 | Position DGS | `position_dgs_gui.py`, a subclass of tab 1 | a dual-gate sweep at each scanner position |
| 8 | Capacitance | `capacitance_gui.py`, a gate-map tab | bridge C and D vs gates |
| 9 | Transport | `transport_gui.py`, a gate-map tab | four-probe R vs gates |

Indices are positional; new tabs are appended so earlier indices stay valid.

## Gate-map tabs (8, 9)

These tabs sweep the same three electrodes (top gate, bottom gate, sample)
and differ only in what they read. They share:

- [`electrode_sweep.py`](../electrode_sweep.py): the n0/p0 and direct
  coordinates, sweep plans, voltage limits and duration estimate (no Qt);
- [`electrode_outputs.py`](../electrode_outputs.py): Keysight, DAC and
  external outputs, coordinated moves, and stop/pause;
- [`measurements/gate_map_gui.py`](../measurements/gate_map_gui.py):
  - `GateMapWindow`, with the measurement, sweep, run, electrode and live
    panels, the preview, maps with cuts and a crosshair, and save, autosave
    and open;
  - the sweep engine `run_plan` and the task and live-polling threads.

A subclass adds its instrument panel, its worker-side point read and its file
format. The dashboard handles every gate-map tab through
`LabMainWindow.gate_map_tabs` (busy state, settings, live poller, device
changes and shutdown), so adding another one, such as a DC IV map, needs no
changes to the dashboard's tab handling.

## Evaluated and left separate

- **Spectroscopy tabs (0–4, 7).** Each has its own view popup, live poller and
  sweep worker. The copies have diverged (45–70 % similar line by line). Each
  one carries tab-specific safety behaviour: current compliance, manual ramps,
  live alignment and Data plotter mode. Merging them, or Vector Map with
  Scalar Map (57 % similar), would mean rewriting working, safety-tuned code
  for little gain in use. If they are reworked later, a good first step is one
  shared view popup in `measurements/utils.py`.
- **Capacitance and Transport** stay separate tabs, although they share all
  of the code above. They need different instruments and wiring, and two tabs
  keep each one's settings and last run on screen.
- **`lab_gui.py`** still handles tabs 0–7 with per-index code; only the
  gate-map tabs go through the shared table.
