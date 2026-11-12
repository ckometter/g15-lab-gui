# Four-probe transport with two lock-ins

The **Transport** tab (index 9) ports the qcodes-qpg
`transport_measurement_v2.py` workflow into the dashboard: lock-in `lia_I`
reads the sample current, `lia_V` the four-probe voltage, and the gates sweep
a map. At every point the tab records both lock-ins' X and Y and derives
`R4 = V_x / I_x` and the two-terminal `R2` of `transport_measurement.py`.

It shares the electrode, coordinate, limit, sweep, plot and file machinery
with the Capacitance tab ([CAPACITANCE_MEASUREMENTS.md](CAPACITANCE_MEASUREMENTS.md)).
The transport logic is in [`transport_lockin.py`](../transport_lockin.py),
without Qt, so scripts can reuse it.

## Instruments

| Device name | Instrument | Address (`station_setup.py`) |
|---|---|---|
| `lockin` | SR830 lock-in 1 (`lia_I`, also the capacitance bridge lock-in) | `LOCKIN_ADDRESS = 'GPIB0::9::INSTR'` |
| `lockin2` | SR830 lock-in 2 (`lia_V`) | `LOCKIN2_ADDRESS = 'GPIB0::10::INSTR'` |
| `dac` | QPG uVoltDAQ (`uvolt`) | `DAC_ADDRESS = 'ASRL15::INSTR'` |
| `smu` | Keysight B2902B | optional gate outputs |

Connect them in the **Device Manager**. **Current read by** selects the
lock-in that reads the current; its sine output is taken as the excitation.
The other lock-in reads the voltage, so the two can be swapped in the GUI.

## Wiring and settings

- Typical circuit: current lock-in SINE OUT → divider and/or series resistor →
  source contact; drain → the current lock-in's current input, or a current
  preamp into input A. The inner probes go (through a voltage preamp) to the
  voltage lock-in's A-B input, whose REF IN takes the current lock-in's
  TTL/REF OUT with External reference.
- **Current input**: *SR830 current input (reads A)* or *Current preamp into
  voltage input* with the preamp sensitivity in A/V. At Begin both lock-ins'
  input modes are read. A mismatch (a current-input choice with the lock-in on
  input A, a preamp choice with the lock-in on a current input, or the voltage
  lock-in on a current input) blocks the run before any electrode moves.
- **Voltage preamp gain** divides the voltage reading (`transport_measurement.py`
  used 100).
- **R2** = V_exc / I_x − R_series, with V_exc = sine amplitude × **Excitation
  divider**. R4 does not depend on the divider or the **Series resistor**.
- **Electrodes and limits**: the defaults follow the script. The back gate is
  on DAC0 and the top gate on DAC1. The sample is not driven: it is grounded
  through the lock-in or preamp and held at 0 V. Coordinates are direct
  (x = V_top slow, y = V_bottom fast, -0.8..0 V × -4.5..0 V). The limits are
  the script station's: |Vt − Vs| ≤ 0.8 V, |Vb − Vs| ≤ 10 V and |V| ≤ 10 V.
  The n0/p0 coordinates (the scripts' `RotatedBasis`) are available too.
- **Time constant**: *Keep front-panel value* (default), or a value applied to
  both lock-ins at Begin.
- **Point settle**: N × the longer time constant (default 3, the script's
  `settle_s`) plus the Sweep group's **Extra point settle**.
- **Autorange** (optional, per lock-in): at every point. A range change repeats
  the settle, and the sensitivity is saved per point.
- **Set the excitation at Begin** (off by default) applies the frequency and
  sine amplitude with the internal reference on the current lock-in, and sets
  the voltage lock-in to External. When it is off, the front-panel settings
  are only read and recorded.
- **Live readout** shows:
  - both lock-ins' X and Y;
  - R4 and R2, using the panel's conversions;
  - each lock-in's τ, frequency and reference;
  - a warning for a wrong input mode or an unlocked reference.

## What Begin does

1. Validate the whole map, the ramp to 0 V and the lock-in settings without
   hardware.
2. Check the outputs (Keysight voltage mode, output on, compliance; DAC
   setpoints). Then read both lock-ins: input mode, reference, frequency,
   amplitude, τ, slope and sensitivity.
   - An input-mode mismatch stops the run here.
   - Reference problems are logged and saved as warnings. These are: the
     voltage lock-in not on External reference, frequencies more than 1 %
     apart, or different time constants.
3. Optionally set the excitation and τ, then compute the settle and V_exc.
4. Ramp to the first point and wait for the filters (5/7/9/10 τ for
   6/12/18/24 dB/oct), like the script's `before_run`. All electrodes move
   together in steps of at most 0.05 V.
5. For every line, wait the line settle. Then, for every point:
   - ramp to the point, clear both status bytes and settle;
   - autorange if it is enabled;
   - read both lock-ins, back to back with one `SNAP` each, and their status
     bytes;
   - read the electrode readbacks and the optional ADC.
6. After a complete run, optionally ramp the driven electrodes to 0 V
   together, then autosave.

## Deliberate differences from the script

- Electrode ramps are coordinated and verified, moving all electrodes along a
  straight line in voltage space in steps of at most 0.05 V. The script relied
  on each station parameter's step and delay.
- Each lock-in's X and Y are read coherently with one `SNAP?`, instead of
  separate queries.
- Overload and reference-unlock bits of both lock-ins are recorded per point
  and counted in the status line.
- The input modes are verified before anything moves. A lock-in left on the
  wrong input cannot silently scale R by 10^6.
- Stop and errors hold the voltages, as in every dashboard tab; the script's
  `finally` ramped to 0 V even after Ctrl+C. Use **Ramp to 0 V** after a stop.
- The settle uses the time constants read at Begin. The simulation has no
  filter, so it waits only the extra settle.

## Files

**Save Dataset** proposes names like
`2026_11_11_transport_4pt_V_top_m0p8-0V_V_bottom_m4p5-0V_20x40_141500.h5` in
`data/YYYY-MM-DD/`. Autosaves go to `data/YYYY-MM-DD/autosave/`, with
`_interrupted` after a stop or error. **Open saved…** displays a file without
sending anything to the instruments.

| Dataset | Meaning |
|---|---|
| `x_coords`, `y_coords`, `target_v_*`, `n0`, `p0` | the sweep grid, as in capacitance files |
| `resistance_4pt` | R4 = `voltage_x / current_x` (Ω) |
| `resistance_2pt` | R2 = `excitation_V_rms / current_x - series_resistance_ohm` (Ω) |
| `current_x`, `current_y` | sample current (A), after the preamp conversion |
| `voltage_x`, `voltage_y` | four-probe voltage at the sample (V), after the gain |
| `lockin_current_x/y`, `lockin_voltage_x/y` | the lock-in readings as read (A or V) |
| `sensitivity_current`, `sensitivity_voltage` | per point |
| `status_current`, `status_voltage` | SR830 `LIAS?` bits |
| `v_top`, `v_bottom`, `v_sample`, `leakage_*`, `aux_adc`, `timestamp` | readbacks, as in capacitance files |
| `measured_mask`, `acquisition_order`, `/metadata/*` | as in capacitance files |

The file attributes record:

- the lock-in roles, current input, preamp sensitivity, voltage gain,
  divider, series resistor and V_exc;
- the settle, both lock-ins' readbacks (`lockin_info`) and any warnings;
- the wiring, and the device and electrode modes;
- the start and end times, the completion status, and the overload and unlock
  counts.

```python
import h5py, numpy as np, matplotlib.pyplot as plt
with h5py.File("2026_11_11_transport_4pt_....h5") as f:
    x, y = f["x_coords"][:], f["y_coords"][:]
    r4 = f["resistance_4pt"][:]            # NaN where not measured
    plt.pcolormesh(x, y, np.log10(np.abs(r4)), shading="nearest")
    plt.xlabel(f["x_coords"].attrs["name"] + " (V)"); plt.ylabel(f["y_coords"].attrs["name"] + " (V)")
plt.colorbar(label="log10 |R4| (Ohm)"); plt.show()
```

## Simulation

When both lock-ins are simulated, the tab binds them to a dual-gated sample
model, whose R4 peaks at charge neutrality and grows with |p0|. It also sets
the simulated front panels to the selected wiring and picks ranges with
headroom. Maps, stop, autorange, overloads and saving can all be rehearsed
offline; `tests/test_transport.py` checks that the recovered R4 matches the
model. The simulation never reconfigures a physical lock-in.

## To do

- Relaunch the dashboard at an idle point to load this code.
- Connect `lockin2` in the Device Manager and check its address,
  `GPIB0::10::INSTR`.
- Set the gate limits for the device in the Transport tab before the first run.
- Compare one live R4 value with the qcodes-qpg script on the same device.

## Potentially dangerous

- The default limits are the script station's (top gate ±0.8 V, back gate
  ±10 V), not a guarantee for another device.
- **Set the excitation at Begin** writes the sine amplitude, up to 5 V rms;
  with a small series resistor that drives a large current.
- Autorange changes the lock-in sensitivity during the run.
- The uVoltDAQ has no current compliance; only Keysight outputs limit leakage.
- Stop, errors and Data plotter mode leave the electrodes at their present
  voltages.
- Validation used simulated instruments and fake I/O only; no physical
  instrument was operated.
