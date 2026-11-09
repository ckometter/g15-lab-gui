# Capacitance bridge measurements

The **Capacitance** tab (last tab, index 8) ports the qcodes-qpg
`capacitance_measurement.py` workflow into the dashboard: balance an AC
capacitance bridge once, then sweep the normalized density `n0` and field `p0`
(or two electrode voltages directly) and convert the off-balance lock-in signal
into capacitance and dissipation at every point. The balancing algorithm is
the original `CapacitanceBridgeSR830Lockin`, ported without Qt to
[`capacitance_bridge.py`](../capacitance_bridge.py), where it can be reused by
scripts and is tested against verbatim copies of the original formulas.

| Type | AC excitation | Bridge (measurement) node | Default DC-fixed electrode |
|---|---|---|---|
| Penetration field capacitance C_P | Top gate | Bottom gate | Bottom gate (`vb`, as in the script's `Cpen`) |
| Top-gate capacitance C_T | Top gate | Sample | Sample (`vs`) |
| Bottom-gate capacitance C_B | Bottom gate | Sample | Sample (`vs`) |
| Symmetric capacitance C_S | Both gates | Sample | Sample (`vs`, as in `Csym`) |
| Custom | recorded in the comment | recorded in the comment | your choice |

The AC excitation and measurement node are wiring facts: the type records them
in the file and selects the usual fixed electrode, which you can override.

## Instruments

Connect them in the **Device Manager** like every other device (Auto,
Physical or Simulation, then **Connect / Retry**). Checks are readback only.

| Device name | Instrument | Address (`station_setup.py`) |
|---|---|---|
| `lockin` | SR830 lock-in (`lia1` in the g15 station.yaml) | `LOCKIN_ADDRESS = 'GPIB0::9::INSTR'` |
| `awg` | Siglent SDG2042X (`awg`) | any USB VISA resource with Siglent vendor ID `F4EC`, else `AWG_ADDRESS` |
| `dac` | QPG uVoltDAQ 4O4I (`da`) | `DAC_ADDRESS = 'ASRL15::INSTR'`; connecting never sends `INITIALIZE`, so present voltages are kept |
| `smu` | Keysight B2902B (existing) | optional electrode outputs |

## Wiring and settings

- **AWG CH1** drives the sample arm at a fixed amplitude; **CH2** drives the
  reference arm. Balancing sets the CH1 phase and the CH2 amplitude, exactly as
  the original `excite()`. The lock-in must be phase-referenced to **CH1** and
  the two channels must be phase-locked (the SDG2000X default); with an
  independent phase mode the balance cannot converge.
- Enter the attenuators actually fitted in each arm. They set
  `ac_scale = 10**(-(ref_atten - sample_atten)/20) / V_CH1`, so the results are
  `C/C_ref` and `D/C_ref`. Enter **C_ref** (pF) to also get farads.
- **Electrodes and limits**: assign the top gate, bottom gate and sample to a
  Keysight channel, a DAC channel, or *Not driven (held externally)*. The
  default is the script's C_P wiring: DAC0 top gate, DAC1 sample, DAC3 bottom
  gate. Unlike the script, the wiring does not change with the measurement
  type; the type only chooses which voltage is held fixed. A swept electrode
  must be driven, and one output cannot drive two electrodes.
- Keysight outputs must be ON and in voltage mode; enable them in the
  Dual-Gate tab (that enable ramps safely) and set their ramp rate there.
  Their verified current compliance applies, and their leakage currents are
  recorded. DAC outputs use the tab's **DAC ramp rate**.
- **Max |V| electrode**, **Max |Vt − Vs|** and **Max |Vb − Vs|** are checked
  for every target of the whole sweep, the balance point and the final
  ramp to 0 V before anything moves. Nothing is clipped; a violation blocks
  the run and the preview shows why. Defaults are 10 V each.

## Coordinates

```
n0 = (1 + delta)(Vt - Vs) + (1 - delta)(Vb - Vs)      delta = (Ct - Cb)/(Ct + Cb)
p0 = (1 + delta)(Vt - Vs) - (1 - delta)(Vb - Vs)
```

The fixed electrode sits at `V_fixed`; the other two follow (`vb_fixed`,
`vt_fixed`, `vs_fixed` in the script). In **Electrode voltages (direct)**
mode, x and y are the voltages of the two non-fixed electrodes, in the
script's order (fixed bottom: top, sample; fixed top: bottom, sample; fixed
sample: top, bottom). The **Electrode targets** plot tab and the preview show
the complete physical trajectory and the run-time estimate before Start.

## What Begin does

1. Validate all targets, the wiring, the probe vectors and the instruments.
2. Check the outputs (Keysight voltage mode, output on, compliance) and the
   lock-in input; warn if the external reference differs from the AWG
   frequency by more than 1 %.
3. Optionally configure the AWG (`init_awg`: sine, frequency, amplitudes,
   zero phases, outputs on) and verify its readback.
4. Ramp to the balance point: the first sweep point (the script balanced at
   `n = p = -1`, its first point) or a custom (x, y).
5. Balance: set the balance time constant, autorange, excite the probe vectors
   `s1`, `s2`, build the response matrix, iterate `vb = -M^-1 c` until
   `|X + iY| < tolerance` or the iteration limit. Each read waits 5/7/9/10
   time constants (6/12/18/24 dB/oct) and autoranges, as in the original.
6. Apply the measurement time constant and sensitivity and let the filter settle.
7. For every line: optionally rebalance (reusing the response matrix), wait the
   line settle time; for every point: ramp, clear the lock-in status, wait the
   point settle time, read X and Y in one `SNAP`, read the status and
   `C = c' X + d' Y + C_s`, `D = d' X - c' Y + D_s`.
8. After a complete run, optionally ramp the driven electrodes to 0 V
   (the script's `thelast()`), then autosave.

All electrodes move together along a straight line in voltage space in steps
of at most 0.05 V. Sequential ramps can transiently put a large voltage across
one dielectric; for example (-5, -5) V to (+5, +5) V passes (+5, -5) V.

**Interrupt** pauses at the next safe point; after **Resume** the full settle
time is repeated. **Stop** ends after the current step, keeps the data, autosaves
it with `_interrupted` and leaves the electrodes where they are. **Balance now**
balances at the present voltages without moving; **Ramp to x, y** and
**Ramp to 0 V** move the electrodes with the same protection.

## Deliberate differences from the script

- Electrode voltages are ramped and verified; the script jumped the DAC.
- X and Y are read coherently with one `SNAP?` instead of two queries.
- Lock-in overload and reference-unlock bits (`LIAS?`) are recorded per point
  and counted in the status line.
- An out-of-range balance stops with an explanation; the original continued
  with `vb = None` and failed later. A balance that hits the iteration limit is
  reported and still used, as in the original.
- **Rebalance every line** is implemented. The original `rebalance()` called
  `measure(vb)`, which does not accept an argument.
- The script addressed `da.DAC0.volt`, while the qcodes-qpg uVoltDAQ driver
  defines `dac0.voltage`; the new driver keeps the driver's names.

## Files

**Save Dataset** proposes names like
`2026_11_06_capacitance_C_P_n0_m1-1V_p0_m1-1V_101x101_153025.h5` in the dated
`data/YYYY-MM-DD` folder (`m` = minus, `p` = decimal point; axes with one point
are omitted from the ranges). Autosaves go to `data/YYYY-MM-DD/autosave/` as
`2026_11_06_autosave_capacitance_…h5`, with `_interrupted` after a stop or error. **Open saved…** displays a file
without sending anything to the instruments.

| Dataset | Shape / meaning |
|---|---|
| `x_coords`, `y_coords` | sweep coordinates (V); `name` attribute is `n0`/`p0` or the electrode |
| `target_v_top`, `target_v_bottom`, `target_v_sample` | `[y, x]` requested voltages |
| `n0`, `p0` | `[y, x]` normalized density and field of every target |
| `lockin_x`, `lockin_y` | `[y, x]` off-balance lock-in signal (V) |
| `capacitance_ratio`, `dissipation_ratio` | `[y, x]` C/C_ref and D/C_ref; `capacitance_F`, `dissipation_F` when C_ref is set |
| `v_top`, `v_bottom`, `v_sample` | readbacks (Keysight measured voltage, DAC setpoint; NaN if not driven) |
| `leakage_top`, `leakage_bottom`, `leakage_sample` | Keysight currents (A); NaN otherwise |
| `lockin_sensitivity`, `lockin_status`, `aux_adc`, `timestamp` | per point |
| `balance_index` | which `/balance` entry calibrated the point |
| `measured_mask`, `acquisition_order` | saved points and the `(iy, ix)` order |
| `/balance/*` | `matrix`, `offset`, `balance_vector`, `capacitance_ratio`, `c_coefficient`, `d_coefficient`, `converged`, … per balance |
| `/metadata/settings`, `/metadata/value_storage` | JSON |

File attributes record the type, fixed electrode, `V_fixed`, `delta`, wiring,
electrode and device simulation modes, AWG readback, lock-in configuration,
channel mapping, `ac_scale`, start/finish time and the completion status.

```python
import h5py, matplotlib.pyplot as plt
with h5py.File("2026_11_06_capacitance_C_P_....h5") as f:
    x, y = f["x_coords"][:], f["y_coords"][:]
    c = f["capacitance_ratio"][:]          # NaN where not measured
    plt.pcolormesh(x, y, c, shading="nearest")
    plt.xlabel(f["x_coords"].attrs["name"] + " (V)"); plt.ylabel(f["y_coords"].attrs["name"] + " (V)")
plt.colorbar(label="C / C_ref"); plt.show()
```

## Simulation

With a simulated lock-in, the tab connects it to a bridge model driven by the
(simulated or cached) AWG settings and electrode voltages: a sample whose
capacitance has a gap at charge neutrality that opens with `|p0|`. Balancing,
off-balance calibration, rebalancing, stop and saving can be rehearsed offline;
`tests/test_capacitance.py` checks that the recovered C/C_ref matches the model.
The tests use simulated instruments and fake I/O only:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_capacitance.py
```

## To do

- Relaunch the dashboard at an idle point to load this code.
- Check the SR830 GPIB address and the DAC COM port in `station_setup.py`,
  then connect each device and inspect its readback in the Device Manager.

## Potentially dangerous

- The voltage limits are code/GUI bounds, not a guarantee for a particular
  sample. Set **Max |V|** and the dielectric limits for your device before
  measuring; the defaults (10 V) are not sample-specific.
- Stop, errors and Data plotter mode leave the electrodes at their present
  voltages; nothing is grounded automatically.
- With **Configure AWG at start** checked, Begin and Balance now enable both
  AWG outputs, as the original script did.
- The uVoltDAQ has no current compliance; only Keysight outputs limit leakage.
- Front-panel changes and other programs remain outside the GUI's control.
- Validation used simulated instruments and fake I/O only; no physical
  instrument was operated.
