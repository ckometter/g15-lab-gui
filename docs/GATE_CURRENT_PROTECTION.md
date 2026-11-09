# Gate protection and current compliance

The central settings are in `gate_safety.py`:

```python
GATE_CURRENT_COMPLIANCE_A = 100e-9
GATE_MIN_CURRENT_COMPLIANCE_A = 1e-9
GATE_MAX_CURRENT_COMPLIANCE_A = 3.0
GATE_MAX_ABS_VOLTAGE_V = 210
GATE_MAX_RAMP_STEP_V = 0.05
DEFAULT_GATE_RAMP_RATE_V_PER_S = 0.1
```

The connection default is 100 nA per channel. In Dual Gate Sweep, edit either
**Current limit** field (in amperes), then press Enter or leave the edited field.
The GUI applies the value on a background worker and verifies instrument
readback. The accepted range is 1 nA through 3 A; the instrument must confirm
the setting for its operating range. See the [Keysight compliance guide](https://www.keysight.com/us/en/assets/9018-01860/user-manuals/9018-01860.pdf)
and [B2902B specifications](https://www.keysight.com/zz/en/product/B2902B/precision-smu-2ch-100fa-resolution-210v-3a-dc-10-5a-pulse.html).

Each channel retains its selected, verified limit throughout subsequent ramps,
output enables and sweeps using that SMU connection, including other tabs.
A failed update is reported and does not replace the previous runtime target.
Reconnecting starts again at 100 nA. Saved settings and measurement files record
the values but never apply them to hardware automatically.

The voltage range and ramp bounds remain configured in code. Both gate channels use the SMU range of -210 through +210 V **after**
any coordinate calculation. A complete sweep is checked before motion starts;
the driver also checks every requested voltage and its initial source readback.
A rejected sweep is not silently clipped.

Position DGS compliance fields remain read-only; voltage compliance in current
source mode is also read-only. Incorrect readbacks remain visible in red; a
failed query says "Readback unavailable". Source current compliance is verified
at connection, before gate ramps and before enabling outputs. If the selected
limit cannot be verified, the operation stops. Display refresh and GUI settings
reload never set compliance, and saved-data plotting cannot edit it. Compliance
editing is blocked during acquisition. DGS exports record the per-channel
current limits confirmed for the measurement.

Dual Gate stores its matrix in `dual_gate.coeff_a` through `coeff_d`. These
take precedence over legacy DGS settings and old shared coefficients. Shared
coefficients remain a compatibility mirror for map panels. B/V stores its
two independent weights in `bv_sweep.control_coeff_ch1` and
`bv_sweep.control_coeff_ch2`; changing either sweep leaves the other's
coefficients intact. Renaming a DGS coordinate does not reset its coefficients.
A saved singular DGS matrix opens with an inline explanation and undefined
target grids. DGS acquisition remains blocked until its determinant is nonzero;
the application does not silently substitute a different calibration.

All direct `channel.volt(...)` and `channel.set_r.volt(...)` writes share one
ramp. It queries the actual source setpoint, checks protection, advances in
steps no larger than 0.05 V using the positive configured rate, and verifies
the final setpoint. Invalid rates/steps cannot fall back to an immediate jump.
Simulations use the same intermediate values without real-time ramp delays.
At each measurement point: ramp channel 1, ramp channel 2, wait the full GUI
settling time, then acquire. Stop or a failed ramp prevents acquisition;
resuming a paused DGS repeats the point's ramps and settling before acquiring.
Stop holds the last gate voltages; B/V ground-on-completion remains a ramp
applied only after successful completion. Dual Gate's **Ramp gates to 0 V when
complete** (off by default, saved as `dual_gate.ramp_to_zero`) likewise runs only
after a complete sweep: both channels move together along a straight line to
0 V in alternating steps of at most 0.05 V at their ramp rates, so the
gate-to-gate voltage never rises on the way. Ramping one channel first could:
(5 V, 5 V) passes (0 V, 5 V). Outputs stay on. The GUI stays locked until 0 V
is reached; Stop holds the voltages reached and Interrupt repeats the step after
Resume. If the end ramp fails, the error states that the sweep data is complete.
Position DGS hides this option and keeps the gates after every pixel.

Enabling a previously disabled voltage output first ramps the dormant setpoint
to zero, enables the output, and ramps back to its original requested voltage.
Manual enable runs in a worker so the GUI can respond. OFF remains immediate.
Changing source mode while an output is enabled is blocked. Gate measurements
require actual voltage-source mode. Manual current-source controls retain a
100 nA source-current bound and <=210 V voltage compliance, but current mode
cannot guarantee a voltage slew rate: sample impedance determines voltage.
Voltage compliance is read-only in the GUI too; code can configure it through
the driver, within the voltage cap.

## Potentially dangerous

- An application already running still has its old code. Relaunch when existing
  measurements have finished; this edit does not modify another running process.
- +210 V on one gate and -210 V on the other gives a 420 V gate-to-gate difference.
  These limits are not a sample-specific safe operating envelope.
- Software ramps are finite voltage increments, not hardware waveform slew
  guarantees. Current compliance cannot guarantee protection against all
  charging transients or dielectric breakdown.
- Front-panel changes and other control programs remain outside this GUI's
  control. The driver checks protection before its next ramp/output operation.
- DGS uses the inverse of its displayed matrix; map modules retain their direct
  mapping. B/V defines one control, `C = a*CH1 + b*CH2`, with user-chosen weights.
  It follows `CH1 = a*C/(a*a+b*b)` and `CH2 = b*C/(a*a+b*b)`, keeping the other
  independent combination zero. With `a = b = 0.5`, both channels equal C.
  B/V weights are stored separately from DGS; old B/V fixed-coordinate and
  matrix settings do not define this new control. Both zero weights and any
  resulting physical target outside the instrument range block acquisition.
  A spectrum is acquired at each control/magnetic-field pair, and exports
  record the acquisition weights and physical target grids.
- No physical instrument was used for validation. Recheck instrument readbacks
  after relaunch before using a sample.
