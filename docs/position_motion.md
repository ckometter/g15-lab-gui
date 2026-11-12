# Scanner and stepper motion

The Dashboard's **Scanner safety settings** apply to every ANC300 axis, manual
move, vector/scalar map, and position dual-gate scan. Press **Apply and save safety
settings** to persist changes in `value_storage`. Applying settings does not move
the scanner.

Defaults are 0–60 V, a maximum ramp rate of **0.2 V/s**, and a maximum increment of
**0.01 V**. The settings permit voltage bounds within 0–150 V, rates from
0.001–1 V/s, and increments from 0.001–0.05 V. Choose limits appropriate to the
connected scanner. Position dual-gate scans also retain their existing 60 V
ceiling; their forward/return rates are capped by the Dashboard rate.

Entering an individual voltage and pressing Enter, clicking a scanner arrow,
or using either XY motion button starts a background ramp. Both XY targets are
validated before either axis moves; X completes before Y starts.

- **Setscan X, Y** uses absolute voltage targets. Command example:
  `setscan 10, 20` ramps X to 10 V, then Y to 20 V.
- **Goto relative X, Y** adds voltage deltas to fresh controller readbacks.
  `goto -0.1, 0.2` reduces X by 0.1 V and increases Y by 0.2 V.
- **Stop scanner ramp** stops further increments and holds the reached voltage.
  It does not jump back or return to zero.

Every increment waits before writing and verifies the offset afterward. Errors,
changed controller state, and readback mismatches halt motion without retries.
An existing offset outside newly tightened limits can ramp slowly back into the
allowed range. Loading settings or opening the Dashboard never restores saved
voltage targets. Invalid saved safety settings block motion until valid settings
are explicitly applied.

Stepper frequency is limited to **100 Hz** and amplitude to **120 V** in the UI
and driver setters. Motion also checks current controller settings, so a higher
frequency or amplitude set from the front panel blocks movement until corrected.
The Dashboard shows such a setting in red, for example `Fx: 102 Hz (max 100)`,
and a refused move says which axis and value, e.g. `X: ANC150 axis 1 is set to
102 Hz, outside the stepping limit of 1 to 100 Hz`. These messages also go to the
Device Manager's application log. If the ANC150 is simulated (not connected in the
Device Manager), the stepper box title says `SIMULATED, nothing moves`.

Stepper X/Y/Z pseudo-coordinates count acknowledged commanded steps relative to
the current session's origin. **Set zero (no motion)** declares the current
location `(0, 0, 0)` without sending motion commands. **Goto X, Y, Z** moves to an
absolute pseudo-coordinate; the relative checkbox instead adds step counts.
An arrow click makes one step. Holding an arrow (longer than 0.5 s) keeps stepping
until released, at about three quarters of the controller's frequency. Moves are
sent as counted bursts: one `stepu`/`stepd <axis> <n>` command per at most 0.25 s
of steps, each after fresh frequency/amplitude readbacks (~25 ms per command), and
the next only once the controller has made them. Releasing the arrow or Stop ends
after the current burst. (One command per single step, as before, managed only ~6
steps/s instead of the set 43-100 Hz.)

These are open-loop step counts, not measured distance: slip and front-panel
movement cannot be detected. Reconnecting starts a new origin. An uncertain
motion write invalidates that axis's coordinate; arrows and relative moves still
work, but set a new zero before an absolute Goto.
The Dashboard blocks device replacement and competing measurements during motion,
and requests motion cancellation before closing.

Safety tests use simulated devices and fake I/O. No physical motion is required:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_position_motion.py
.venv/Scripts/python.exe -m unittest discover -s tests -p test_dashboard_workflow.py
```
