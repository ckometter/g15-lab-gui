# Measurement changes: checklist and validation

## Completed

- [x] Default 100 nA compliance, editable and verified per channel in DGS; +/-210 V per physical gate channel.
- [x] Actual compliance display, with unavailable/incorrect readbacks visible; settings reload and plotting never apply limits.
- [x] Whole-sweep physical-voltage validation before movement; driver checks every gate target.
- [x] Gate ramp, then GUI settling delay, then acquisition; failures/cancellation stop capture.
- [x] Scroll gestures cannot modify numeric inputs, dropdowns or sliders, even when focused.
- [x] Stop remains available during manual ramps; competing tabs/Start/Reset are blocked.
- [x] Spectrum/DGS/PSDGS data-plotter slider, asynchronous file loading and measurement-state restoration.
- [x] Recorded physical grids/calibrations retained; missing points stay visibly missing.
- [x] Last completed live spectrum retained; automatic scaling yields to manual zoom/pan.
- [x] Latest-frame display avoids stale-frame queues and the former 100 ms artificial delay.
- [x] Direct gate-coefficient proposal documented; existing transform not migrated.
- [x] Native Qt shutdown crash reproduced, isolated and fixed while keeping wheel protection active.
- [x] Optional Dual-Gate end ramp: both gates to 0 V together, only after a complete sweep (off by default).
- [x] Dual-Gate outputs: Keysight or a uVoltDAQ DAC per gate; DAC gates checked against +/-10 V, currents saved as NaN.
- [x] Transport tab: both lock-ins' input modes are checked against the selected wiring before any electrode moves; whole-map limits as in the Capacitance tab (default top-gate dielectric limit 0.8 V, from the script's station).

## To do after editing

- [ ] Relaunch the measurement GUI at an idle point; source edits do not update a running process.
- [ ] Check physical device/readback status before connecting a sample or measuring.
- [ ] Check that each gate is wired to its selected Dual-Gate **Output** before measuring with the uVoltDAQ.
- [ ] Before the first Transport run: connect `lockin2` (SR830 at `GPIB0::10::INSTR`) in the Device Manager, set the Transport tab's gate limits for the device, and compare one live R4 reading with the qcodes-qpg script.
- [ ] Review `GATE_COORDINATE_PROPOSAL.md` before implementing a new coordinate convention.

## Validation (2026-10-01)

All checks used simulation/mocks, including fake SCPI; no physical instruments
were operated. The first 73 integration checks passed. A subsequent full
159-test run exposed two subtest failures because old fake SMUs did not supply
the now-required voltage-source readback, plus a closed-camera status callback.
Those fixtures and callback were corrected. Follow-up interaction checks exposed
a manual-ramp tab lockout gap, now fixed and covered.

The final affected-test run passed **51 tests** with process exit code 0:
`data/maintenance/gate-ramp-plotter-final-verified.txt`. The final busy-tab,
manual-output and responsive-Stop checks passed **3 tests**, also exit code 0.
The dedicated driver/BV/map safety checks passed **38 tests**. These runs overlap;
the counts must not be added as a count of distinct tests.

A native Windows access violation occurred after an early two-test run printed
OK. It was NOT treated as a passing shutdown check. An A/B run isolated the
application-wide wheel filter lifetime. Weak root references and explicit
uninstall on accepted close fixed it. With wheel protection enabled, the two
manual-output tests then passed twice with normal exit code 0; a separate nine
plotter/wheel lifecycle test run also exited 0. Native crash dialogs are disabled
only in the test helpers, and native exit codes remain checked.

Modified source files compile. The simulated plotter was visually inspected;
`data/maintenance/plotter-preview.png` shows loaded DGS data and a missing pixel.
Source backups are under `data/maintenance/before-*.zip`; earlier user changes
and Git history were preserved. No running measurement app or NIRvana software
was restarted or updated on its host PC.

## Potentially dangerous

- An already-running application still uses its old loaded code.
- +/-14 V per channel permits 28 V between opposite-polarity gates. Neither this
  cap nor 100 nA compliance establishes a sample-specific safe electric field.
- Manual current-source mode cannot guarantee voltage slew rate; gate sweeps
  require voltage sourcing. Explicit OFF remains immediate.
- Stop and Data plotter mode leave gate outputs at existing setpoints; they do
  not automatically disconnect or ground the sample. The optional Dual-Gate end
  ramp only brings the setpoints to 0 V after a complete sweep; outputs stay on.
- Other programs and instrument front-panel changes remain outside GUI control.

See `GATE_CURRENT_PROTECTION.md` for the exact protections and
`DATA_PLOTTER_AND_LIVE.md` for operation and performance limits.
