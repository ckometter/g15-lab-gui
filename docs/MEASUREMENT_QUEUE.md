# Measurement queue

The measurement queue runs several Dual-Gate Sweeps one after another, for
example overnight with different ranges. Every run goes through the Dual-Gate
tab's own Begin. So it gets the same checks, current limits, protected ramps,
autosave, filenames and metadata as when you press Begin. Each run gets its own
files, and a record lists what ran.

## Building a queue

- Set up a sweep on the **Dual-Gate Sweep** tab as usual, then press
  **+ Add to queue** below **Begin**.
  - The **Measurement queue** panel opens below the tabs, and the
    **Measurement queue** button in the status bar shows or hides it.
  - The panel can be dragged out as its own window.
- Each entry is one line. It shows:
  - both ranges and point counts;
  - the matrix, if it is not the identity;
  - the variable order, exposure, settling time and the end ramp.
- **Time** is the estimate, from the same model as Expected End
  ([SWEEP_TIMING.md](SWEEP_TIMING.md)). It covers ramps, settling, exposure and
  the last sweep's pace per point. Each run's estimate starts where the
  previous run leaves the gates.
- An entry Begin would refuse is not added; the message gives Begin's reason.
- To edit an entry:
  - **Show on tab** puts its settings in the Dual-Gate fields, without any
    instrument command. Edit them there, then press **Replace with tab**.
  - **Up**, **Down** and **Remove** reorder or delete entries.
  - **Note** is free text. It is saved in the files and the record.
- **Save…** writes the queue as a JSON file (default folder `data/queues/`).
  **Open…** reads a queue or a record. Every opened entry is checked, and one
  that needs a change says why under **State**.
- While the queue runs, the list cannot be changed.

### What an entry holds

| In the entry | Stays as set on the tab |
|---|---|
| C1/C2 start, end and points; virtual-gate names; matrix a–d | V1/V2 **Output** (the wiring) |
| variable order, serpentine | voltage windows (**ch1/ch2 allowed from … to …**) |
| **Ramp gates to 0 V when complete** | current limits, source modes, **Output on** |
| exposure, settling time | ramp rates |
| filename wavelength, note | Single Spectra settings: frames per point, grating, centre wavelength, background and reference |

The queue never changes the right-hand column, and it checks every entry against
it. Ramp rates and frames per point are therefore the same for every run of a
queue. An entry made with other V1/V2 outputs is refused, because the wiring
cannot change overnight.

## Running

Press **Run queue**:

1. It is refused while another measurement or movement is active.
2. Every entry not yet done is checked with Begin's own checks:
   - the whole sweep, through `validate_dgs_parameters`: ±210 V, the uVoltDAQ's
     ±10 V, each gate's voltage window, and a singular matrix;
   - voltage-source mode, and a connected uVoltDAQ for DAC gates;
   - the V1/V2 outputs the entry was made with;
   - that every value fits its field exactly. A spin box would silently turn a
     20 s exposure into 10 s, or 0.00125 V into 0.0013 V.

   If any entry fails, nothing starts or moves, and a message lists the problems.
3. A confirmation gives the entries, the estimated time and the expected end.
4. For each entry, the queue puts its settings in the tab's fields and saves
   them as an edit would. Then it calls `begin_sweep()`, the method Begin calls.
   - Begin then confirms the current limits again, which only the instrument
     can do. If that fails, the queue stops there.
   - The run's messages (blocked, Sweep Complete, Sweep Error) go to the status
     line and the Application log instead of dialogs. Nothing waits for a click.
5. After each run the queue waits until the run's thread has ended. Then it
   starts the next entry.

While it runs:

- The dashboard treats the queue as an active measurement, also between runs.
  These stay locked:
  - the other tabs;
  - the Device Manager's Connect and Initialize;
  - gate outputs and current limits;
  - the Dual-Gate inputs.
- **Interrupt** and **Resume** on the tab pause and resume the run as usual; the
  queue waits.
- **Stop after this run**: the run finishes, including its end ramp, then the
  queue stops.
- **Stop now**, or **Reset** on the tab, stops the run as Reset does: its data is
  autosaved with `_interrupted` and the gates are held. The queue stops.
- **State** shows each entry's progress (`running 37 %`, `done · <file>`,
  `failed: <reason>`), and the status line shows the queue's expected end.
- **Run queue** after a stop or an error runs the entries not yet done.

### Gates between runs and at the end

- **Between runs.** The queue sends no instrument command of its own. The gates
  stay where the previous run left them: at its last point, or at 0 V if its
  **Ramp gates to 0 V when complete** is ticked. The next run's Begin ramps them
  to its first point with the usual protected ramp: V1 first, then V2, in steps
  of at most 0.05 V at the set rates.
  - This is the same as pressing Begin twice by hand, and each gate stays inside
    its window.
  - Because V1 moves before V2, the voltage between the gates on this move can
    exceed both runs'.
  - Tick the end ramp on the entries if the gates should meet at 0 V between runs.
- **After the last entry.** With **Ramp gates to 0 V after the last entry** (on by
  default), both gates ramp to 0 V together after the last entry's complete
  sweep, as that entry's own end ramp would.
  - The record notes when the queue added this ramp.
  - Afterwards the tab keeps the last entry's settings, including the ticked box.
- **After an error or a stop.** The gates are held where they are, the rule in
  every tab. Later entries do not run. The record and the panel give the
  voltages last commanded.

### Errors

An error in a run stops the queue:

- that run's data is autosaved with `_interrupted`;
- the gates are held where they are;
- the remaining entries are marked **not run**;
- a message says what failed.

The queue does not go on with the next entry. An error can be a failed ramp or
lost communication, and the next entry would then move the gates unattended
right after it.

## Files and the record

- **Autosave.** Each run autosaves as before:
  `data/<date>/autosave/<date>_autosave_gate_sweep_HHMMSS.h5`, with `_interrupted`
  after a stop or error. A run that ends in the same second as an earlier one
  gets `_2` instead of overwriting it.
- **Named copy.** Each run is also saved under the name Save Dataset proposes,
  plus its start time, in `data/<date>/`, for example
  `2026_11_25_DGS_RC_0T_0p1s_2a_1090nm_Vc1_m0p5-0p5V_Vc2_m1-1V_21x41_221930.h5`.
  - A run that did not complete gets `_interrupted`.
  - The names describe the files, but this doubles the disk space per run.
  - `<date>` is each run's start date, so runs started after midnight go to the
    next day's folder.
- **File attributes.** The files of a queued run record:
  - `queue_record`, the record's path;
  - `queue_entry` and `queue_entries`, for example 2 and 5;
  - `queue_note`;
  - `queue_started_at`, the queue's start.
- **The record** is `data/<date>/queue/<date>_queue_HHMMSS.json`, dated by the
  queue's start. It is rewritten after every change and holds:
  - the options;
  - the tab settings the entries were checked against: outputs, windows, ramp
    rates, current limits, source modes, frames per point and device modes;
  - for each entry, its settings and note, and its `result`: status, start and
    end, message, `file` (the autosave), `copy`, timing summary and
    `gates_after_V`.
- **Reopening.** A record opens again as a queue; the results are not loaded.
- **Write failures.** If the record cannot be written, for example because
  another program holds it open, the Application log says so. It is written
  again at the next change, and the runs go on.
- **Application log.** It gets a line when the queue starts, when each entry
  starts and ends, and when the queue ends.

```python
import json
record = json.load(open("data/2026-11-25/queue/2026_11_25_queue_221500.json"))
for entry in record["entries"]:
    result, settings = entry["result"], entry["settings"]
    print(result["status"], settings["c1_start"], settings["c1_end"], result.get("copy"))
```

## Writing a queue file by hand or by script

A queue file has this form:

```json
{"format": "g15-measurement-queue", "version": 1,
 "options": {"ramp_to_zero_after_last": true},
 "entries": [{"tab": "dual_gate", "note": "", "settings": {...}}]}
```

The settings are those of an entry made with **Add to queue**, so save one to
start from. Every key must be present and every value must fit its field. A
misspelled key is refused, not replaced by a default. This makes five entries
with growing ranges from a saved one:

```python
import json
queue = json.load(open("data/queues/template.json"))
template, queue["entries"] = queue["entries"][0], []
for end in (0.5, 1.0, 1.5, 2.0, 2.5):
    entry = json.loads(json.dumps(template))
    entry["settings"].update(c1_start=-end, c1_end=end)
    entry["note"] = f"Vc1 +/-{end} V"
    queue["entries"].append(entry)
json.dump(queue, open("data/queues/growing_ranges.json", "w"), indent=2)
```

## Other tabs

- **Joining.** A tab joins the queue through an adapter: `QueueAdapter` in
  `measurements/queue_panel.py`.
  - It provides capture, problems, summary, estimate, show, start, idle,
    stop_now and hold, and the `run_ended` signal.
  - The Dual-Gate adapter is in `measurements/dgs_queue.py`.
- **Transport and Capacitance** share `GateMapWindow`, so one adapter can cover
  both.
- **Mixed queues.** A queue with entries for several tabs also needs the runner
  to open each entry's tab while the queue holds the tab lock.
- **Position DGS** is not queued.

## Code

- `measurement_queue.py`: entries, queue files and the record (no Qt).
- `measurements/dgs_queue.py`: the Dual-Gate entry fields, checks, summary,
  estimate and adapter.
- `measurements/queue_panel.py`: the runner and the panel.
- `measurements/Dual_gate_sweep_gui.py`:
  - `run_ended`, `queued_run` and `tell_user`;
  - `begin_blocked_reason` and the files' queue attributes;
  - an autosave name that never replaces an earlier file.
- `lab_gui.py`: the dock and its buttons, `measurement_active()` and closing.
- Tests: `tests/test_measurement_queue.py`.

## Potentially dangerous

- A queue runs unattended for hours with the same windows, current limits and
  ramp rates, so check them before starting. The windows and the 100 nA limit
  are not a sample-specific safe envelope.
- Between runs, Begin's ramp moves V1 before V2, so the voltage between the gates
  can briefly exceed both runs'.
- The queue stops at the first error and holds the gates there, possibly for the
  rest of the night.
- Closing the dashboard ends the queue. The run in progress stops, its data is
  autosaved, and later entries do not run.
- The queue does not change the spectrometer or camera between runs, beyond each
  run's exposure.
- Validation used simulated instruments only; no physical instrument was operated.
