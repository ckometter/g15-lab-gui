# NIRvana grating, wavelength and live-preview update

Install this package **on the NIRvana PC**, in the existing setup shown in the screenshot:
`C:\Users\QPG\Documents\NIRvana_LN_operation`.
The existing `basi_automation_test.py`, `.nirvana_venv`, LightField experiment and Data folder are retained.

1. Finish the current measurement. In the old server console, press Ctrl+C and wait for its cleanup to finish.
2. Extract this update ZIP into a **separate folder** on the NIRvana PC (for example Downloads).
3. Run `install_on_nirvana_pc.cmd` from that separate folder. It checks that the ports are idle, backs up the original scripts under `NIRvana_LN_operation\server-backups\<timestamp>`, and installs the server/helper/launcher. It does not start hardware.
4. In the original `NIRvana_LN_operation` folder, run `start_NIRvana_server.cmd`. It uses the existing `.nirvana_venv`. If the existing server used a different Python environment, run `python NIRvana_device_server.py` there instead.
5. Wait for the console to report acquisition on **65432** and controls on **65433**. If Windows Firewall restricts the existing Python program by port, allow TCP 65433 from the measurement PC on the laboratory network, alongside the existing 65432 rule.
6. In Device Manager on the measurement PC, check/reconnect the NIRvana camera and monochromator. Open Single Spectra. The displayed grating list and centre wavelength should come from LightField. Select the desired grating or enter the required wavelength; the dashboard waits for LightField readback. There is **no automatic grating movement at connection or startup**.

For a different destination, use PowerShell:

```powershell
.\install_on_nirvana_pc.ps1 -Destination 'C:\your\NIRvana_LN_operation'
```

## What changed

- Original acquisition clients still send `start_acquisition` on 65432 and receive CSV until the connection closes. Exposure is still milliseconds on the wire. The original experiment's file-export settings apply to recorded acquisitions.
- A separate, versioned JSON service on 65433 supports status, exact grating selection, centre wavelength, and native LightField preview. The GUI never sends control probes to an old acquisition server.
- Grating identifiers come from `GetCurrentCapabilities(GratingSelected)` (or `Grating` on older LightField). Missing settings remain unavailable. Requests are checked with `IsValid` and then read back with `GetValue`.
- Live preview uses `Preview()` and `ImageDataSetReceived`, not `Acquire()` or CSV. It writes no spectra on either PC. For a full image, detector rows are summed to produce the alignment spectrum. One ROI and a matching LightField column calibration are required. A disconnected preview client expires after 15 seconds.
- Acquiring, previewing and spectrometer changes cannot overlap. A busy request fails instead of queuing a later movement. Unknown commands do not delete temporary CSV files. Motion requests are never automatically retried.
- The grating and wavelength readbacks report LightField's values; independent mechanical positioning has **not** been verified. Offline tests use simulated LightField and loopback TCP only.

If controls are unavailable, check the server console, port 65433 and that the loaded LightField experiment includes the spectrometer. Ordinary acquisition remains available through 65432. The updated GUI disables no-file live preview until the new service is available.

## Restore the previous server

Stop the new server with Ctrl+C after measurements finish. Copy `NIRvana_device_server.py` from the timestamped `server-backups` folder back into `NIRvana_LN_operation`, then start it using the previous command/environment. The original automation script and measurement data were never overwritten. The extra helper file can remain unused.

The GUI source before this update is separately archived on the measurement PC under `C:\G-15_qcodes_labGUI\gui-history\before-nirvana-controls-20260929_074147.zip`.

API reference checked against the [areaDetector LightField driver's source](https://github.com/areaDetector/ADLightField/blob/master/lightFieldApp/src/LightField.cpp) and the existing G15 LightField server; the installed LightField version still needs a hardware check after deployment.
