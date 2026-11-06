param([string]$Destination = (Join-Path $env:USERPROFILE 'Documents\NIRvana_LN_operation'))
$ErrorActionPreference = 'Stop'
$destinationPath = (Resolve-Path -LiteralPath $Destination).Path
if (-not (Test-Path -LiteralPath (Join-Path $destinationPath 'basi_automation_test.py'))) {
    throw "The existing basi_automation_test.py was not found in $destinationPath. Run this on the NIRvana PC."
}
$listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
if ($listeners | Where-Object { $_.Port -in @(65432, 65433) }) {
    throw 'Finish measurements and stop the old NIRvana server with Ctrl+C before installing. A server port is still in use.'
}
$files = @('NIRvana_device_server.py', 'NIRvana_control.py', 'start_NIRvana_server.cmd')
foreach ($name in $files) {
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $name))) { throw "Missing package file: $name" }
    if ((Join-Path $PSScriptRoot $name) -eq (Join-Path $destinationPath $name)) {
        throw 'Extract the update into a separate folder first; the installer needs the original files intact for backup.'
    }
}
$backupPath = Join-Path $destinationPath ('server-backups\' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $backupPath | Out-Null
# Copy all old files before replacing any file. The automation script and Data stay intact.
foreach ($name in $files) {
    $oldPath = Join-Path $destinationPath $name
    if (Test-Path -LiteralPath $oldPath) { Copy-Item -LiteralPath $oldPath -Destination $backupPath }
}
Copy-Item -LiteralPath (Join-Path $destinationPath 'basi_automation_test.py') -Destination $backupPath
foreach ($name in $files) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination (Join-Path $destinationPath $name) -Force
}
Write-Host "Installed in: $destinationPath"
Write-Host "Original files backed up in: $backupPath"
Write-Host 'Start start_NIRvana_server.cmd in the original operation folder. No hardware was initialized by this installer.'
