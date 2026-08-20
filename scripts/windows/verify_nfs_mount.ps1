# verify_nfs_mount.ps1
# Verifies the 4 NFS mounts: visibility, read-only enforcement, and rw behavior of datasets_annotation.
$ErrorActionPreference = 'Continue'
$log = Join-Path $env:TEMP 'nfs_verify.log'
function Log($msg) { $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg; $line | Tee-Object -FilePath $log -Append }

Log "=== NFS mount verification ==="

$mountExe = "$env:SystemRoot\System32\mount.exe"
$mounted = (& $mountExe 2>&1 | Out-String)
Log "--- mount list ---"
$mounted -split "`r?`n" | Where-Object { $_ } | ForEach-Object { Log "  $_" }

$targets = @(
    @{ Letter = 'Z:'; Label = 'datasets (ro)' },
    @{ Letter = 'Y:'; Label = 'models (ro)' },
    @{ Letter = 'X:'; Label = 'outputs (ro)' },
    @{ Letter = 'W:'; Label = 'datasets_annotation (rw)' }
)

foreach ($t in $targets) {
    $ok = Test-Path "$($t.Letter)\"
    Log "--- $($t.Label) -> $($t.Letter) exists: $ok ---"
    if ($ok) {
        $entries = Get-ChildItem "$($t.Letter)\" -ErrorAction SilentlyContinue | Select-Object -First 8
        foreach ($e in $entries) { Log "    $($e.Name)$(if ($e.PSIsContainer) {'/'} else {''})" }
    }
}

# Write test into the rw share
$testFile = 'W:\__windows_nfs_write_test__.txt'
try {
    Set-Content -Path $testFile -Value ("write-test " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding utf8 -ErrorAction Stop
    Log "RW TEST: write OK -> $testFile"
    $content = Get-Content $testFile -ErrorAction SilentlyContinue
    Log "RW TEST: read back: $content"
    Remove-Item $testFile -ErrorAction SilentlyContinue
    Log "RW TEST: cleanup OK (file removed)"
} catch {
    Log "RW TEST: FAILED -> $($_.Exception.Message)"
}

# Write test into a ro share (should fail)
try {
    Set-Content -Path 'Z:\__windows_ro_write_test__.txt' -Value 'x' -ErrorAction Stop
    Log "RO TEST: UNEXPECTED SUCCESS writing to Z: (datasets should be read-only!)"
    Remove-Item 'Z:\__windows_ro_write_test__.txt' -ErrorAction SilentlyContinue
} catch {
    Log "RO TEST: write blocked as expected: $($_.Exception.Message)"
}

Log "=== DONE ==="
