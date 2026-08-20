# mount_nfs_shares.ps1
# Mounts the 4 NFS shares from 10.200.10.10 (GPU host) onto drive letters.
# Designed to run from Task Scheduler at boot/logon: waits for network, retries mounts.
# NOTE: uses mount.exe explicitly because PowerShell aliases `mount` to New-PSDrive.
$ErrorActionPreference = 'Continue'
$log = Join-Path $env:TEMP 'nfs_mount.log'
function Log($msg) { $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg; $line | Tee-Object -FilePath $log -Append }

Log "=== NFS mount attempt (elevated: $([Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))) ==="

$mountExe = "$env:SystemRoot\System32\mount.exe"
if (-not (Test-Path $mountExe)) {
    Log "FATAL: $mountExe not found - NFS client not installed? (feature enabled but reboot may be required)"
    exit 1
}

# Wait for the network stack (up to 60s) so a boot-time run does not race the NIC.
for ($i = 1; $i -le 12; $i++) {
    if (Test-Connection -ComputerName 10.200.10.10 -Count 1 -Quiet -ErrorAction SilentlyContinue) { Log "ping 10.200.10.10 OK (attempt $i)"; break }
    Log "ping 10.200.10.10 failed (attempt $i), waiting 5s..."
    Start-Sleep -Seconds 5
}

$shares = @(
    @{ Letter = 'Z:'; Unc = '\\10.200.10.10\home\ubuntu\data_1t\shoplift_detection_paddle\datasets' },
    @{ Letter = 'Y:'; Unc = '\\10.200.10.10\home\ubuntu\data_1t\shoplift_detection_paddle\models' },
    @{ Letter = 'X:'; Unc = '\\10.200.10.10\home\ubuntu\data_1t\shoplift_detection_paddle\outputs' },
    @{ Letter = 'W:'; Unc = '\\10.200.10.10\home\ubuntu\data_1t\shoplift_detection_paddle\datasets_annotation' }
)

foreach ($s in $shares) {
    $mounted = $false
    for ($try = 1; $try -le 3 -and -not $mounted; $try++) {
        $existing = & $mountExe 2>&1 | Select-String -Pattern ('^' + [regex]::Escape($s.Letter))
        if ($existing) {
            Log "SKIP $($s.Letter) already mounted: $($existing.Line.Trim())"
            $mounted = $true
            break
        }
        Log "MOUNT (try $try) $($s.Letter) <- $($s.Unc)"
        $out = & $mountExe -o anon $s.Unc $s.Letter 2>&1
        $code = $LASTEXITCODE
        foreach ($l in $out) { Log "  out: $l" }
        Log "  exit: $code"
        if ($code -eq 0) { $mounted = $true }
        else { Start-Sleep -Seconds 10 }
    }
}

Log "=== Current mounts ==="
& $mountExe 2>&1 | ForEach-Object { Log "  $_" }
Log "=== DONE ==="
