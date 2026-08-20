# register_nfs_mount_task.ps1 (run elevated)
# Registers a scheduled task that mounts NFS shares at boot/logon and links them
# into the repo (runs elevated, with a startup delay for network readiness).
$ErrorActionPreference = 'Stop'
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$scriptPath = Join-Path $PSScriptRoot 'mount_nfs_shares.ps1'
$taskName = 'ShopliftNFSMount'
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`""
$triggerBoot = New-ScheduledTaskTrigger -AtStartup
$triggerLogon = New-ScheduledTaskTrigger -AtLogOn -User $user
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Highest

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger @($triggerBoot, $triggerLogon) -Settings $settings -Principal $principal -Force | Out-Null
Write-Output "Task registered: $taskName (repo: $repo)"
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State | Format-List
