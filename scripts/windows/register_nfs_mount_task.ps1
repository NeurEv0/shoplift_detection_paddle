# register_nfs_mount_task.ps1
# Registers a scheduled task that mounts NFS shares at boot/logon and links them
# into the repo. The task runs NON-ELEVATED (RunLevel Limited) on purpose:
# Windows NFS drive letters are per logon session - mounts made in an elevated
# session are invisible to Explorer/normal apps ("位置不可用"). mount.exe works
# without elevation; symlinks are created once by running link_nfs_to_repo.ps1
# as admin (they persist; the task only re-mounts and re-applies git tweaks).
$ErrorActionPreference = 'Stop'
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$scriptPath = Join-Path $PSScriptRoot 'mount_nfs_shares.ps1'
$taskName = 'ShopliftNFSMount'
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`""
$triggerBoot = New-ScheduledTaskTrigger -AtStartup
$triggerLogon = New-ScheduledTaskTrigger -AtLogOn -User $user
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger @($triggerBoot, $triggerLogon) -Settings $settings -Principal $principal -Force | Out-Null
Write-Output "Task registered: $taskName (repo: $repo, RunLevel: Limited/non-elevated)"
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State | Format-List
