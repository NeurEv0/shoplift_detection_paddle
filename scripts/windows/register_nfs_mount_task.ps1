# register_nfs_mount_task.ps1 (run elevated)
# Registers a scheduled task that mounts NFS shares at boot/logon (runs elevated, delayed 30s).
$ErrorActionPreference = 'Stop'
$scriptPath = 'C:\Users\TaoJing\Desktop\超市\shoplift_detection_paddle\scripts\windows\mount_nfs_shares.ps1'
$taskName = 'ShopliftNFSMount'

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`""
$triggerBoot = New-ScheduledTaskTrigger -AtStartup
$triggerLogon = New-ScheduledTaskTrigger -AtLogOn -User 'DESKTOP-62RCR64\TaoJing'
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$principal = New-ScheduledTaskPrincipal -UserId 'DESKTOP-62RCR64\TaoJing' -LogonType Interactive -RunLevel Highest

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger @($triggerBoot, $triggerLogon) -Settings $settings -Principal $principal -Force | Out-Null
Write-Output "Task registered: $taskName"
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State | Format-List
