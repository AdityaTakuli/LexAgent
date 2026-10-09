' Runs keep_running.ps1 without showing a console window (used by the scheduled task).
Set shell = CreateObject("WScript.Shell")
folder = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
shell.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & folder & "\keep_running.ps1""", 0, True
