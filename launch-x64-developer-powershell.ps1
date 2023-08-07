$currentPath = Get-Location
$path = Read-Host "Please enter full path to 'Launch-VsDevShell.ps1'"
& "$path" -Arch arm64 -HostArch amd64
Set-Location $currentPath
Import-Module -Force .\init.psm1