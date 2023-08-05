function Delete-DirectoryIfExists {
    param (
        [parameter(ValueFromPipeline)]
        [ValidateNotNull()]
        [string]$Path
    )

    if (Test-Path $Path) {
        Write-Host "Deleting file '$Path'"
        Remove-Item -Path $Path -Recurse
    }
}

function Create-DirectoryIfNotExists {
    param (
        [parameter(ValueFromPipeline)]
        [ValidateNotNull()]
        [string]$Path
    )

    if (!(Test-Path $Path)) {
        Write-Host "Creating directory '$Path'"
        New-Item -Path $Path -Type Directory | Out-Null
    }
}

function Build-RayTracer {
    Create-DirectoryIfNotExists -Path bin
    CL /EHsc /Fo:"bin\" /Fe:"bin\main.exe" /O2 /Wall /wd4711 /wd4710 /wd5045 src\*.cc
}

function Render-Output {
    param (
        [parameter(ValueFromPipeline)]
        [ValidateNotNull()]
        [string]$Filename = "output.ppm"
    )

    (.\bin\main.exe) -join "`n" | Out-File -FilePath ".\bin\$Filename" -Encoding ascii
}

function Clean-RayTracer {
    Delete-DirectoryIfExists -Path bin
}