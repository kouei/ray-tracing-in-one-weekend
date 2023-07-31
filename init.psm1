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
    CL /EHsc /Fo"bin\\" /Fe"bin\\main.exe" src\*.cc
}

function Clean-RayTracer {
    Delete-DirectoryIfExists -Path bin
}