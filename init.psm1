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
    CL /EHsc src\*.cc /Fo:bin\ /Fe:bin\main.exe
}

function Clean-RayTracer {
    Delete-DirectoryIfExists -Path bin
}