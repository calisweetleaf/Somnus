param(
    [Parameter(Mandatory=$true)]
    [string]$Path
)

try {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "Path cannot be empty"
    }

    if (!(Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force -ErrorAction Stop | Out-Null
        Write-Output "SUCCESS: Created folder `"$Path`""
    } else {
        Write-Output "INFO: Folder `"$Path`" already exists."
    }
} catch {
    Write-Output "ERROR: $($_.Exception.Message)"
    exit 1
}