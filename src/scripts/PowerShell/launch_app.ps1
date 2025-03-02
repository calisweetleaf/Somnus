param(
    [Parameter(Mandatory=$true)]
    [string]$App
)

try {
    if ([string]::IsNullOrWhiteSpace($App)) {
        throw "Application name/path cannot be empty"
    }
    
    # If it's a known program name, let Start-Process find it in PATH or default locations
    Start-Process -FilePath $App -ErrorAction Stop
    Write-Output "SUCCESS: Launched application `"$App`"."
} catch {
    Write-Output "ERROR: Failed to launch `"$App`": $($_.Exception.Message)"
    exit 1
}