
<#
.SYNOPSIS
    Trains multiple states in parallel using separate Python processes.
    6 states at a time — optimised for i9 + 16 GB + 4070.

.USAGE
    .\scripts\parallel_train.ps1
    .\scripts\parallel_train.ps1 -Workers 4
    .\scripts\parallel_train.ps1 -States "Texas","California"
#>
param(
    [int]$Workers = 6,
    [string[]]$States = @(
        "Missouri","Nebraska","Nevada","New Hampshire","New Mexico","New York",
        "North Carolina","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island",
        "South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont",
        "Virginia","Washington","West Virginia","Wisconsin","Wyoming"
    ),
    [int]$CvSplits = 2
)

$PYTHON   = "D:\Professional\microgcc\gcc_env\python.exe"
$WORKDIR  = "D:\Professional\microgcc"
$LOGDIR   = "$WORKDIR\logs\parallel_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

$total    = $States.Count
$done     = 0
$failed   = @()
$succeeded= @()

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host " Parallel state training  |  Workers: $Workers" -ForegroundColor Cyan
Write-Host " States: $total  |  CV splits: $CvSplits" -ForegroundColor Cyan
Write-Host " Logs -> $LOGDIR" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Batch states into groups of $Workers
$batches = @()
for ($i = 0; $i -lt $States.Count; $i += $Workers) {
    $batches += ,($States[$i..([Math]::Min($i + $Workers - 1, $States.Count - 1))])
}

$batchNum = 0
foreach ($batch in $batches) {
    $batchNum++
    $batchStart = Get-Date
    Write-Host "--- Batch $batchNum/$($batches.Count): $($batch -join ', ')" -ForegroundColor Yellow

    $jobs = @()
    foreach ($state in $batch) {
        $slug    = $state.ToLower() -replace '\s+','_'
        $logFile = "$LOGDIR\${slug}.log"

        $proc = Start-Process -FilePath $PYTHON `
            -ArgumentList "-m src.pipeline.train --data data.csv --state `"$state`" --config config\training_config.yaml --cv-splits $CvSplits" `
            -WorkingDirectory $WORKDIR `
            -RedirectStandardError  $logFile `
            -RedirectStandardOutput "$LOGDIR\${slug}_out.log" `
            -PassThru `
            -WindowStyle Hidden

        $jobs += [PSCustomObject]@{ Process=$proc; State=$state; Log=$logFile }
        Write-Host "  Started $state (PID $($proc.Id))" -ForegroundColor Gray
    }

    # Wait for entire batch
    foreach ($job in $jobs) {
        $job.Process.WaitForExit()
        $done++
        if ($job.Process.ExitCode -eq 0) {
            $succeeded += $job.State
            # Extract champion from log
            $champ = Select-String -Path $job.Log -Pattern "Champion selected.*champion=(\w+)" |
                     Select-Object -Last 1 |
                     ForEach-Object { if ($_ -match "champion=(\w+)") { $Matches[1] } }
            Write-Host "  [OK ] $($job.State) -> $champ" -ForegroundColor Green
        } else {
            $failed += $job.State
            Write-Host "  [ERR] $($job.State) - see $($job.Log)" -ForegroundColor Red
        }
    }

    $elapsed = [Math]::Round(((Get-Date) - $batchStart).TotalSeconds)
    Write-Host "  Batch done in ${elapsed}s  ($done/$total total)" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host " DONE  Succeeded: $($succeeded.Count)  Failed: $($failed.Count)" -ForegroundColor Cyan
if ($failed.Count -gt 0) {
    Write-Host " Failed: $($failed -join ', ')" -ForegroundColor Red
}
Write-Host "=================================================" -ForegroundColor Cyan
