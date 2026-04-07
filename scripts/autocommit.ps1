<#
.SYNOPSIS
    Auto-commit all changes with a generated conventional-commit message.
.EXAMPLE
    .\scripts\autocommit.ps1
    .\scripts\autocommit.ps1 -Message "fix: manual override"
#>
param(
    [string]$Message = "",
    [string]$GitExe  = "C:\Users\abulh\AppData\Local\Programs\Git\cmd\git.exe"
)

Set-Location (Split-Path $PSScriptRoot)

if (-not (Test-Path $GitExe)) {
    $GitExe = "git"
}

function Invoke-Git { & $GitExe @args }

# --- Check for changes -----------------------------------------------
$porcelain = Invoke-Git status --porcelain 2>&1
if (-not $porcelain) {
    Write-Host "Nothing to commit - working tree clean." -ForegroundColor Yellow
    exit 0
}

# --- Stage everything ------------------------------------------------
Invoke-Git add -A | Out-Null

# --- Gather changed file list ----------------------------------------
$files    = @(Invoke-Git diff --cached --name-only 2>&1)
$count    = $files.Count
$statLine = Invoke-Git diff --cached --stat --no-color 2>&1 | Select-Object -Last 1

if ($count -eq 0) {
    Write-Host "Nothing staged after git add -A. Aborting." -ForegroundColor Red
    exit 1
}

# --- Generate commit message if none supplied ------------------------
if (-not $Message) {

    $docs   = @()
    $prompt = @()
    $config = @()
    $py     = @()
    $other  = @()

    foreach ($f in $files) {
        if     ($f -match '\.(md|rst|txt)$' -and $f -notmatch '^prompts/') { $docs   += $f }
        elseif ($f -match '^prompts/')                                      { $prompt += $f }
        elseif ($f -match '(config.*\.yaml|\.env)')                        { $config += $f }
        elseif ($f -match '\.py$')                                         { $py     += $f }
        else                                                                { $other  += $f }
    }

    $type = "chore"
    if     ($prompt.Count -gt 0 -and $py.Count -eq 0 -and $config.Count -eq 0) { $type = "prompt"   }
    elseif ($docs.Count   -gt 0 -and $py.Count -eq 0 -and $config.Count -eq 0) { $type = "docs"     }
    elseif ($config.Count -gt 0 -and $py.Count -eq 0)                          { $type = "config"   }
    elseif ($py.Count     -gt 0) {
        $newPy = @(Invoke-Git diff --cached --diff-filter=A --name-only 2>&1) |
                 Where-Object { $_ -match '\.py$' }
        $type  = if ($newPy.Count -gt 0) { "feat" } else { "refactor" }
    }

    if ($count -eq 1) {
        $desc = Split-Path $files[0] -Leaf
    } elseif ($count -le 4) {
        $desc = ($files | ForEach-Object { Split-Path $_ -Leaf }) -join ", "
    } else {
        $first = Split-Path $files[0] -Leaf
        $desc  = "$first and $($count - 1) others"
    }

    $Message = "${type}: ${desc}"
}

# --- Commit ----------------------------------------------------------
$result = Invoke-Git commit -m $Message 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] $Message" -ForegroundColor Green
    Write-Host "     $statLine" -ForegroundColor DarkGray
} else {
    Write-Host "Commit failed:" -ForegroundColor Red
    Write-Host $result
    exit 1
}
