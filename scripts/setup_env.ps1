# scripts/setup_env.ps1
# First-time setup for the Email Intent Classifier.
#
# USAGE (right-click this file -> "Run with PowerShell", OR from a terminal):
#   powershell.exe -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
# Supports two install modes (auto-detected):
#   CONDA MODE  -- preferred. Uses conda-forge pre-built llama-cpp-python binary.
#   VENV MODE   -- fallback when conda is absent. Uses Python venv + PyPI wheels.
#
# What this script does:
#   1. Detects conda or Python (>= 3.10) on the system
#   2. Creates the environment  (conda env  -OR-  .venv/ in the project root)
#   3. Installs llama-cpp-python
#   4. Installs all other Python packages from requirements.txt
#   5. Creates required folders  (data/, models/, output/, logs/, prompts/)
#   6. Copies config.example.yaml -> config.yaml  (if config.yaml does not exist)
#   7. Runs the verification script to confirm everything is correct
#
# After this script finishes successfully:
#   - Copy your .gguf model file into the models\ folder
#   - Copy your knowledge base .xlsx file into the data\ folder
#   - Open config.yaml and update model_path and kb_file

$ErrorActionPreference = "Stop"

# ── Helpers ───────────────────────────────────────────────────────────────────

function Write-Banner($msg) {
    Write-Host ""
    Write-Host ("=" * 60)
    Write-Host "  $msg"
    Write-Host ("=" * 60)
}

function Write-Step($n, $msg) {
    Write-Host ""
    Write-Host "  STEP $n : $msg" -ForegroundColor Cyan
    Write-Host ("  " + ("-" * 52))
}

function Write-Pass($msg) { Write-Host "    [PASS] $msg" -ForegroundColor Green  }
function Write-Fail($msg) { Write-Host "    [FAIL] $msg" -ForegroundColor Red    }
function Write-Note($msg) { Write-Host "    [NOTE] $msg" -ForegroundColor Yellow }
function Write-Info($msg) { Write-Host "    [INFO] $msg"                          }

function Pause-AndExit($code) {
    Write-Host ""
    Read-Host "  Press Enter to close this window"
    exit $code
}

# ── Constants ─────────────────────────────────────────────────────────────────

$Root        = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$EnvName     = "email_intent"
$PyVerMajor  = 3
$PyVerMinor  = 10
$LlamaPipPkg = "llama-cpp-python==0.3.16"
$LlamaCondaPkg = "llama-cpp-python=0.3.16"
$VenvDir     = Join-Path $Root ".venv"

Write-Banner "Email Intent Classifier -- First-Time Setup"
Write-Info "Project folder: $Root"

# ── STEP 1 : Detect conda or Python ──────────────────────────────────────────

Write-Step 1 "Detecting environment manager (conda or Python venv)"

$CondaExe  = $null
$UseVenv   = $false
$PythonExe = $null

# Search for conda
$CondaSearchPaths = @(
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:LOCALAPPDATA\conda\conda\Scripts\conda.exe",
    "C:\ProgramData\Anaconda3\Scripts\conda.exe",
    "C:\ProgramData\Miniconda3\Scripts\conda.exe",
    "C:\tools\miniconda3\Scripts\conda.exe"
)

if (Get-Command conda -ErrorAction SilentlyContinue) {
    $CondaExe = "conda"
} else {
    foreach ($p in $CondaSearchPaths) {
        if (Test-Path $p) { $CondaExe = $p; break }
    }
}

if ($CondaExe) {
    Write-Pass "Found conda: $CondaExe  --> using CONDA mode"
} else {
    Write-Note "Conda not found -- checking for Python >= $PyVerMajor.$PyVerMinor ..."

    # Search for a Python 3.10+ executable
    $PySearchNames = @("python3.11", "python3.10", "python3", "python")
    foreach ($name in $PySearchNames) {
        $found = Get-Command $name -ErrorAction SilentlyContinue
        if ($found) {
            $ver = & $found.Source --version 2>&1
            if ($ver -match "Python (\d+)\.(\d+)") {
                $maj = [int]$Matches[1]; $min = [int]$Matches[2]
                if ($maj -gt $PyVerMajor -or ($maj -eq $PyVerMajor -and $min -ge $PyVerMinor)) {
                    $PythonExe = $found.Source
                    Write-Pass "Found $($found.Name) $ver  --> using VENV mode"
                    $UseVenv = $true
                    break
                }
            }
        }
    }

    if (-not $UseVenv) {
        Write-Fail "Neither conda nor Python >= $PyVerMajor.$PyVerMinor was found."
        Write-Host ""
        Write-Host "  You need ONE of the following:"
        Write-Host ""
        Write-Host "  OPTION A -- Install Miniconda (recommended):"
        Write-Host "    https://docs.conda.io/en/latest/miniconda.html"
        Write-Host ""
        Write-Host "  OPTION B -- Install Python 3.11 directly:"
        Write-Host "    https://www.python.org/downloads/"
        Write-Host "    (tick 'Add Python to PATH' during installation)"
        Write-Host ""
        Write-Host "  After installing, re-run this script."
        Pause-AndExit 1
    }
}

# ── STEP 2 : Create the environment ──────────────────────────────────────────

if ($UseVenv) {
    Write-Step 2 "Creating Python virtual environment  (.venv)"

    if (Test-Path $VenvDir) {
        Write-Pass ".venv already exists -- skipping creation."
    } else {
        Write-Info "Running: python -m venv .venv"
        & $PythonExe -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "Failed to create virtual environment."
            Write-Host "  Try manually: python -m venv .venv"
            Pause-AndExit 1
        }
        Write-Pass ".venv created."
    }

    # Point PythonExe at the venv Python from here on
    $PythonExe = Join-Path $VenvDir "Scripts\python.exe"

} else {
    Write-Step 2 "Setting up conda environment '$EnvName'"

    $EnvList   = & $CondaExe env list 2>&1
    $EnvExists = ($EnvList | Select-String -SimpleMatch $EnvName).Count -gt 0

    if ($EnvExists) {
        Write-Pass "Environment '$EnvName' already exists -- skipping creation."
    } else {
        Write-Info "Creating environment (this takes 1-2 minutes) ..."
        & $CondaExe create -n $EnvName python=3.11 -y
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "Failed to create conda environment."
            Write-Host "  Try: conda create -n $EnvName python=3.11 -y"
            Pause-AndExit 1
        }
        Write-Pass "Environment '$EnvName' created."
    }
    # Resolve the conda python path for later steps
    $PythonExe = (& $CondaExe run -n $EnvName python -c "import sys; print(sys.executable)" 2>&1).Trim()
}

# ── STEP 3 : Install llama-cpp-python ────────────────────────────────────────
#
# Both modes use a pre-built binary wheel:
#   CONDA : conda-forge  (most reliable on Windows)
#   VENV  : PyPI wheel   (no C++ compiler required -- pip fetches the .whl)

Write-Step 3 "Installing llama-cpp-python (pre-built binary)"
Write-Info "This may take 3-5 minutes on first install ..."

if ($UseVenv) {
    & $PythonExe -m pip install --upgrade pip -q
    & $PythonExe -m pip install $LlamaPipPkg
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "pip install $LlamaPipPkg failed."
        Write-Host ""
        Write-Host "  If the error mentions 'no matching distribution', your Python"
        Write-Host "  version may not have a pre-built wheel.  Try Python 3.11:"
        Write-Host "    https://www.python.org/downloads/release/python-3110/"
        Pause-AndExit 1
    }
} else {
    & $CondaExe install -n $EnvName -c conda-forge $LlamaCondaPkg -y
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "conda install $LlamaCondaPkg failed."
        Write-Host "  Try: conda install -n $EnvName -c conda-forge $LlamaCondaPkg -y"
        Pause-AndExit 1
    }
}
Write-Pass "llama-cpp-python installed."

# ── STEP 4 : Install remaining packages ──────────────────────────────────────

Write-Step 4 "Installing remaining Python packages"

$ReqFile = Join-Path $Root "requirements.txt"

if ($UseVenv) {
    & $PythonExe -m pip install -r $ReqFile
} else {
    & $CondaExe run -n $EnvName pip install -r $ReqFile
}

if ($LASTEXITCODE -ne 0) {
    Write-Fail "Package installation failed."
    Write-Host ""
    if ($UseVenv) {
        Write-Host "  Try manually:"
        Write-Host "    .venv\Scripts\activate"
        Write-Host "    pip install -r requirements.txt"
    } else {
        Write-Host "  Try manually:"
        Write-Host "    conda activate $EnvName"
        Write-Host "    pip install -r requirements.txt"
    }
    Pause-AndExit 1
}
Write-Pass "All packages installed."

# ── STEP 5 : Create required folders ─────────────────────────────────────────

Write-Step 5 "Creating required folders"

foreach ($f in @("data", "models", "output", "logs", "prompts")) {
    $p = Join-Path $Root $f
    if (Test-Path $p) {
        Write-Info "$f\ already exists -- skipped."
    } else {
        New-Item -ItemType Directory -Path $p | Out-Null
        Write-Pass "Created: $f\"
    }
}

# ── STEP 6 : Create config.yaml if missing ────────────────────────────────────

Write-Step 6 "Checking config.yaml"

$ConfigPath  = Join-Path $Root "config.yaml"
$ExamplePath = Join-Path $Root "config.example.yaml"

if (Test-Path $ConfigPath) {
    Write-Pass "config.yaml already exists."
} elseif (Test-Path $ExamplePath) {
    Copy-Item $ExamplePath $ConfigPath
    Write-Pass "Copied config.example.yaml -> config.yaml"
    Write-Host ""
    Write-Note "You must edit config.yaml before running the classifier."
    Write-Note "Set these two values:"
    Write-Note "  model_path: models\your-model-file.gguf"
    Write-Note "  kb_file:    data\your-knowledge-base.xlsx"
} else {
    Write-Fail "config.example.yaml not found -- cannot create config.yaml."
    Write-Host "  Please re-clone the repository and try again."
}

# ── STEP 7 : Run verify_env.py ────────────────────────────────────────────────

Write-Step 7 "Running environment verification"
Write-Info "Python: $PythonExe"

& $PythonExe "$Root\scripts\verify_env.py"

# ── Done ──────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Banner "Setup complete!"
Write-Host ""
Write-Host "  NEXT STEPS:" -ForegroundColor Cyan
Write-Host ""
Write-Host "    1. Copy your model file (.gguf) into:"
Write-Host "         $Root\models\"
Write-Host ""
Write-Host "    2. Copy your knowledge base (.xlsx) into:"
Write-Host "         $Root\data\"
Write-Host ""
Write-Host "    3. Open config.yaml and set model_path and kb_file."
Write-Host ""
Write-Host "    4. Activate your environment and run the classifier:"
Write-Host ""
if ($UseVenv) {
    Write-Host "         .venv\Scripts\activate" -ForegroundColor White
} else {
    Write-Host "         conda activate $EnvName" -ForegroundColor White
}
Write-Host "         python claude.py --config config.yaml" -ForegroundColor White
Write-Host ""

Pause-AndExit 0
