# scripts/setup_env.ps1
# First-time setup for the Email Intent Classifier.
#
# USAGE (right-click this file -> "Run with PowerShell", OR from a terminal):
#   powershell.exe -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
# Optional parameters:
#   -ExistingPython   "C:\path\to\python.exe"   Use an already-configured Python
#                                                (any venv, conda env, system Python).
#                                                Skips environment creation entirely.
#   -ExistingCondaEnv "env_name"                Use an existing conda environment.
#                                                Skips environment creation entirely.
#   -LlamaWhl         "C:\path\to\*.whl"        Install llama-cpp-python from a local
#                                                .whl file instead of downloading it.
#
# Examples:
#   # Re-use a venv you already have:
#   powershell.exe -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -ExistingPython ".venv\Scripts\python.exe"
#
#   # Re-use a colleague's conda environment:
#   powershell.exe -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -ExistingCondaEnv "my_nlp_env"
#
#   # Install llama-cpp-python from a downloaded .whl (no internet needed):
#   powershell.exe -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -LlamaWhl "C:\Downloads\llama_cpp_python-0.3.16-cp311-win_amd64.whl"
#
# Supports three install modes (auto-detected unless overridden):
#   CONDA MODE      -- preferred. conda-forge pre-built llama-cpp-python binary.
#   VENV MODE       -- fallback when conda is absent. Python venv + PyPI wheels.
#   EXISTING ENV    -- skip env creation; use a Python / conda env you already have.
#
# What this script does:
#   1. Detects conda or Python (>= 3.10)  [skipped when -ExistingPython/-ExistingCondaEnv used]
#   2. Creates the environment            [skipped when existing env provided]
#   3. Installs llama-cpp-python          (from -LlamaWhl, wheels\ folder, or network)
#   4. Installs all other Python packages from requirements.txt
#   5. Creates required folders  (data/, models/, output/, logs/, prompts/)
#   6. Copies config.example.yaml -> config.yaml  (if config.yaml does not exist)
#   7. Runs the verification script to confirm everything is correct
#
# After this script finishes successfully:
#   - Copy your .gguf model file into the models\ folder
#   - Copy your knowledge base .xlsx file into the data\ folder
#   - Open config.yaml and update model_path and kb_file

param(
    [string]$ExistingPython   = "",   # path to an existing python.exe
    [string]$ExistingCondaEnv = "",   # name of an existing conda environment
    [string]$LlamaWhl         = ""    # path to a local llama_cpp_python .whl file
)

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

# Resolve a local llama-cpp-python wheel.
# -LlamaWhl parameter takes priority; otherwise auto-scan project root and wheels\ subfolder.
if ($LlamaWhl -and -not (Test-Path $LlamaWhl)) {
    Write-Host "  [FAIL] -LlamaWhl path not found: $LlamaWhl" -ForegroundColor Red
    exit 1
}
if (-not $LlamaWhl) {
    foreach ($scanDir in @($Root, (Join-Path $Root "wheels"))) {
        if (Test-Path $scanDir) {
            $hit = Get-ChildItem -Path $scanDir -Filter "llama_cpp_python*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($hit) { $LlamaWhl = $hit.FullName; break }
        }
    }
    if ($LlamaWhl) {
        Write-Host "    [INFO] Found local wheel: $LlamaWhl"
    }
}

Write-Banner "Email Intent Classifier -- First-Time Setup"
Write-Info "Project folder: $Root"

# ── STEP 1 : Detect conda or Python ──────────────────────────────────────────

Write-Step 1 "Detecting environment manager (conda or Python venv)"

$CondaExe      = $null
$UseVenv       = $false
$PythonExe     = $null
$SkipEnvCreate = $false

$CondaSearchPaths = @(
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:LOCALAPPDATA\conda\conda\Scripts\conda.exe",
    "C:\ProgramData\Anaconda3\Scripts\conda.exe",
    "C:\ProgramData\Miniconda3\Scripts\conda.exe",
    "C:\tools\miniconda3\Scripts\conda.exe"
)

# ---- Existing Python executable supplied via -ExistingPython -----------------
if ($ExistingPython) {
    $resolved = if ([System.IO.Path]::IsPathRooted($ExistingPython)) {
        $ExistingPython
    } else {
        Join-Path $Root $ExistingPython
    }
    if (-not (Test-Path $resolved)) {
        Write-Fail "-ExistingPython not found: $resolved"
        Pause-AndExit 1
    }
    $PythonExe     = $resolved
    $UseVenv       = $true
    $SkipEnvCreate = $true
    $ver = & $PythonExe --version 2>&1
    Write-Pass "Using existing Python ($ver): $PythonExe"

# ---- Existing conda environment supplied via -ExistingCondaEnv ---------------
} elseif ($ExistingCondaEnv) {
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        $CondaExe = "conda"
    } else {
        foreach ($p in $CondaSearchPaths) {
            if (Test-Path $p) { $CondaExe = $p; break }
        }
    }
    if (-not $CondaExe) {
        Write-Fail "-ExistingCondaEnv was specified but conda could not be found."
        Pause-AndExit 1
    }
    $envList = & $CondaExe env list 2>&1
    if (($envList | Select-String -SimpleMatch $ExistingCondaEnv).Count -eq 0) {
        Write-Fail "Conda environment '$ExistingCondaEnv' does not exist."
        Write-Host "  Available environments:"
        $envList | Where-Object { $_ -match "^\S" } | ForEach-Object { Write-Host "    $_" }
        Pause-AndExit 1
    }
    $EnvName       = $ExistingCondaEnv
    $SkipEnvCreate = $true
    $PythonExe     = (& $CondaExe run -n $EnvName python -c "import sys; print(sys.executable)" 2>&1).Trim()
    Write-Pass "Using existing conda environment: $EnvName  ($PythonExe)"

# ---- Auto-detect conda or Python ---------------------------------------------
} else {
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
}

# ── STEP 2 : Create the environment ──────────────────────────────────────────

if ($SkipEnvCreate) {
    Write-Step 2 "Environment creation"
    Write-Pass "Skipped -- using existing environment."

} elseif ($UseVenv) {
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
# Priority order:
#   1. Local .whl file  (-LlamaWhl param, or auto-found in project root / wheels\)
#   2. conda-forge      (conda mode, requires network)
#   3. PyPI wheel       (venv / existing-Python mode, requires network)
#
# When re-using an existing environment, install is skipped if llama_cpp already works.

Write-Step 3 "Installing llama-cpp-python"

# Check if llama_cpp is already importable (saves time when re-using an existing env)
$checkResult = & $PythonExe -c "import llama_cpp; print('ok')" 2>&1
$LlamaAlreadyInstalled = ($checkResult -match "^ok")

if ($LlamaAlreadyInstalled -and -not $LlamaWhl) {
    Write-Pass "llama-cpp-python is already installed -- skipping."

} elseif ($LlamaWhl) {
    Write-Info "Installing from local wheel: $LlamaWhl"
    & $PythonExe -m pip install --upgrade pip -q
    & $PythonExe -m pip install $LlamaWhl
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "pip install from wheel failed: $LlamaWhl"
        Pause-AndExit 1
    }
    Write-Pass "llama-cpp-python installed from local wheel."

} elseif ($UseVenv -or $SkipEnvCreate) {
    Write-Info "Downloading from PyPI (this may take 3-5 minutes) ..."
    & $PythonExe -m pip install --upgrade pip -q
    & $PythonExe -m pip install $LlamaPipPkg
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "pip install $LlamaPipPkg failed."
        Write-Host ""
        Write-Host "  If the error mentions 'no matching distribution', your Python"
        Write-Host "  version may not have a pre-built wheel.  Try Python 3.11:"
        Write-Host "    https://www.python.org/downloads/release/python-3110/"
        Write-Host ""
        Write-Host "  Alternatively, download a pre-built .whl from:"
        Write-Host "    https://github.com/abetlen/llama-cpp-python/releases"
        Write-Host "  then re-run with:  -LlamaWhl path\to\file.whl"
        Pause-AndExit 1
    }
    Write-Pass "llama-cpp-python installed."

} else {
    Write-Info "Installing from conda-forge (this may take 3-5 minutes) ..."
    & $CondaExe install -n $EnvName -c conda-forge $LlamaCondaPkg -y
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "conda install $LlamaCondaPkg failed."
        Write-Host "  Try: conda install -n $EnvName -c conda-forge $LlamaCondaPkg -y"
        Write-Host ""
        Write-Host "  Or download a pre-built .whl from:"
        Write-Host "    https://github.com/abetlen/llama-cpp-python/releases"
        Write-Host "  then re-run with:  -LlamaWhl path\to\file.whl"
        Pause-AndExit 1
    }
    Write-Pass "llama-cpp-python installed."
}

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
