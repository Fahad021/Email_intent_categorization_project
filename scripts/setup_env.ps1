# scripts/setup_env.ps1
# First-time setup for the Email Intent Classifier.
#
# USAGE (right-click this file -> "Run with PowerShell", OR from a terminal):
#   powershell.exe -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
# What this script does:
#   1. Finds (or asks you to install) conda / Miniconda
#   2. Creates the 'email_intent' conda environment
#   3. Installs llama-cpp-python from conda-forge (pre-built, no C++ compiler needed)
#   4. Installs all other Python packages
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
    Write-Host ("=" * 56)
    Write-Host "  $msg"
    Write-Host ("=" * 56)
}

function Write-Step($n, $msg) {
    Write-Host ""
    Write-Host "  STEP $n : $msg" -ForegroundColor Cyan
    Write-Host ("  " + ("-" * 50))
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

# ── Project root (two levels above this script) ───────────────────────────────

$Root    = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$EnvName = "email_intent"
$PyVer   = "3.11"
$LlamaPkg = "llama-cpp-python=0.3.16"

$OtherPackages = @(
    "pandas>=1.5.0",
    "openpyxl>=3.0.0",
    "pyodbc>=4.0.0",
    "sqlalchemy>=2.0.0",
    "tqdm>=4.60.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
    "pyarrow>=14.0.0"
)

Write-Banner "Email Intent Classifier -- First-Time Setup"
Write-Info "Project folder: $Root"

# ── STEP 1 : Find conda ───────────────────────────────────────────────────────

Write-Step 1 "Finding conda"

$CondaExe = $null
$SearchPaths = @(
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
    foreach ($p in $SearchPaths) {
        if (Test-Path $p) { $CondaExe = $p; break }
    }
}

if (-not $CondaExe) {
    Write-Fail "Conda was not found on this machine."
    Write-Host ""
    Write-Host "  Conda (Miniconda) is required to run this project."
    Write-Host "  It is free, lightweight, and takes about 5 minutes to install."
    Write-Host ""
    Write-Host "  Download Miniconda here:"
    Write-Host "    https://docs.conda.io/en/latest/miniconda.html"
    Write-Host ""
    Write-Host "  After installing, close and reopen this window, then run this script again."
    Pause-AndExit 1
}

Write-Pass "Found conda: $CondaExe"

# ── STEP 2 : Create conda environment ─────────────────────────────────────────

Write-Step 2 "Setting up conda environment '$EnvName' (Python $PyVer)"

$EnvList  = & $CondaExe env list 2>&1
$EnvExists = ($EnvList | Select-String -SimpleMatch $EnvName).Count -gt 0

if ($EnvExists) {
    Write-Pass "Environment '$EnvName' already exists -- skipping creation."
} else {
    Write-Info "Creating environment (this takes about 1-2 minutes) ..."
    & $CondaExe create -n $EnvName python=$PyVer -y
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Failed to create conda environment."
        Write-Host ""
        Write-Host "  Try running this command manually and check for errors:"
        Write-Host "    conda create -n $EnvName python=$PyVer -y"
        Pause-AndExit 1
    }
    Write-Pass "Environment '$EnvName' created."
}

# ── STEP 3 : Install llama-cpp-python from conda-forge ────────────────────────
#
# IMPORTANT: llama-cpp-python must be installed via conda-forge, not pip.
# The conda-forge build is a pre-compiled binary -- no C++ compiler is needed.
# Installing via pip would try to compile from source and fail on most machines.

Write-Step 3 "Installing llama-cpp-python (pre-built binary from conda-forge)"
Write-Info "This may take 3-5 minutes on first install ..."

& $CondaExe install -n $EnvName -c conda-forge $LlamaPkg -y
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Failed to install llama-cpp-python."
    Write-Host ""
    Write-Host "  Try running this command manually:"
    Write-Host "    conda install -n $EnvName -c conda-forge $LlamaPkg -y"
    Pause-AndExit 1
}
Write-Pass "llama-cpp-python installed."

# ── STEP 4 : Install remaining packages via pip ───────────────────────────────

Write-Step 4 "Installing remaining Python packages"

& $CondaExe run -n $EnvName pip install @OtherPackages
if ($LASTEXITCODE -ne 0) {
    Write-Fail "pip install failed."
    Write-Host ""
    Write-Host "  Try running these commands manually:"
    Write-Host "    conda activate $EnvName"
    Write-Host "    pip install -r requirements.txt"
    Pause-AndExit 1
}
Write-Pass "All packages installed."

# ── STEP 5 : Create required folders ─────────────────────────────────────────

Write-Step 5 "Creating required folders"

$Folders = @("data", "models", "output", "logs", "prompts")
foreach ($f in $Folders) {
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
    Write-Host "  Please contact the project maintainer or re-clone the repository."
}

# ── STEP 7 : Run verify_env.py ────────────────────────────────────────────────

Write-Step 7 "Running environment verification"

$PythonExe = & $CondaExe run -n $EnvName python -c "import sys; print(sys.executable)" 2>&1
Write-Info "Python: $PythonExe"

& $CondaExe run -n $EnvName python "$Root\scripts\verify_env.py"

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
Write-Host "    4. Run the classifier:"
Write-Host "         conda activate $EnvName"
Write-Host "         python claude.py --config config.yaml --kb data\your_kb.xlsx ..."
Write-Host ""

Pause-AndExit 0
