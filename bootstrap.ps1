<#
  Build the isolated Stable Audio 3 sidecar environment on Windows.

  Creates .\.venv with a CUDA torch + the stable-audio-3 package, fully separate
  from the main OmniVoice venv (which pins a different torch). Re-runnable.

  Usage (normally invoked via bootstrap.bat):
    .\bootstrap.bat                 # build the env (no model download)
    .\bootstrap.bat --with-model    # also download SA3 Medium (needs HF token)
    # or directly:
    powershell -ExecutionPolicy Bypass -File bootstrap.ps1 [--with-model]

  Overrides (env): SA3_CUDA=cu126  SA3_TORCH=2.7.1  SA3_PYTHON=3.10
                   SA3_FLASH_WHEEL=<url-to-a-windows-flash_attn-wheel>

  flash-attn has no maintained Windows wheel and is painful to build there, so it
  is skipped — Stable Audio 3 falls back to torch SDPA, which is fast and
  numerically fine. Provide SA3_FLASH_WHEEL to install one anyway.

  The SA3 Medium weights are gated on HuggingFace. Accept the license at
    https://huggingface.co/stabilityai/stable-audio-3-medium
  then set a token before --with-model:  $env:HF_TOKEN = "hf_xxx"
#>
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $Rest)

$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here
$Venv = Join-Path $Here '.venv'

$Py    = if ($env:SA3_PYTHON) { $env:SA3_PYTHON } else { '3.10' }
$Cuda  = if ($env:SA3_CUDA)   { $env:SA3_CUDA }   else { 'cu126' }
$Torch = if ($env:SA3_TORCH)  { $env:SA3_TORCH }  else { '2.7.1' }
$Index = "https://download.pytorch.org/whl/$Cuda"
$WithModel = ($Rest -contains '--with-model')

function Log  ($m) { Write-Host "`n[sa3-bootstrap] $m" -ForegroundColor Cyan }
function Warn ($m) { Write-Host "`n[sa3-bootstrap] $m" -ForegroundColor Yellow }

Log "Target: torch $Torch ($Cuda), python $Py  (flash-attn skipped on Windows -> torch SDPA)"

# --- Find a working uv (validate by running it) or fall back to venv + pip ------
$Uv = $null
$cands = @($env:UV, "$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe")
$onPath = (Get-Command uv -ErrorAction SilentlyContinue)
if ($onPath) { $cands += $onPath.Source }
foreach ($c in $cands) {
    if ($c -and (Test-Path $c)) {
        try { & $c --version *> $null; if ($LASTEXITCODE -eq 0) { $Uv = $c; break } } catch {}
    }
}

$PyExe = Join-Path $Venv 'Scripts\python.exe'
if ($Uv) {
    Log "Using uv at $Uv"
    # --no-config: this venv must NOT inherit the host repo's [tool.uv] torch pin.
    & $Uv venv --no-config --python $Py $Venv
    $PipExe = $Uv
    $PipArgs = @('pip', 'install', '--no-config', '--python', $PyExe)
} else {
    Log "uv not found - falling back to the py launcher / python + venv + pip"
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try { & py "-$Py" -m venv $Venv } catch { & py -m venv $Venv }
    } else {
        & python -m venv $Venv
    }
    & $PyExe -m pip install --upgrade pip
    $PipExe = $PyExe
    $PipArgs = @('-m', 'pip', 'install')
}

Log "Installing torch $Torch ($Cuda) ..."
& $PipExe @PipArgs "torch==$Torch" "torchaudio==$Torch" --index-url $Index

Log "Installing stable-audio-3 ..."
& $PipExe @PipArgs "stable-audio-3 @ git+https://github.com/Stability-AI/stable-audio-3"

# --- flash-attn (optional on Windows) -----------------------------------------
if ($env:SA3_FLASH_WHEEL) {
    Log "Installing flash-attn from SA3_FLASH_WHEEL ..."
    try { & $PipExe @PipArgs $env:SA3_FLASH_WHEEL }
    catch { Warn "flash-attn wheel install failed - SA3 will use torch SDPA." }
} else {
    Warn "Skipping flash-attn on Windows (no maintained wheel) - SA3 uses torch SDPA. Set SA3_FLASH_WHEEL=<url> to install one."
}

Log "Verifying imports ..."
$verify = @'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn NOT importable (SDPA fallback):", e)
import stable_audio_3
print("stable_audio_3 OK")
'@
$verify | & $PyExe -

if ($WithModel) {
    Log "Downloading SA3 Medium weights (gated - needs HF token) ..."
    if (-not $env:HF_TOKEN -and $env:HUGGING_FACE_HUB_TOKEN) { $env:HF_TOKEN = $env:HUGGING_FACE_HUB_TOKEN }
    $dl = @'
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN") or None
path = snapshot_download("stabilityai/stable-audio-3-medium", token=tok)
print("Downloaded to", path)
'@
    $dl | & $PyExe -
}

Log "Done. Env at $Venv"
