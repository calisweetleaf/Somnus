# Enhanced Model Download Script for AI Agents
[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)]
    [string]$ModelName = "NousResearch/Hermes-3-Llama-3.1-8B",
    
    [Parameter(Mandatory=$false)]
    [ValidateScript({Test-Path (Split-Path $_) -PathType Container})]
    [string]$OutputDir = ".\models",
    
    [Parameter(Mandatory=$false)]
    [ValidatePattern('^[a-zA-Z0-9]{40}$')]
    [string]$HFToken,

    [Parameter(Mandatory=$false)]
    [switch]$Force
)

# Setup logging
$LogFile = Join-Path $PSScriptRoot "model_download.log"
$ErrorActionPreference = "Stop"

function Write-Log {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Message,
        
        [Parameter(Mandatory=$false)]
        [ValidateSet('Info', 'Warning', 'Error')]
        [string]$Level = 'Info'
    )
    $LogMessage = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Level]: $Message"
    Write-Host $LogMessage -ForegroundColor $(
        switch($Level) {
            'Info' { 'White' }
            'Warning' { 'Yellow' }
            'Error' { 'Red' }
        }
    )
    Add-Content -Path $LogFile -Value $LogMessage
}

function Test-PythonEnvironment {
    try {
        $pythonVersion = python --version 2>&1
        Write-Log "Python version: $pythonVersion"
        
        $requiredPackages = @(
            "torch",
            "transformers",
            "huggingface_hub",
            "tqdm",
            "numpy",
            "safetensors"
        )
        foreach ($package in $requiredPackages) {
            python -c "import $package" 2>&1 | Out-Null
            Write-Log "Package verified: $package"
        }
        return $true
    }
    catch {
        Write-Log "Error: Python environment check failed - $($_.Exception.Message)" -Level 'Error'
        return $false
    }
}

function Test-ModelIntegrity {
    param([string]$OutputDir)
    
    $pythonVerify = @"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

try:
    # Verify files exist
    if not os.path.exists('$OutputDir/pytorch_model.bin') and not os.path.exists('$OutputDir/model.safetensors'):
        raise Exception("Model files not found")
    if not os.path.exists('$OutputDir/config.json'):
        raise Exception("Model config not found")
    if not os.path.exists('$OutputDir/tokenizer.json'):
        raise Exception("Tokenizer not found")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained('$OutputDir')
    tokenizer = AutoTokenizer.from_pretrained('$OutputDir')
    
    # Test basic functionality
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=20)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model test output: {decoded}")
    print("Model verification successful")
    exit(0)
except Exception as e:
    print(f"Model verification failed: {str(e)}")
    exit(1)
"@
    
    try {
        $pythonVerify | python
        Write-Log "Model integrity verified" -Level 'Info'
        return $true
    }
    catch {
        Write-Log "Model integrity check failed" -Level 'Error'
        return $false
    }
}

function Start-ModelDownload {
    $pythonScript = @"
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from tqdm.auto import tqdm
import shutil

# Set HuggingFace token if provided
if 'HF_TOKEN' in os.environ:
    login(os.environ['HF_TOKEN'])

model_name = os.environ.get('MODEL_NAME', '$ModelName')
output_dir = os.environ.get('OUTPUT_DIR', '$OutputDir')
temp_dir = output_dir + "_temp"

def progress_callback(current, total):
    if not hasattr(progress_callback, 'pbar'):
        progress_callback.pbar = tqdm(total=total, desc='Downloading', 
                                    unit='B', unit_scale=True, 
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    progress_callback.pbar.update(current - progress_callback.pbar.n)

print(f"Downloading model {model_name} ...")

# Check GPU availability and memory
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    if gpu_memory < 8:
        print("Warning: Low GPU memory detected")

try:
    # Download to temporary directory first
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        progress_callback=progress_callback
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save model and tokenizer to temp directory
    print("\nSaving model and tokenizer...")
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)

    # Move files to final location
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.move(temp_dir, output_dir)
    
    print("Model and tokenizer downloaded successfully.")
except Exception as e:
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    raise e
"@

    try {
        # Set environment variables
        if ($HFToken) {
            $env:HF_TOKEN = $HFToken
            Write-Log "HuggingFace token configured" -Level 'Info'
        }
        $env:MODEL_NAME = $ModelName
        $env:OUTPUT_DIR = $OutputDir

        Write-Log "Starting model download..." -Level 'Info'
        $pythonScript | python
        Write-Log "Download completed successfully" -Level 'Info'
    }
    catch {
        Write-Log "Error during download: $($_.Exception.Message)" -Level 'Error'
        throw
    }
    finally {
        # Cleanup environment variables
        Remove-Item Env:\HF_TOKEN -ErrorAction SilentlyContinue
        Remove-Item Env:\MODEL_NAME -ErrorAction SilentlyContinue
        Remove-Item Env:\OUTPUT_DIR -ErrorAction SilentlyContinue
    }
}

# Main execution
try {
    Write-Log "Initializing download process..." -Level 'Info'
    
    # Verify Python environment
    if (-not (Test-PythonEnvironment)) {
        throw "Python environment verification failed"
    }
    
    # Create output directory if it doesn't exist
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Write-Log "Output directory verified: $OutputDir" -Level 'Info'
    
    Start-ModelDownload
    
    # Verify downloaded model
    if (-not (Test-ModelIntegrity -OutputDir $OutputDir)) {
        throw "Model verification failed"
    }
    
    Write-Log "Process completed successfully" -Level 'Info'
}
catch {
    Write-Log "Fatal error: $($_.Exception.Message)" -Level 'Error'
    exit 1
}
finally {
    Write-Log "Script execution completed" -Level 'Info'
}