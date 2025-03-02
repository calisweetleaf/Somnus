# Agent Setup Guide

## 1. Environment Setup

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Install AutoHotkey v2.0
1. Download from: https://www.autohotkey.com/
2. Run installer
3. Verify installation: `AutoHotkey.exe --version`

## 2. Model Setup
1. First run will automatically download Hermes 3 model
2. Ensure ~15GB free disk space for model files
3. Recommended: GPU with 8GB+ VRAM

## 3. Directory Structure
```
/e:/Erebus_DevDrive/scripts/Agent Scripts/
├── orchestrator.py
├── core_sys_security.py
├── download_file.py
├── launch_app.ps1
└── new_folder.ps1
```

## 4. Running the Agent

1. Open PowerShell as Administrator
2. Navigate to scripts directory
3. Run:
```bash
python orchestrator.py
```

## 5. Usage Examples

Basic commands to test:
```
>> You: Create a new folder called "test" on the desktop
>> You: Open notepad and type "Hello World"
>> You: Calculate 2 + 2 * 4
>> You: Launch calculator
```

## 6. Troubleshooting

1. If model fails to load:
   - Check available VRAM
   - Try setting `device_map="cpu"` in orchestrator.py

2. If AutoHotkey scripts fail:
   - Verify AutoHotkey v2.0 installation
   - Check script paths in orchestrator.py

3. If PowerShell scripts fail:
   - Run PowerShell as Administrator
   - Check execution policy: `Get-ExecutionPolicy`
   - If restricted, run: `Set-ExecutionPolicy RemoteSigned`
