# ASL Sign Language Recognition - Setup Guide

This guide explains how to set up and run the ASL Recognition system on a new computer without re-training the model.

## Prerequisites

- **Python 3.10 or 3.11** (Ensure it is added to your system PATH)
- **Webcam**
- **Git** (to clone the repository)

## 1. Clone the Repository

Open a terminal (Command Prompt or PowerShell) and run:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd sign-language-to-text-speech
```

## 2. Set Up Virtual Environment

Create a clean virtual environment to isolate dependencies:

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

Install the required libraries. Note that PyTorch installation depends on whether you have a GPU (NVIDIA) or not.

### Option A: Standard Install (CPU - Works on any computer)
This is the safest option for running on generic laptops.

```bash
pip install -r ml-model/requirements_inference.txt
```

### Option B: GPU Install (NVIDIA GPU required)
If you have an NVIDIA GPU, install PyTorch with CUDA support first:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r ml-model/requirements_inference.txt
```

## 4. Run the Application

The project contains a trained PyTorch model (`.pth`) for vision-based recognition.

To run the real-time camera recognition system:

```bash
python ml-model/7_realtime_camera.py
```

## Troubleshooting

- **"Module not found"**: Ensure you activated the virtual environment (`venv`) before running the script.
- **Webcam validation**: If the window opens and closes immediately, check if another app is using the camera.
- **Slow performance**: Ensure you are running on a machine with a decent CPU or GPU.
