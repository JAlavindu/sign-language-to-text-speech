# GPU Support Guide for Windows

You have a powerful **NVIDIA GeForce RTX 3090**, but TensorFlow is running on CPU because **TensorFlow 2.11+ dropped native Windows GPU support**.

Since you are using **TensorFlow 2.20.0** on **native Windows 11/10**, it strictly defaults to CPU execution.

## How to Fix It

You have two options to unlock your RTX 3090 for training:

### Option 1: Use WSL2 (Recommended by TensorFlow)
This allows you to use the latest TensorFlow versions (like 2.20) with GPU support.

1.  **Install WSL2**:
    Open PowerShell as Administrator and run:
    ```powershell
    wsl --install
    ```
    (Restart your computer if prompted).

2.  **Open Ubuntu/WSL**:
    Search for "Ubuntu" in your start menu and open it.

3.  **Setup Environment in WSL**:
    Inside the Ubuntu terminal:
    ```bash
    # Update system
    sudo apt update && sudo apt upgrade -y
    
    # Install Python and pip
    sudo apt install python3 python3-pip -y
    
    # Clone your project (Windows drives are mounted at /mnt/c, /mnt/e, etc.)
    cp -r /mnt/e/Lavindu/HCI/sign-language-to-text-speech ~/my-project
    cd ~/my-project
    
    # Create venv and install requirements
    python3 -m venv venv
    source venv/bin/activate
    pip install -r ml-model/requirements.txt
    
    # Run training
    python3 ml-model/3_train_model.py
    ```

### Option 2: Downgrade to TensorFlow 2.10 (Native Windows)
If you prefer to stay in standard Windows PowerShell, you must use the last version of TensorFlow that supported Windows GPU (v2.10). This requires **Python 3.10** (you are currently on 3.13).

1.  **Install Python 3.10**: Download and install it from python.org.
2.  **Create a new virtual environment with Python 3.10**:
    ```powershell
    py -3.10 -m venv venv_gpu
    .\venv_gpu\Scripts\Activate
    ```
3.  **Install compatible TensorFlow**:
    ```powershell
    pip install "tensorflow<2.11"
    pip install -r ml-model/requirements.txt
    ```
