#!/bin/bash

echo "========================================"
echo "ASL Model Training - Quick Setup (macOS/Linux)"
echo "========================================"
echo ""

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found. Please install Python 3."
    exit 1
fi

echo "[1/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install -r ml-model/requirements.txt
# Install additional dependencies
pip install mediapipe pyttsx3

echo "[4/4] Checking datasets..."
# Note: This check assumes datasets are in the parent directory, similar to Windows setup
if [ -d "../SignAlphaSet" ]; then
    echo "    [OK] SignAlphaSet found in parent directory"
else
    echo "    [WARNING] SignAlphaSet not found in parent directory."
    echo "    Make sure to update paths in python scripts if your datasets are elsewhere."
fi

if [ -d "../asl_dataset" ]; then
    echo "    [OK] asl_dataset found in parent directory"
else
    echo "    [WARNING] asl_dataset not found in parent directory."
    echo "    Make sure to update paths in python scripts if your datasets are elsewhere."
fi

echo ""
echo "Setup complete!"
echo ""
echo "========================================"
echo "Next steps:"
echo "========================================"
echo "1. Open Terminal in this folder"
echo "2. Activate venv:     source venv/bin/activate"
echo "3. Update paths in:   ml-model/1_data_exploration.py"
echo "                      ml-model/2_prepare_dataset.py"
echo "                      ml-model/3_train_model.py"
echo "4. Run camera:        python ml-model/7_realtime_camera.py"
echo "========================================"
