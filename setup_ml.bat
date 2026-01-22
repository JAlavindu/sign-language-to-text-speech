@echo off
echo ========================================
echo ASL Model Training - Quick Setup
echo ========================================
echo.

cd /d "E:\Lavindu\HCI\sign-language-to-text-speech"

echo [1/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Virtual environment not found!
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
)
echo.

echo [2/4] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r ml-model\requirements.txt
echo.

echo [3/4] Checking datasets...
if exist "E:\Lavindu\HCI\SignAlphaSet" (
    echo     [OK] SignAlphaSet found
) else (
    echo     [WARNING] SignAlphaSet not found!
)

if exist "E:\Lavindu\HCI\asl_dataset" (
    echo     [OK] asl_dataset found
) else (
    echo     [WARNING] asl_dataset not found!
)
echo.

echo [4/4] Setup complete!
echo.
echo ========================================
echo Next steps:
echo ========================================
echo 1. Explore data:     python ml-model\1_data_exploration.py
echo 2. Prepare dataset:  python ml-model\2_prepare_dataset.py
echo 3. Train model:      python ml-model\3_train_model.py
echo.
echo Monitor training:    tensorboard --logdir ml-model\logs
echo ========================================
echo.

pause
