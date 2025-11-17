@echo off
echo ========================================
echo ASL Model Training - Quick Setup
echo ========================================
echo.

cd /d "e:\UNI sub\ICT\3rd yr\HCI\sign-language-glove"

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
pip install --upgrade pip
pip install -r ml-model\requirements.txt
echo.

echo [3/4] Checking datasets...
if exist "e:\UNI sub\ICT\3rd yr\HCI\SignAlphaSet" (
    echo     [OK] SignAlphaSet found
) else (
    echo     [WARNING] SignAlphaSet not found!
)

if exist "e:\UNI sub\ICT\3rd yr\HCI\asl_dataset" (
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
