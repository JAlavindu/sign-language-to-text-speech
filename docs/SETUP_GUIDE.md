# Development Environment Setup Guide

## Prerequisites

- Windows PC (you're already set!)
- USB-C cable for ESP32-S3
- Internet connection for downloads

---

## Step 1: Install Arduino IDE 2.x

### Download & Install

1. Go to: https://www.arduino.cc/en/software
2. Download **Arduino IDE 2.3.2** (or latest 2.x version) for Windows
3. Run installer → Install for current user
4. Launch Arduino IDE

---

## Step 2: Add ESP32-S3 Board Support

### Install ESP32 Board Package

1. In Arduino IDE, go to **File → Preferences**
2. In "Additional Board Manager URLs", add:
   ```
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```
3. Click **OK**

4. Go to **Tools → Board → Boards Manager**
5. Search for: `esp32`
6. Install **"esp32 by Espressif Systems"** (version 3.0.0 or latest)
7. Wait for installation to complete

### Select Your Board

1. Connect ESP32-S3 via USB-C
2. Go to **Tools → Board → esp32**
3. Select: **"ESP32-S3 Dev Module"**
4. Configure board settings:
   - **USB CDC On Boot:** Enabled
   - **CPU Frequency:** 240MHz
   - **Flash Size:** 4MB (or match your module)
   - **Partition Scheme:** Default
   - **PSRAM:** Disabled (or OPI if you have 8MB PSRAM)
   - **Upload Mode:** UART0 / Hardware CDC
   - **Port:** Select COM port (e.g., COM3)

---

## Step 3: Install Required Libraries

Go to **Tools → Manage Libraries** (or Ctrl+Shift+I), search and install:

### Core Sensor Libraries

1. **Adafruit MPU6050** (by Adafruit)

   - Also installs: Adafruit BusIO, Adafruit Unified Sensor
   - For IMU (accelerometer/gyroscope)

2. **Wire** (Built-in)
   - I2C communication library (pre-installed with ESP32)

### BLE (Bluetooth Low Energy)

3. **ESP32 BLE Arduino** (Built-in with ESP32 package)
   - Already included in ESP32 board package
   - No separate installation needed

### Data Processing

4. **ArduinoJson** (by Benoit Blanchon)
   - Version 7.x or latest
   - For packaging sensor data

### Optional (Install Later)

5. **TensorFlowLite_ESP32** (for on-device ML inference)
   - We'll add this in later steps

---

## Step 4: Test Installation

### Upload Blink Test

1. Go to **File → Examples → 01.Basics → Blink**
2. Change LED pin to `38` (ESP32-S3 onboard LED):
   ```cpp
   #define LED_BUILTIN 38
   ```
3. Click **Upload** (→ icon)
4. Watch for "Done uploading" message
5. **Verify:** Onboard LED should blink every second

### Troubleshooting Upload Issues

If upload fails:

- **"Serial port not found"**: Install CH340/CP2102 USB drivers
- **"Failed to connect"**: Hold BOOT button during upload
- **"Timeout"**: Try different USB cable or port
- **"Permission denied"**: Close Serial Monitor first

---

## Step 5: Install CP2102 USB Driver (If Needed)

If your ESP32-S3 doesn't show up as a COM port:

1. Download from: https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers
2. Install "CP210x Windows Drivers"
3. Restart computer
4. Reconnect ESP32-S3
5. Check **Device Manager → Ports (COM & LPT)** for new COM port

---

## Step 6: Set Up Python Environment (For ML Training)

### Install Python 3.10+

1. Download from: https://www.python.org/downloads/
2. Run installer → **Check "Add Python to PATH"**
3. Install for all users
4. Verify in PowerShell:
   ```powershell
   python --version
   ```

### Create Virtual Environment

```powershell
cd "e:\UNI sub\ICT\3rd yr\HCI\sign-language-glove"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install ML Libraries

```powershell
pip install --upgrade pip
pip install numpy pandas matplotlib
pip install scikit-learn
pip install tensorflow
pip install jupyter notebook
```

### Test TensorFlow Installation

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## Step 7: Install Visual Studio Code (Optional but Recommended)

Better than Arduino IDE for complex projects:

1. Download from: https://code.visualstudio.com/
2. Install extensions:
   - **PlatformIO IDE** (for ESP32 development)
   - **Python** (by Microsoft)
   - **Jupyter** (for ML notebooks)
3. Open project folder in VS Code

### PlatformIO Setup (Alternative to Arduino IDE)

1. Install PlatformIO extension
2. Create new project:
   - Board: ESP32-S3-DevKitC-1
   - Framework: Arduino
3. Dependencies auto-managed via `platformio.ini`

---

## Step 8: Set Up Git (Optional - For Version Control)

```powershell
# Install Git for Windows
# Download from: https://git-scm.com/download/win

# Initialize repository
cd "e:\UNI sub\ICT\3rd yr\HCI\sign-language-glove"
git init
git add .
git commit -m "Initial commit: Project setup"
```

---

## Directory Structure Created

Your workspace should now have:

```
sign-language-glove/
├── firmware/                  # ESP32 Arduino sketches
│   ├── test_sensors/         # Individual sensor tests
│   └── data_collector/       # Full data collection firmware
├── ml-model/                 # Python ML training scripts
│   ├── notebooks/            # Jupyter notebooks
│   ├── datasets/             # Collected sensor data
│   └── models/               # Trained models
├── mobile-app/               # Smartphone app (Flutter/React Native)
├── hardware/                 # Wiring diagrams, schematics
├── docs/                     # Documentation
└── venv/                     # Python virtual environment
```

---

## Quick Reference Commands

### Arduino IDE

- **Verify code:** Ctrl+R
- **Upload:** Ctrl+U
- **Open Serial Monitor:** Ctrl+Shift+M
- **Serial Plotter:** Tools → Serial Plotter

### Python Virtual Environment

```powershell
# Activate
.\venv\Scripts\Activate.ps1

# Deactivate
deactivate

# Install package
pip install package-name

# List installed
pip list
```

---

## Verification Checklist

Before moving to coding:

- [ ] Arduino IDE 2.x installed
- [ ] ESP32 board package installed
- [ ] ESP32-S3 detected as COM port
- [ ] Blink example uploads successfully
- [ ] All required libraries installed
- [ ] Python 3.10+ with TensorFlow working
- [ ] Virtual environment created
- [ ] Project folders created

---

## Next Steps

Now that your environment is ready, we'll:

1. Write test code for each sensor
2. Verify all hardware connections
3. Build the data collection firmware
4. Create the smartphone app

**Ready to write your first sensor test code?** (Step 5)
