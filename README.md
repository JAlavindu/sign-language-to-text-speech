# 🧤 Sign Language to Text & Speech System

> **A Multi-Modal Approach to ASL Translation using Computer Vision and Wearable Sensors**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![ESP32](https://img.shields.io/badge/Hardware-ESP32-red)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

## 📖 About The Project

This project aims to bridge the communication gap for the deaf and hard-of-hearing community by translating American Sign Language (ASL) into text and spoken speech in real-time.

Unlike traditional systems that rely solely on cameras _or_ gloves, this project implements a **Dual-Input System**:

1.  **The Eyes (Camera)**: Uses Computer Vision (MediaPipe + CNN) to recognize hand shapes visually.
2.  **The Feel (Glove)**: Uses Flex Sensors and IMUs to capture finger bending and hand motion.
3.  **The Voice (TTS)**: Converts recognized gestures into spoken words using Text-to-Speech.

## 🛠️ Prerequisites

### Hardware

- **PC/Laptop**: Windows recommended for full compatibility.
- **Webcam**: Required for the vision-based recognition.
- **ESP32-S3 Board**: (Optional) For the smart glove component.
- **Flex Sensors & IMU**: (Optional) For the smart glove component.

### Software

- **Python 3.10 or 3.11**: Required for MediaPipe compatibility.
- **Arduino IDE 2.x**: Required if you are building the glove firmware.

## 🚀 Installation & Setup

Follow these steps to set up the project from scratch.

### 1. Clone the Repository

```bash
git clone https://github.com/JAlavindu/sign-language-to-text-speech.git
cd sign-language-to-text-speech
```

### 2. Automatic Environment Setup (Windows)

We have provided a script to automatically create a virtual environment and install necessary dependencies.

```powershell
.\setup_ml.bat
```

### 3. Manual Setup (Alternative)

If the batch file doesn't work, you can set it up manually:

```powershell
# Create virtual environment
python -m venv venv

# Activate environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r ml-model/requirements.txt
```

---

## 🎮 How to Run (Step-by-Step)

This project is divided into phases. You can start with just the camera (Phase 1) and later add the glove (Phase 2).

### 📸 Phase 1: The "Eyes" (Camera-Only Mode)

This mode uses your webcam to recognize hand signs. It requires no special hardware.

#### Step 1: Explore & Prepare Data

First, we need to process the raw image datasets into a format suitable for training.

```powershell
# Navigate to the ML folder
cd ml-model

# (Optional) Explore the dataset statistics
python 1_data_exploration.py

# Prepare and split the dataset (Train/Val/Test)
python 2_prepare_dataset.py
```

_What this does:_ Organizes your images from `asl_dataset` and `SignAlphaSet` into a unified structure in `datasets/processed`.

#### Step 2: Train the Vision Model

Now, train the Convolutional Neural Network (CNN) to recognize signs.

```powershell
python 3_train_model.py
```

_What this does:_ Uses Transfer Learning (MobileNetV2) to train a high-accuracy model.
_Time:_ 2-4 hours on GPU, or overnight on CPU.

#### Step 3: Run Real-Time Camera Demo

Once trained, you can test the camera recognition immediately.

```powershell
python 7_realtime_camera.py
```

**Controls:**

- **Spacebar**: Speak the current sentence.
- **Backspace**: Clear the sentence.
- **Q**: Quit.

---

### 🧤 Phase 2: The "Feel" (Smart Glove Mode)

This mode adds the wearable sensor glove for higher accuracy and occlusion handling.

#### Step 1: Firmware Setup

1. Open `firmware/sensor_streamer/sensor_streamer.ino` in Arduino IDE.
2. Select your board (ESP32-S3 Dev Module).
3. Install required libraries (Adafruit MPU6050, ArduinoJson).
4. Upload the code to your ESP32.

#### Step 2: Collect Sensor Data

You need to record sensor data for each sign to train the glove model.

```powershell
python 8_collect_sensor_data.py
```

_Instructions:_ Follow the on-screen prompts to perform each sign while the script records sensor values.

#### Step 3: Train Sensor Model

Train a separate model specifically for the glove data.

```powershell
python 9_train_sensor_model.py
```

#### Step 4: Run Sensor Demo

Test the glove-only recognition.

```powershell
python 10_realtime_sensor.py
```

---

### 🚀 Phase 3: The Complete System (Multimodal)

Combine both Camera and Glove for the best performance. This uses "Adaptive Fusion" to trust the most reliable sensor at any moment.

#### Run the Final Application

You have two options for the final product:

**Option A: Professional GUI App (Recommended)**
A modern, dark-themed interface with live video feed and text display.

```powershell
python 14_gui_app.py
```

**Option B: Command Line Interface (CLI)**
A lightweight version that runs in the terminal.

```powershell
python 12_final_app.py
```

## 📂 Project Structure

```
sign-language-glove/
├── docs/                   # Documentation & Guides
├── firmware/               # ESP32 Microcontroller Code
│   └── sensor_streamer/    # BLE Sensor Streaming Firmware
├── hardware/               # Wiring Diagrams & Parts Lists
├── ml-model/               # Machine Learning Core
│   ├── datasets/           # Raw & Processed Data
│   ├── models/             # Trained .h5 Models
│   ├── reports/            # Generated Graphs & Confusion Matrices
│   ├── utils/              # Helper Modules (HandDetector, Smoother, Fusion, etc.)
│   ├── 1_data_exploration.py     # 📊 Analyze datasets
│   ├── 2_prepare_dataset.py      # ⚙️ Process images
│   ├── 3_train_model.py          # 🧠 Train Vision Model
│   ├── 7_realtime_camera.py      # 📷 Camera-Only Demo
│   ├── 8_collect_sensor_data.py  # 🧤 Sensor Data Collector
│   ├── 9_train_sensor_model.py   # 🧠 Train Sensor Model
│   ├── 10_realtime_sensor.py     # 🧤 Sensor-Only Demo
│   ├── 11_multimodal_fusion.py   # 🚀 Hybrid System Demo
│   ├── 12_final_app.py           # 🏆 CLI Product
│   ├── 13_generate_report_graphs.py # 📈 Generate Reports
│   ├── 14_gui_app.py             # 🖥️ GUI Product
│   └── requirements.txt
├── setup_ml.bat            # Windows Setup Script
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source and available under the MIT License.
