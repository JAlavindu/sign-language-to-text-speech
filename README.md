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

## ✨ Key Features

- **Real-Time Recognition**: Instant translation of ASL signs via Webcam.
- **Smart Smoothing**: Temporal smoothing algorithms to prevent jittery predictions.
- **Text-to-Speech (TTS)**: Speaks the recognized sentence out loud.
- **Sentence Building**: Automatically constructs sentences when signs are held stable.
- **Wireless Glove**: ESP32-based wearable streaming sensor data via Bluetooth Low Energy (BLE).
- **Custom Training Pipeline**: Complete scripts to train your own models on custom datasets.

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
│   ├── 1_data_exploration.py
│   ├── 2_prepare_dataset.py
│   ├── 3_train_model.py
│   ├── 7_realtime_camera.py      # 📷 Camera-Only Demo
│   ├── 8_collect_sensor_data.py  # 🧤 Sensor Data Collector
│   ├── 9_train_sensor_model.py   # 🧠 Train Sensor Model
│   ├── 10_realtime_sensor.py     # 🧤 Sensor-Only Demo
│   ├── 11_multimodal_fusion.py   # 🚀 Hybrid System Demo
│   ├── 12_final_app.py           # 🏆 CLI Product (TTS + Sentence Builder)
│   ├── 13_generate_report_graphs.py # 📊 Generate Report Artifacts
│   ├── 14_gui_app.py             # 🖥️ Professional GUI Product
│   └── requirements.txt
├── setup_ml.bat            # Windows Setup Script
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- **Software**: Python 3.10 or 3.11 (Recommended for MediaPipe compatibility).
- **Hardware**: Webcam (for camera mode) OR ESP32 + Flex Sensors (for glove mode).

### Installation

1.  **Clone the repository**

    ```bash
    git clone https://github.com/JAlavindu/sign-language-to-text-speech.git
    cd sign-language-to-text-speech
    ```

2.  **Set up the environment**

    ```powershell
    .\setup_ml.bat
    ```

3.  **Install additional dependencies** (for Camera & Sensors)
    ```powershell
    .\venv\Scripts\Activate.ps1
    pip install opencv-python mediapipe pyttsx3 bleak
    ```

## 🎮 Usage Guide

### Phase 1: The "Eyes" (Camera System)

1.  **Train the Model** (If you haven't yet):
    ```powershell
    python ml-model/3_train_model.py
    ```
2.  **Run Real-Time Recognition**:
    ```powershell
    python ml-model/7_realtime_camera.py
    ```
    - **Spacebar**: Speak the current sentence.
    - **Backspace**: Clear the sentence.
    - **Q**: Quit.

### Phase 2: The "Feel" (Glove System)

1.  **Upload Firmware**: Flash `firmware/sensor_streamer/sensor_streamer.ino` to your ESP32.
2.  **Collect Training Data**:
    ```powershell
    python ml-model/8_collect_sensor_data.py
    ```
    - Follow the prompts to record sensor data for each sign.

## 🗺️ Roadmap

- [x] **Hardware Design**: Parts list and wiring diagrams complete.
- [x] **ML Pipeline**: Data exploration, processing, and training scripts ready.
- [x] **Camera System**: Real-time detection, temporal smoothing, and TTS integration.
- [x] **Glove Firmware**: BLE streaming implemented.
- [x] **Sensor Collection**: Python script to record glove data.
- [ ] **Sensor Model**: Train LSTM/CNN model on sensor data.
- [ ] **Fusion**: Combine Camera + Glove predictions for maximum accuracy.
- [ ] **Mobile App**: Port inference to a mobile application.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source and available under the MIT License.
