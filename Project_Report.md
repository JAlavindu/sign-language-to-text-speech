# Project Report: Sign Language to Text & Speech System

**Course:** Human-Computer Interaction (HCI)
**Project Title:** Sign Language to Text & Speech: A Multi-Modal Approach using Computer Vision and Wearable Sensors
**Date:** January 29, 2026
**Student Name:** [Your Name Here]

---

## 1. Abstract

This project addresses the communication barrier faced by the deaf and hard-of-hearing community by developing a real-time American Sign Language (ASL) translation system. Unlike traditional single-modality systems, this project implements a **Dual-Input System** combining computer vision ("The Eyes") and wearable sensors ("The Feel"). The system utilizes a MobileNetV2-based Convolutional Neural Network (CNN) for visual recognition and a custom glove with flex sensors and IMUs for motion tracking. A user-friendly web interface provides real-time text and speech feedback, aiming to create a seamless interaction bridge between signers and non-signers.

## 2. Introduction

### 2.1 Problem Statement
Sign language is the primary mode of communication for millions of people worldwide. However, the majority of the hearing population does not understand sign language, creating a significant social and professional barrier. Existing translation tools are often expensive, obtrusive, or lack real-time capabilities.

### 2.2 Objectives
The primary objective is to build a robust, cost-effective, and real-time ASL recognition system.
*   **Develop a Vision Model:** Utilize transfer learning to recognize static hand signs (A-Z, 0-9).
*   **Develop a Sensor Glove:** Prototype a wearable device to capture dynamic hand movements.
*   **Implement Multimodal Fusion:** Combine visual and sensor data to improve recognition accuracy.
*   **Create a User Interface:** valid web application to display translations and synthesized speech.

## 3. Literature Review & Background

Sign language recognition (SLR) has been extensively studied using two main approaches:
1.  **Vision-based:** Uses cameras and image processing (e.g., CNNs) to interpret gestures. While non-invasive, these systems suffer from occlusion and lighting issues.
2.  **Sensor-based:** Uses data gloves with flex sensors and accelerometers. These offer high precision for finger positioning but are often expensive and cumbersome.

Recent advancements in deep learning have made lightweight models like **MobileNetV2** suitable for real-time deployment on standard hardware. This project leverages these advancements, proposing a hybrid solution to mitigate the limitations of individual modalities.

## 4. Methodology & System Design

The system follows a modular architecture integrating hardware, machine learning models, and a web-based frontend.

### 4.1 Datasets
The model was trained on a combination of two datasets to ensure robustness:
*   **SignAlphaSet:** ~26,000 images covering 26 classes (A-Z).
*   **asl_dataset:** ~2,500 images covering 36 classes (A-Z, 0-9), providing diverse viewing angles.
*   **Total:** ~28,500 images across 36 classes.

### 4.2 Hardware Architecture
*   **Vision:** Standard Webcam for capturing video frames.
*   **Sensors (Glove):** ESP32 microcontroller, Flex Sensors (finger bending), and IMU (hand orientation).
*   **Compute:** PC/Laptop with GPU support for model inference.

### 4.3 Software Architecture
*   **Backend:** Python with FastAPI. Handles model inference and business logic.
*   **Frontend:** React (TypeScript) with Vite. Provides the user interface for video streaming and result display.
*   **ML Framework:** PyTorch for both Vision and Sensor models.
    *   **Vision Model:** MobileNetV2 (Transfer Learning). Pre-trained on ImageNet and fine-tuned for ASL.
    *   **Sensor Model:** Hybrid CNN-LSTM to process time-series data from the glove.

### 4.4 Multimodal Fusion
The system employs an **Adaptive Fusion** mechanism. Probabilities from the Vision Model and Sensor Model are weighted and combined to produce a final prediction. This allows the system to rely more on sensors when visual data is ambiguous (e.g., poor lighting) and vice versa.

## 5. Implementation

### 5.1 Data Preprocessing
Images are resized, normalized, and augmented (rotations, zoom, brightness) to prevent overfitting. Sensor data is smoothed using a temporal buffer to reduce noise.

### 5.2 Model Training
*   **Vision:** The MobileNetV2 model was trained in two phases:
    1.  **Phase 1:** Frozen base layers, training only the classification head.
    2.  **Phase 2:** Fine-tuning the upper layers of the base model.
*   **Sensors:** The CNN-LSTM model was trained on collected sensor data to recognize dynamic gestures.

### 5.3 Web Interface
 The web application connects to the backend via REST APIs. It streams the webcam feed, overlays detection landmarks (using MediaPipe), and displays the translated text. A Text-to-Speech (TTS) engine reads the recognized words aloud.

## 6. Results & Evaluation

*(Note: Preliminary results based on current training logs)*

*   **Training Accuracy:** The vision model achieves high accuracy on the validation set, demonstrating the effectiveness of transfer learning.
*   **Real-time Performance:** The system operates at a usable frame rate for real-time interaction, with low latency in the FastAPI backend.
*   **Fusion Benefits:** Initial tests suggest that combining sensor data helps distinguish between signs that look visually similar but have different finger configurations.

## 7. Challenges & Limitations

*   **Occlusion:** The camera sometimes loses track of the hand if it moves too fast or is obstructed.
*   **Hardware Calibration:** The glove sensors require calibration for different hand sizes.
*   **Lighting Sensitivity:** Computer vision performance degrades in low-light conditions.

## 8. Conclusion & Future Work

The "Sign Language to Text & Speech" project successfully demonstrates a functional prototype of a multimodal translation system. By combining the strengths of vision and sensor-based approaches, it offers a more robust solution than single-modality alternatives.

**Future Work:**
*   Expand vocabulary to include full words and dynamic sentence-level recognition.
*   Optimize the sensor glove design for better ergonomics.
*   Deploy the model to edge devices (e.g., mobile phones) for greater accessibility.

---
*Note: This report structure is based on standard HCI project requirements and the contents of the provided codebase. Specific metrics should be updated after final training runs.*
