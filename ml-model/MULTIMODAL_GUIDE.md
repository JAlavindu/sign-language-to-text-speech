# Dual Input System - Camera + Glove Sensors

## 🎯 Two Recognition Approaches

Your system will support **BOTH** input methods:

### Approach 1: Camera-Based (Current ML Pipeline) 📷

- Uses trained CNN model on images
- Works with any camera (webcam, phone, glove-mounted)
- **Real-time video processing** (frame-by-frame)
- No sensors needed - just visual recognition

### Approach 2: Glove Sensor-Based 🧤

- Uses flex sensors, IMU, touch sensors
- Direct measurement of hand configuration
- More accurate, no lighting/occlusion issues
- Works in dark, from any angle

### Approach 3: Hybrid (Best!) 🚀

- Combines both inputs for maximum accuracy
- Camera validates sensor readings
- Sensor data helps resolve visual ambiguities
- Redundancy increases reliability

---

## 📹 Adding Real-Time Camera Recognition

### What's Missing in Current Pipeline?

The current pipeline trains on **static images** but doesn't include:

- ❌ Live webcam/video capture
- ❌ Real-time frame processing
- ❌ Video sequence handling
- ❌ Gesture start/end detection
- ❌ Temporal smoothing (consecutive frame predictions)

### What We Need to Add:

1. **Real-time video capture module**
2. **Hand detection & tracking** (MediaPipe/YOLO)
3. **Frame preprocessing pipeline**
4. **Temporal prediction smoothing**
5. **Gesture segmentation** (start/end detection)
6. **Multi-input fusion** (camera + sensors)

---

## 🛠️ Implementation Roadmap

### PHASE 1: Real-Time Camera Recognition (Video Input)

#### Step 1.1: Add Hand Detection

**What**: Detect and crop hand region from video frames
**Why**: Model trained on cropped hands, not full frames
**Tools**: MediaPipe Hands or YOLOv8-hand

**Implementation Steps**:

1. Install MediaPipe library
2. Create hand detection pipeline
3. Extract bounding box coordinates
4. Crop and resize to 224x224
5. Pass to trained CNN model

**Files to Create**:

- `ml-model/utils/hand_detector.py` - Hand detection class
- `ml-model/7_realtime_camera.py` - Main camera inference script
- `ml-model/utils/video_processor.py` - Video frame processing

**Key Concepts**:

- Process 30 FPS video stream
- Detect hand in each frame
- Handle no-hand-detected cases
- Maintain consistent hand tracking

---

#### Step 1.2: Temporal Smoothing

**What**: Combine predictions across multiple frames
**Why**: Single frame might be ambiguous, sequence is clearer
**Methods**:

- Sliding window averaging (last 10 frames)
- Majority voting (most common prediction)
- Confidence-weighted averaging

**Implementation Steps**:

1. Maintain rolling buffer of last N predictions
2. Apply smoothing algorithm (moving average)
3. Output prediction only when confidence threshold met
4. Implement debouncing (prevent flickering)

**Files to Create**:

- `ml-model/utils/temporal_smoother.py` - Prediction smoothing
- Config parameters: window_size, confidence_threshold

**Key Concepts**:

- Buffer last 10-15 frames
- Output prediction when 60%+ frames agree
- Hold prediction for minimum duration (500ms)

---

#### Step 1.3: Gesture Segmentation

**What**: Detect when a sign starts and ends
**Why**: Separate continuous signing into discrete signs
**Methods**:

- Motion-based (IMU acceleration spikes)
- Visual-based (hand movement detection)
- Pause-based (no motion = sign complete)

**Implementation Steps**:

1. Track hand velocity frame-to-frame
2. Detect motion start (velocity > threshold)
3. Capture sign during stable period
4. Detect motion end (velocity < threshold)
5. Output recognized sign

**Files to Create**:

- `ml-model/utils/gesture_segmenter.py` - Start/end detection
- Parameters: motion_threshold, stable_duration

**Key Concepts**:

- State machine: IDLE → MOTION → STABLE → SIGN → IDLE
- Minimum sign duration: 300ms
- Maximum sign duration: 2000ms

---

#### Step 1.4: Camera Integration Script

**What**: Complete real-time recognition system
**Components**:

- Video capture (OpenCV)
- Hand detection (MediaPipe)
- Model inference (TensorFlow Lite)
- Temporal smoothing
- Display with overlays

**Implementation Steps**:

1. Open webcam stream (cv2.VideoCapture)
2. For each frame:
   - Detect hand region
   - Preprocess to model input
   - Run inference
   - Apply temporal smoothing
   - Display result with bounding box
3. Handle keyboard input (q to quit)
4. Save session log

**Files to Create**:

- `ml-model/7_realtime_camera.py` - Main script
- `ml-model/utils/display_utils.py` - Visualization helpers

**Expected Output**:

```
[Camera Feed]
┌─────────────────────┐
│  [Hand detected]    │  Sign: A (95%)
│   ╔═══════╗         │  FPS: 28
│   ║  👋   ║         │  Latency: 35ms
│   ╚═══════╝         │
└─────────────────────┘
```

---

### PHASE 2: Glove Sensor Integration

#### Step 2.1: Sensor Data Collection System

**What**: Capture flex, IMU, touch sensor data from glove
**Output**: Time-series sensor readings (not images!)

**Implementation Steps**:

1. Stream sensor data via BLE to computer/phone
2. Save as CSV: timestamp, flex1-5, accel_x/y/z, gyro_x/y/z, touch1-5
3. Label each gesture recording with class name
4. Collect 50+ samples per sign per person

**Files to Create**:

- `firmware/sensor_streamer/` - ESP32 BLE streaming code
- `ml-model/8_collect_sensor_data.py` - PC receiver script
- `ml-model/datasets/sensor_data/` - Storage folder

**Data Format**:

```csv
timestamp,flex1,flex2,flex3,flex4,flex5,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,touch1,touch2,touch3,touch4,touch5,label
0.001,512,678,890,456,234,0.1,0.2,9.8,0.01,0.02,0.01,0,1,0,0,0,A
0.011,515,680,892,458,236,0.1,0.2,9.8,0.01,0.02,0.01,0,1,0,0,0,A
```

---

#### Step 2.2: Sensor-Based Model Training

**What**: Train a different model for sensor input (not images!)
**Architecture**: LSTM or 1D-CNN for time-series data

**Implementation Steps**:

1. Load sensor CSV files
2. Create sliding windows (e.g., 50 timesteps = 500ms)
3. Normalize sensor values (0-1 range)
4. Build LSTM/CNN-LSTM model
5. Train with similar callbacks as image model

**Files to Create**:

- `ml-model/9_train_sensor_model.py` - Sensor model training
- `ml-model/models/asl_sensor_model.h5` - Trained sensor model

**Model Architecture**:

```python
Input (50 timesteps × 17 features)
    ↓
Conv1D(64) + ReLU
    ↓
Conv1D(128) + ReLU
    ↓
LSTM(128)
    ↓
Dense(64) + Dropout(0.5)
    ↓
Dense(36) + Softmax
```

**Key Differences from Image Model**:

- Input: Time-series (not 224×224 image)
- Architecture: LSTM/1D-CNN (not MobileNetV2)
- Data: Sensor readings (not pixel values)
- Training: Faster (smaller model)

---

#### Step 2.3: Real-Time Sensor Recognition

**What**: Live inference from glove sensor data

**Implementation Steps**:

1. Receive BLE sensor stream
2. Maintain sliding window buffer (last 50 readings)
3. Preprocess and normalize
4. Run model inference
5. Apply temporal smoothing
6. Output prediction

**Files to Create**:

- `ml-model/10_realtime_sensor.py` - Real-time sensor inference
- `ml-model/utils/ble_receiver.py` - BLE data handler

**Expected Output**:

```
Glove Status: Connected
Sensors: ✓ Flex ✓ IMU ✓ Touch
Sign: A (confidence: 97%)
FPS: 50 | Latency: 20ms
```

---

### PHASE 3: Multi-Modal Fusion (Camera + Sensors)

#### Step 3.1: Fusion Architecture

**What**: Combine predictions from both models
**Methods**:

- **Early Fusion**: Combine raw inputs before model
- **Late Fusion**: Combine model predictions (simpler!)
- **Hybrid Fusion**: Combine at multiple levels

**Recommended: Late Fusion**

```
Camera Frame → Image Model → P_camera (36 classes)
Sensor Data  → Sensor Model → P_sensor (36 classes)
                    ↓
         Weighted Average / Max / Ensemble
                    ↓
              Final Prediction
```

**Implementation Steps**:

1. Run both models in parallel
2. Get confidence scores from each
3. Combine using weighted average:
   ```python
   P_final = 0.6 * P_camera + 0.4 * P_sensor
   ```
4. Choose class with highest combined confidence
5. Apply temporal smoothing on fused predictions

**Files to Create**:

- `ml-model/11_multimodal_fusion.py` - Fusion inference
- `ml-model/utils/fusion_strategies.py` - Different fusion methods

---

#### Step 3.2: Adaptive Weighting

**What**: Dynamically adjust camera vs sensor weights
**Why**: Camera unreliable in dark, sensors fail if glove loose

**Implementation Steps**:

1. Track per-model confidence over time
2. Increase weight for high-confidence model
3. Decrease weight when model uncertain
4. Handle single-input failures gracefully

**Weighting Logic**:

```python
if camera_confidence > 0.9:
    weight_camera = 0.7  # Trust camera more
elif sensor_confidence > 0.9:
    weight_sensor = 0.7  # Trust sensors more
else:
    weight_camera = weight_sensor = 0.5  # Equal
```

---

### PHASE 4: Continuous Signing (Advanced)

#### Step 4.1: Sequence-to-Sequence Model

**What**: Recognize sequences of signs (words/sentences)
**Current**: Isolated signs (one at a time)
**Goal**: Continuous signing (no pauses needed)

**Implementation Steps**:

1. Collect continuous signing videos/sensor data
2. Annotate with sign sequences: "H-E-L-L-O"
3. Train CTC or Attention-based sequence model
4. Output sequences instead of single signs

**Model Architecture**:

```
Video/Sensor Stream
    ↓
Encoder (CNN/LSTM)
    ↓
Decoder (LSTM + Attention)
    ↓
Sign Sequence Output
```

**Files to Create**:

- `ml-model/12_train_sequence_model.py` - Seq2Seq training
- `ml-model/13_realtime_continuous.py` - Continuous recognition

---

#### Step 4.2: Language Model Post-Processing

**What**: Correct recognition errors using language context
**Example**: "THAMK YOU" → "THANK YOU"

**Implementation Steps**:

1. Train n-gram language model on ASL phrases
2. Apply beam search with language model scoring
3. Correct likely mistakes (similar hand shapes)
4. Suggest next likely signs (autocomplete)

**Files to Create**:

- `ml-model/utils/language_model.py` - N-gram LM
- `ml-model/utils/spell_corrector.py` - Error correction

---

## 📋 Complete Implementation Guide

### Part A: Real-Time Camera Recognition

#### A1. Install Additional Dependencies

```powershell
pip install opencv-python
pip install mediapipe
pip install pyttsx3  # Text-to-speech
```

#### A2. Create Hand Detection Module

**File**: `ml-model/utils/hand_detector.py`

**What it does**:

- Takes video frame as input
- Detects hand using MediaPipe
- Returns cropped hand image (224×224)
- Handles no-hand-detected cases

**Key functions**:

- `detect_hand(frame)` → Returns hand bounding box
- `crop_hand(frame, bbox)` → Returns 224×224 image
- `preprocess_for_model(image)` → Normalizes for model

---

#### A3. Create Temporal Smoother

**File**: `ml-model/utils/temporal_smoother.py`

**What it does**:

- Maintains buffer of last N predictions
- Applies majority voting or averaging
- Outputs stable prediction
- Prevents flickering between classes

**Key functions**:

- `add_prediction(class_id, confidence)` → Add to buffer
- `get_smoothed_prediction()` → Returns stable prediction
- `reset()` → Clear buffer

**Parameters**:

- `window_size=10` → Keep last 10 frames
- `min_confidence=0.6` → Minimum to accept
- `hold_duration=500ms` → Minimum display time

---

#### A4. Create Real-Time Camera Script

**File**: `ml-model/7_realtime_camera.py`

**What it does**:

1. Opens webcam
2. Detects hand in each frame
3. Runs model inference
4. Applies temporal smoothing
5. Displays result with overlay
6. Converts to speech (optional)

**Main loop**:

```
WHILE camera is open:
    1. Read frame
    2. Detect hand → crop to 224×224
    3. Preprocess image
    4. Run model inference → get probabilities
    5. Add prediction to smoother
    6. Get stable prediction
    7. Display frame with:
       - Hand bounding box
       - Predicted sign
       - Confidence score
       - FPS counter
    8. If space pressed → speak prediction
    9. If 'q' pressed → quit
```

**Display features**:

- Green box around detected hand
- Large text showing predicted sign
- Confidence percentage
- Real-time FPS
- Instructions overlay

---

#### A5. Create Gesture Segmentation Module

**File**: `ml-model/utils/gesture_segmenter.py`

**What it does**:

- Detects when sign starts (hand enters, movement begins)
- Captures sign during stable period
- Detects when sign ends (hand exits, movement stops)
- Prevents duplicate detections

**State machine**:

```
IDLE (no hand detected)
    ↓ [hand detected]
WAITING (hand present, moving)
    ↓ [motion stops, stable >300ms]
CAPTURING (recording sign)
    ↓ [prediction confidence >90%]
RECOGNIZED (sign confirmed)
    ↓ [wait cooldown 500ms]
IDLE (ready for next sign)
```

**Key functions**:

- `update(hand_detected, motion_detected)` → Update state
- `get_current_state()` → Returns state
- `should_capture()` → True if ready to recognize
- `should_output()` → True if sign complete

---

### Part B: Sensor Data Collection & Training

#### B1. ESP32 Firmware - BLE Streaming

**File**: `firmware/sensor_streamer/sensor_streamer.ino`

**What it does**:

- Reads all sensors (flex, IMU, touch)
- Packages data into BLE characteristic
- Streams at 100 Hz
- Low-latency transmission

**Data packet structure** (60 bytes):

```
[timestamp:4][flex1:2][flex2:2][flex3:2][flex4:2][flex5:2]
[accel_x:4][accel_y:4][accel_z:4]
[gyro_x:4][gyro_y:4][gyro_z:4]
[touch1:1][touch2:1][touch3:1][touch4:1][touch5:1]
```

**BLE Service UUID**: Custom service for sensor data
**Characteristic**: Notify (automatic streaming)

---

#### B2. Sensor Data Collection Tool

**File**: `ml-model/8_collect_sensor_data.py`

**What it does**:

1. Connects to glove via BLE
2. Shows live sensor readings
3. User presses key to start recording
4. Records 2 seconds of data (200 samples)
5. User labels the sign (A-Z, 0-9)
6. Saves to CSV file
7. Repeat for all signs

**UI**:

```
=================================
Sensor Data Collector
=================================
Connected to: ASL_Glove_001
Status: Ready

Flex Sensors: [▓▓░░░] [▓▓▓░░] [▓▓▓▓░] [▓▓░░░] [▓░░░░]
IMU: X:0.1 Y:0.2 Z:9.8 | Gyro: 0.01 0.02 0.01
Touch: ● ○ ○ ○ ○

Press SPACE to start recording
Enter sign label: _
```

**Workflow**:

1. User performs sign
2. Press space → Records 2 seconds
3. Enter label (e.g., "A")
4. Saved to `sensor_data/raw/A_001.csv`
5. Repeat 50 times per sign

---

#### B3. Sensor Data Preprocessing

**File**: `ml-model/utils/sensor_preprocessor.py`

**What it does**:

- Normalizes sensor values (0-1 range)
- Creates sliding windows (50 timesteps)
- Augments data (add noise, time warping)
- Splits train/val/test

**Normalization**:

```python
flex_normalized = (flex_raw - 0) / 4095  # ADC 12-bit
accel_normalized = accel_raw / 16.0      # ±16g range
gyro_normalized = gyro_raw / 2000.0      # ±2000°/s
touch_binary = 1 if touch_raw > 2000 else 0
```

**Sliding window**:

```
Original: [200 timesteps]
Windows:  [0:50], [10:60], [20:70], ..., [150:200]
Result:   15 windows per recording
```

---

#### B4. Sensor Model Training

**File**: `ml-model/9_train_sensor_model.py`

**What it does**:

- Loads sensor CSV files
- Creates windowed dataset
- Builds 1D-CNN + LSTM model
- Trains with same callbacks
- Saves sensor model

**Model architecture**:

```python
Input: (50, 17)  # 50 timesteps, 17 features
    ↓
Conv1D(64, kernel=5) + ReLU + BatchNorm
    ↓
Conv1D(128, kernel=5) + ReLU + BatchNorm
    ↓
MaxPooling1D(2)
    ↓
LSTM(128, return_sequences=False)
    ↓
Dense(128) + ReLU + Dropout(0.5)
    ↓
Dense(64) + ReLU + Dropout(0.3)
    ↓
Dense(36, softmax)
```

**Training configuration**:

- Batch size: 64 (larger than image model)
- Epochs: 30-40 (converges faster)
- Learning rate: 0.001
- Expected accuracy: 92-95% (slightly lower than camera)

---

#### B5. Real-Time Sensor Recognition

**File**: `ml-model/10_realtime_sensor.py`

**What it does**:

1. Connects to glove via BLE
2. Receives sensor stream
3. Maintains 50-sample buffer
4. Runs inference every 100ms
5. Applies temporal smoothing
6. Displays prediction

**Main loop**:

```
WHILE connected:
    1. Receive sensor packet
    2. Add to rolling buffer (size 50)
    3. If buffer full:
       a. Normalize data
       b. Run model inference
       c. Add to smoother
       d. Get stable prediction
       e. Display result
    4. Update UI
```

---

### Part C: Multi-Modal Fusion

#### C1. Fusion Inference System

**File**: `ml-model/11_multimodal_fusion.py`

**What it does**:

- Runs camera model in one thread
- Runs sensor model in another thread
- Synchronizes predictions
- Combines using weighted average
- Outputs final prediction

**Architecture**:

```
┌──────────────┐     ┌──────────────┐
│ Camera Feed  │     │ Sensor Data  │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────┐
│ Image Model  │     │ Sensor Model │
│  P_camera    │     │  P_sensor    │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └────────┬───────────┘
                ▼
         ┌──────────────┐
         │ Fusion Layer │
         │ (Weighted Avg)│
         └──────┬───────┘
                ▼
         Final Prediction
```

**Fusion methods**:

1. **Weighted Average**: `P = w1*P_cam + w2*P_sens`
2. **Maximum Confidence**: Choose model with higher confidence
3. **Voting**: Each model votes, majority wins
4. **Learned Fusion**: Train small MLP to combine

---

#### C2. Adaptive Weighting Strategy

**File**: `ml-model/utils/adaptive_fusion.py`

**What it does**:

- Monitors per-model performance
- Adjusts weights based on confidence
- Handles sensor/camera failures
- Falls back to single input if needed

**Weighting logic**:

```python
def compute_weights(cam_conf, sens_conf, history):
    # Start with equal weights
    w_cam = w_sens = 0.5

    # Boost confident model
    if cam_conf > 0.95:
        w_cam = 0.7
        w_sens = 0.3
    elif sens_conf > 0.95:
        w_cam = 0.3
        w_sens = 0.7

    # Handle failures
    if cam_conf < 0.3:  # Camera unreliable
        w_cam = 0.2
        w_sens = 0.8

    # Consider historical accuracy
    if history.camera_accuracy > history.sensor_accuracy:
        w_cam += 0.1
        w_sens -= 0.1

    return normalize(w_cam, w_sens)
```

---

## 📊 Expected Performance

### Camera-Only Recognition:

- **Accuracy**: 95-97% (good lighting, clear view)
- **FPS**: 25-30 (webcam) or 15-20 (phone)
- **Latency**: 50-100ms per frame
- **Limitations**: Lighting, occlusion, background

### Sensor-Only Recognition:

- **Accuracy**: 92-95% (if glove fitted properly)
- **FPS**: 50-100 (sensor sampling rate)
- **Latency**: 20-30ms
- **Limitations**: Sensor drift, loose fit, calibration

### Multi-Modal Fusion:

- **Accuracy**: 97-99% (best of both!)
- **Latency**: 60-120ms (combined)
- **Robustness**: Works in dark, with occlusion, loose glove
- **Redundancy**: Degrades gracefully if one fails

---

## 🗺️ Implementation Order

### Priority 1: Camera Recognition (Week 1)

1. A2: Hand detection module
2. A3: Temporal smoother
3. A4: Real-time camera script
4. Test with existing trained model

### Priority 2: Sensor Collection (Week 2)

1. B1: ESP32 BLE streaming firmware
2. B2: Sensor data collector
3. Collect data for all 36 signs (50 samples each)

### Priority 3: Sensor Training (Week 3)

1. B3: Sensor preprocessing
2. B4: Train sensor model
3. B5: Real-time sensor recognition
4. Evaluate accuracy

### Priority 4: Fusion (Week 4)

1. C1: Multi-modal fusion system
2. C2: Adaptive weighting
3. Integration testing
4. Performance optimization

### Priority 5: Advanced (Optional)

1. Continuous signing (sequence model)
2. Language model post-processing
3. Mobile app integration
4. ESP32 on-device inference

---

## 📁 New Files to Create

```
ml-model/
├── utils/
│   ├── hand_detector.py           # MediaPipe hand detection
│   ├── temporal_smoother.py       # Prediction smoothing
│   ├── gesture_segmenter.py       # Sign start/end detection
│   ├── sensor_preprocessor.py     # Sensor normalization
│   ├── ble_receiver.py            # BLE communication
│   ├── fusion_strategies.py       # Multi-modal fusion
│   ├── adaptive_fusion.py         # Dynamic weighting
│   └── display_utils.py           # Visualization helpers
│
├── 7_realtime_camera.py           # Camera-based recognition
├── 8_collect_sensor_data.py       # Sensor data collector
├── 9_train_sensor_model.py        # Train on sensor data
├── 10_realtime_sensor.py          # Sensor-based recognition
├── 11_multimodal_fusion.py        # Combined system
├── 12_train_sequence_model.py     # Continuous signing (optional)
└── 13_realtime_continuous.py      # Continuous recognition (optional)

firmware/
└── sensor_streamer/
    └── sensor_streamer.ino        # ESP32 BLE streaming

mobile-app/
├── models/
│   ├── asl_model_quantized.tflite  # Camera model
│   └── asl_sensor_model.tflite     # Sensor model
└── src/
    ├── camera_inference.dart       # Camera processing
    ├── sensor_receiver.dart        # BLE sensor handling
    └── fusion_engine.dart          # Multi-modal fusion
```

---

## ✅ Success Criteria

### Camera System:

- [ ] Detects hand in 95% of frames
- [ ] Recognizes signs with 95%+ accuracy
- [ ] Runs at 20+ FPS
- [ ] Latency <100ms
- [ ] Works in various lighting conditions

### Sensor System:

- [ ] Streams data at 100 Hz
- [ ] Collects 50+ samples per sign
- [ ] Sensor model accuracy >92%
- [ ] Real-time recognition <50ms latency
- [ ] Works regardless of lighting

### Fusion System:

- [ ] Combined accuracy >97%
- [ ] Adaptive weighting works
- [ ] Handles single-input failures
- [ ] Latency <150ms total
- [ ] Smooth, stable predictions

---

## 🎯 Your Next Steps

1. **Read this document thoroughly**
2. **Start with Priority 1** (Camera recognition)
3. **Install new dependencies**: opencv-python, mediapipe
4. **Create the utility modules** (hand_detector, temporal_smoother)
5. **Build the real-time camera script**
6. **Test with your trained image model**
7. **Move to sensor collection** once camera works

**This transforms your system from static image classification to real-time, multi-modal recognition!** 🚀
