# Real-Time Camera Implementation Guide 📷

This guide will help you build the real-time camera recognition system step-by-step. You will write the code for the core components.

## 📚 Prerequisites

Before starting, make sure you have the necessary libraries:

```bash
pip install opencv-python mediapipe tensorflow numpy
```

---

## 🧩 Component 1: Hand Detector Wrapper

**File:** `ml-model/utils/hand_detector.py`

**Goal:** Create a class that handles MediaPipe complexity and returns clean bounding boxes and cropped images.

### 📖 Where to Look Up:

- [MediaPipe Hands Python Solution](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python)
- [OpenCV Image Processing](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### 🛠️ What to Implement:

1. **`__init__(self)`**:

   - Initialize `mp.solutions.hands.Hands`.
   - Set parameters: `static_image_mode=False`, `max_num_hands=1`, `min_detection_confidence=0.7`.

2. **`process(self, frame)`**:

   - Convert frame from BGR (OpenCV default) to RGB.
   - Pass RGB frame to `hands.process()`.
   - If results found:
     - Extract landmarks.
     - Calculate bounding box (min/max x and y of landmarks).
     - Add padding to the bounding box (so the hand isn't tight against the edge).
     - **Crucial**: Ensure bounding box doesn't go outside image dimensions (clamp values).
   - Return: `landmarks`, `bbox` (x, y, w, h).

3. **`crop_hand(self, frame, bbox)`**:
   - Use numpy slicing to crop the image: `hand_img = frame[y:y+h, x:x+w]`.
   - Resize `hand_img` to `(224, 224)` (Model input size).
   - Return processed image.

---

## 🧩 Component 2: Temporal Smoother

**File:** `ml-model/utils/temporal_smoother.py`

**Goal:** Prevent "flickering" predictions (e.g., A -> A -> B -> A) by averaging results over time.

### 📖 Where to Look Up:

- [Python collections.deque](https://docs.python.org/3/library/collections.html#collections.deque)
- [Numpy bincount (for voting)](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html)

### 🛠️ What to Implement:

1. **`__init__(self, buffer_size=10)`**:

   - Create a buffer (list or deque) to store the last `buffer_size` predictions.

2. **`add_prediction(self, prediction_index)`**:

   - Add the new prediction to the buffer.
   - If buffer is full, remove the oldest one.

3. **`get_smoothed_prediction(self)`**:
   - **Method A (Voting)**: Count occurrences of each class in the buffer. Return the most frequent one.
   - **Method B (Average)**: If you store probabilities, average them and return the argmax.
   - _Recommendation_: Start with Voting (Method A) as it's simpler and robust.

---

## 🧩 Component 3: Main Application

**File:** `ml-model/7_realtime_camera.py`

**Goal:** The main loop that runs the camera, uses the tools, and displays results.

### 📖 Where to Look Up:

- [OpenCV VideoCapture](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)
- [TensorFlow Load Model](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model)

### 🛠️ Workflow Logic:

1. **Setup**:

   - Load your trained model: `model = tf.keras.models.load_model('...')`.
   - Load your class labels (A-Z, 0-9).
   - Initialize `HandDetector`, `TemporalSmoother`, and `GestureSegmenter`.
   - Start Webcam: `cap = cv2.VideoCapture(0)`.

2. **Main Loop (`while True`)**:

   - Read frame: `ret, frame = cap.read()`.
   - **Detect**: Call `detector.process(frame)`.
   - **If Hand Detected**:
     - **Segment**: Call `segmenter.process(landmarks)`.
     - **Check State**:
       - If `segmenter.state == "STABLE"` (or just always for testing):
         - **Crop**: Get hand image.
         - **Preprocess**: Normalize (0-1) and reshape to `(1, 224, 224, 3)`.
         - **Predict**: `preds = model.predict(img)`.
         - **Smooth**: Add result to smoother.
         - **Result**: Get final text.
     - **Draw**: Draw bounding box and text on frame using `cv2.rectangle` and `cv2.putText`.
   - **Display**: `cv2.imshow('Sign Language', frame)`.
   - **Exit**: Check for 'q' key press.

3. **Cleanup**:
   - `cap.release()`
   - `cv2.destroyAllWindows()`

---

## 💡 Tips for Success

- **Lighting**: Ensure your hand is well-lit.
- **Background**: A plain background helps the model focus.
- **Model Input**: Remember your model expects `(1, 224, 224, 3)` input range `[0, 1]` or `[-1, 1]` depending on how you trained it (MobileNet usually likes `[-1, 1]`). Check your `2_prepare_dataset.py` to see if you rescaled by `1./255`.
- **Debug Mode**: Print the raw prediction probabilities to the console to see how confident the model is.

Good luck! You have the logic, now write the code! 🚀
