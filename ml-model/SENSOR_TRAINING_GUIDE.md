# Sensor Model Training Guide 🧤

This guide covers how to train the machine learning model for your glove sensors. Unlike the camera model (which uses images), this model uses **time-series data** from flex sensors, accelerometer, gyroscope, and touch sensors.

## 📚 Prerequisites

Ensure you have collected data using `8_collect_sensor_data.py`.
You should have CSV files in `ml-model/datasets/sensor_data/raw/` with columns:
`timestamp, flex1..5, accel_x..z, gyro_x..z, touch1..5, label`

---

## 🧩 Component 1: Sensor Preprocessor

**File:** `ml-model/utils/sensor_preprocessor.py`

**Goal:** Prepare raw sensor data for the model. This class must be used during **both** training and real-time inference to ensure consistency.

### 🛠️ What to Implement:

1. **`__init__(self, window_size=50, num_features=16)`**:

   - `window_size`: Number of time steps to feed into the model (e.g., 50 samples ≈ 1 second at 50Hz).
   - Initialize `StandardScaler` (from sklearn) or manual min/max values.

2. **`fit(self, data)`**:

   - Compute mean and std dev for normalization from the training data.
   - Save these statistics (e.g., using `joblib` or `pickle`).

3. **`transform(self, data)`**:

   - Normalize the data: `(x - mean) / std`.
   - **Important**: Flex sensors might need MinMax scaling (0-4095 -> 0-1), while IMU data works best with Standard scaling.

4. **`create_sequences(self, data, labels)`**:
   - Convert continuous data into sliding windows.
   - Input: `(N, features)`
   - Output: `(N - window_size, window_size, features)`
   - Example: If you have 100 samples and window_size=10, you get 90 sequences of length 10.

---

## 🧩 Component 2: Training Script

**File:** `ml-model/9_train_sensor_model.py`

**Goal:** Load data, train the LSTM/CNN model, and save artifacts.

### 📖 Where to Look Up:

- [Keras LSTM Layer](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Sklearn LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

### 🛠️ Workflow Logic:

1. **Load Data**:

   - Read all CSV files from `datasets/sensor_data/raw/`.
   - Combine them into a single DataFrame.
   - Separate features (columns 1-16) and target (label).

2. **Preprocess**:

   - Initialize `SensorPreprocessor`.
   - `fit` on the data.
   - `transform` the data.
   - Encode labels (A -> 0, B -> 1) using `LabelEncoder`.
   - Create sequences using sliding window.

3. **Build Model**:

   - **Architecture Recommendation**:
     ```python
     model = Sequential([
         # Input shape: (window_size, num_features)
         Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, num_features)),
         MaxPooling1D(pool_size=2),
         LSTM(64, return_sequences=False),
         Dropout(0.2),
         Dense(32, activation='relu'),
         Dense(num_classes, activation='softmax')
     ])
     ```
   - Compile with `adam` optimizer and `sparse_categorical_crossentropy`.

4. **Train**:

   - `model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)`
   - Use `ModelCheckpoint` to save the best model.

5. **Save Artifacts**:
   - Save model: `model.save('models/sensor_model.h5')`
   - Save preprocessor/scaler: `joblib.dump(scaler, 'models/sensor_scaler.pkl')`
   - Save label encoder: `joblib.dump(encoder, 'models/sensor_labels.pkl')`

---

## 💡 Tips for Success

- **Data Balance**: Ensure you have roughly the same number of recordings for each sign.
- **Window Size**:
  - Too small (<20): Model can't see the "movement".
  - Too large (>100): Latency increases, model gets confused by pauses.
  - **50** is a good starting point (approx 1 second of data).
- **Features**:
  - You have 17 columns. `timestamp` is not a feature.
  - `touch` sensors are binary (0 or 1) or analog? If binary, they don't need scaling.
  - `flex` sensors are 0-4095. Divide by 4095.0 to normalize.
  - `accel/gyro` can be negative. Standard scaling is best.

Good luck! 🚀
