import asyncio
import struct
import numpy as np
import tensorflow as tf
import joblib
import os
import sys
from bleak import BleakClient, BleakScanner
from utils.sensor_preprocessor import SensorPreprocessor
from utils.temporal_smoother import TemporalSmoother

# Configuration
DEVICE_NAME = "ASL_Glove_001"
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
CHARACTERISTIC_UUID_TX = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
MODEL_PATH = os.path.join("models", "sensor_model_final.h5")
SCALER_PATH = os.path.join("models", "sensor_scaler.pkl")
LABELS_PATH = os.path.join("models", "sensor_labels.pkl")
WINDOW_SIZE = 50
NUM_FEATURES = 16

class RealTimeSensor:
    def __init__(self):
        self.buffer = []
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.smoother = TemporalSmoother(buffer_size=5)
        self.running = True

    def load_artifacts(self):
        """Load model, scaler, and labels."""
        try:
            print("Loading model and artifacts...")
            self.model = tf.keras.models.load_model(MODEL_PATH)
            
            # Load scaler and wrap in preprocessor
            scaler = joblib.load(SCALER_PATH)
            self.preprocessor = SensorPreprocessor(window_size=WINDOW_SIZE, num_features=NUM_FEATURES)
            self.preprocessor.scaler = scaler
            
            self.label_encoder = joblib.load(LABELS_PATH)
            print("✓ Artifacts loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            print("Did you run 9_train_sensor_model.py?")
            return False

    def parse_packet(self, data):
        """Parse 43-byte packet from ESP32."""
        try:
            if len(data) != 43:
                return None
            # Unpack: I (4) + 5H (10) + 3f (12) + 3f (12) + 5B (5) = 43
            unpacked = struct.unpack('<I5H3f3f5B', data)
            
            # Extract features (excluding timestamp)
            # Flex (5) + Accel (3) + Gyro (3) + Touch (5)
            features = list(unpacked[1:17])
            return features
        except Exception:
            return None

    def notification_handler(self, sender, data):
        """Handle incoming BLE data."""
        features = self.parse_packet(data)
        if features:
            self.buffer.append(features)
            
            # Keep buffer at window size
            if len(self.buffer) > WINDOW_SIZE:
                self.buffer.pop(0)
                
            # Run inference if buffer is full
            if len(self.buffer) == WINDOW_SIZE:
                self.predict()

    def predict(self):
        """Run model inference on current buffer."""
        # Convert buffer to numpy array
        data = np.array(self.buffer)
        
        # Preprocess (Scale)
        # Note: transform expects (N, features), returns (N, features)
        scaled_data = self.preprocessor.transform(data)
        
        # Reshape for model: (1, window_size, features)
        input_data = scaled_data.reshape(1, WINDOW_SIZE, NUM_FEATURES)
        
        # Predict
        prediction = self.model.predict(input_data, verbose=0)
        predicted_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Smooth
        self.smoother.add_prediction(predicted_idx)
        smoothed_idx = self.smoother.get_smoothed_prediction()
        
        if smoothed_idx is not None:
            label = self.label_encoder.inverse_transform([smoothed_idx])[0]
            print(f"\rPrediction: {label} ({confidence*100:.1f}%)", end="")

    async def run(self):
        if not self.load_artifacts():
            return

        print(f"Scanning for {DEVICE_NAME}...")
        device = await BleakScanner.find_device_by_name(DEVICE_NAME)
        
        if not device:
            print(f"Device '{DEVICE_NAME}' not found.")
            return

        print(f"Connecting to {device.address}...")
        async with BleakClient(device) as client:
            print("Connected! Streaming data...")
            
            await client.start_notify(CHARACTERISTIC_UUID_TX, self.notification_handler)
            
            print("Press Ctrl+C to stop")
            try:
                while self.running:
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                await client.stop_notify(CHARACTERISTIC_UUID_TX)

if __name__ == "__main__":
    app = RealTimeSensor()
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\nStopped.")
