import asyncio
import struct
import numpy as np
import torch
import torch.nn as nn
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
MODEL_PATH = os.path.join("models", "sensor_model_best.pth")
PREPROCESSOR_PATH = os.path.join("models", "sensor_preprocessor.pkl")
LABELS_PATH = os.path.join("models", "sensor_labels.pkl")
WINDOW_SIZE = 50
NUM_FEATURES = 16

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SensorModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SensorModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 64)
        self.fc_dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.bn(x_cnn)
        x_cnn = self.pool(x_cnn)
        x_lstm = x_cnn.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_lstm)
        last_out = lstm_out[:, -1, :] 
        out = self.dropout1(last_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc_dropout(out)
        out = self.fc2(out)
        return out

class RealTimeSensor:
    def __init__(self):
        self.buffer = []
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.smoother = TemporalSmoother(window_size=5) # Renamed to window_size to match other scripts
        self.running = True

    def load_artifacts(self):
        """Load model, scaler, and labels."""
        try:
            print("Loading model and artifacts...")
            
            # Load labels first to determine num_classes
            self.label_encoder = joblib.load(LABELS_PATH)
            num_classes = len(self.label_encoder.classes_)
            
            # Load model
            self.model = SensorModel(num_features=NUM_FEATURES, num_classes=num_classes)
            if os.path.exists(MODEL_PATH):
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                self.model.to(device)
                self.model.eval()
            else:
                 print(f"Model file not found at {MODEL_PATH}")
                 return False

            # Load preprocessor
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
            
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
        except Exception as e:
            print(f"Parse error: {e}")
            return None

    def notification_handler(self, sender, data):
        """Handle incoming BLE notifications."""
        features = self.parse_packet(data)
        if features:
            self.buffer.append(features)
            
            # Keep buffer size in check
            if len(self.buffer) > WINDOW_SIZE:
                 self.buffer.pop(0)
            
            # Run inference if buffer is full
            if len(self.buffer) == WINDOW_SIZE:
                self.run_inference()

    def run_inference(self):
        """Preprocess buffer and run inference."""
        if not self.model or not self.preprocessor:
            return

        try:
            # 1. Preprocess
            # Need to transform the buffer using `transform` logic from preprocessor
            # We assume Buffer is (50, 16)
            data_array = np.array(self.buffer)
            
            # Important: Preprocessor expects shape (N, 16). 
            scaled_data = self.preprocessor.transform(data_array)
            
            # Expand dims to (1, 50, 16) for batch
            input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 2. Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                idx = predicted.item()
                conf = confidence.item()
            
            # 3. Label
            label = self.label_encoder.inverse_transform([idx])[0]
            
            # 4. Smooth
            self.smoother.add_prediction(idx) # Using index for smoothing
            smoothed_idx = self.smoother.get_smoothed_prediction()
            smoothed_label = self.label_encoder.inverse_transform([smoothed_idx])[0]
            
            print(f"Pred: {label} ({conf:.2f}) -> Smooth: {smoothed_label}")
            
        except Exception as e:
            print(f"Inference error: {e}")

async def run_ble_client():
    sensor = RealTimeSensor()
    if not sensor.load_artifacts():
        return

    print(f"Scanning for {DEVICE_NAME}...")
    device_ble = await BleakScanner.find_device_by_name(DEVICE_NAME)
    
    if not device_ble:
        print(f"Device {DEVICE_NAME} not found.")
        return

    print(f"Connecting to {device_ble.address}...")
    async with BleakClient(device_ble.address) as client:
        print("Connected!")
        
        await client.start_notify(CHARACTERISTIC_UUID_TX, sensor.notification_handler)
        print("Listening for data... (Ctrl+C to stop)")
        
        while sensor.running:
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(run_ble_client())
    except KeyboardInterrupt:
        print("\nStopping...")
