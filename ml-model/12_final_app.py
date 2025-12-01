import asyncio
import cv2
import numpy as np
import tensorflow as tf
import joblib
import os
import sys
import time
import pyttsx3
from bleak import BleakClient, BleakScanner

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.hand_detector import HandDetector
from utils.temporal_smoother import TemporalSmoother
from utils.sensor_preprocessor import SensorPreprocessor
from utils.adaptive_fusion import AdaptiveFusion

# Configuration
DEVICE_NAME = "ASL_Glove_001"
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
CHARACTERISTIC_UUID_TX = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# Model Paths
IMG_MODEL_PATH = os.path.join(current_dir, "models", "asl_model_final.h5")
SENS_MODEL_PATH = os.path.join(current_dir, "models", "sensor_model_final.h5")
SENS_SCALER_PATH = os.path.join(current_dir, "models", "sensor_scaler.pkl")
SENS_LABELS_PATH = os.path.join(current_dir, "models", "sensor_labels.pkl")
IMG_LABELS_PATH = os.path.join(current_dir, "datasets", "processed", "class_mapping.json")

class TextToSpeech:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed
        except Exception as e:
            print(f"TTS Error: {e}")
            self.engine = None

    def speak(self, text):
        if self.engine and text:
            print(f"Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()

class SentenceBuilder:
    def __init__(self):
        self.sentence = []
        self.last_added_time = 0
        self.cooldown = 2.0  # Seconds before adding same letter again

    def add(self, sign):
        current_time = time.time()
        
        # Logic: Don't add the same letter repeatedly unless 2 seconds passed
        if self.sentence and self.sentence[-1] == sign:
            if current_time - self.last_added_time < self.cooldown:
                return False
        
        self.sentence.append(sign)
        self.last_added_time = current_time
        return True

    def get_text(self):
        return "".join(self.sentence)

    def clear(self):
        self.sentence = []

    def backspace(self):
        if self.sentence:
            self.sentence.pop()

class FinalApp:
    def __init__(self):
        print("Initializing Final Application...")
        
        # Components
        self.tts = TextToSpeech()
        self.builder = SentenceBuilder()
        self.detector = HandDetector(max_num_hands=1)
        self.fusion = AdaptiveFusion()
        self.smoother = TemporalSmoother(buffer_size=10)
        
        # Load Models
        self.load_models()
        
        # State
        self.sensor_buffer = []
        self.latest_sensor_probs = None
        self.latest_cam_probs = None
        self.ble_connected = False
        self.cap = None

    def load_models(self):
        try:
            self.img_model = tf.keras.models.load_model(IMG_MODEL_PATH)
            self.sens_model = tf.keras.models.load_model(SENS_MODEL_PATH)
            self.sens_scaler = joblib.load(SENS_SCALER_PATH)
            self.sens_preprocessor = SensorPreprocessor()
            self.sens_preprocessor.scaler = self.sens_scaler
            
            import json
            with open(IMG_LABELS_PATH, 'r') as f:
                mapping = json.load(f)
                self.labels = {int(k): v for k, v in mapping.get('idx_to_class', mapping).items()}
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)

    def process_sensor_packet(self, data):
        import struct
        try:
            if len(data) != 43: return
            unpacked = struct.unpack('<I5H3f3f5B', data)
            features = list(unpacked[1:17])
            
            self.sensor_buffer.append(features)
            if len(self.sensor_buffer) > 50:
                self.sensor_buffer.pop(0)
                
            if len(self.sensor_buffer) == 50:
                arr = np.array(self.sensor_buffer)
                scaled = self.sens_preprocessor.transform(arr)
                input_data = scaled.reshape(1, 50, 16)
                self.latest_sensor_probs = self.sens_model.predict(input_data, verbose=0)[0]
        except: pass

    async def run(self):
        self.cap = cv2.VideoCapture(0)
        
        print(f"Scanning for {DEVICE_NAME}...")
        device = await BleakScanner.find_device_by_name(DEVICE_NAME)
        
        if device:
            async with BleakClient(device) as client:
                self.ble_connected = True
                await client.start_notify(CHARACTERISTIC_UUID_TX, lambda s, d: self.process_sensor_packet(d))
                await self.main_loop()
        else:
            print("Running Camera Only Mode")
            await self.main_loop()

    async def main_loop(self):
        last_sign = None
        stable_frames = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # 1. Camera Inference
            landmarks, bbox = self.detector.process(frame)
            if bbox:
                hand_img = self.detector.crop_hand(frame, bbox)
                if hand_img is not None:
                    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(hand_img.astype(np.float32))
                    img_array = np.expand_dims(img_array, axis=0)
                    self.latest_cam_probs = self.img_model.predict(img_array, verbose=0)[0]
                    self.detector.draw_landmarks(frame, landmarks)
            else:
                self.latest_cam_probs = None

            # 2. Fusion
            fused_probs, pred_idx, conf = self.fusion.fuse(self.latest_cam_probs, self.latest_sensor_probs)
            
            # 3. Smoothing & Sentence Logic
            current_sign = "?"
            if pred_idx is not None:
                self.smoother.add_prediction(pred_idx)
                smoothed_idx = self.smoother.get_smoothed_prediction()
                if smoothed_idx is not None:
                    current_sign = self.labels.get(smoothed_idx, "?")
            
            # Auto-add to sentence if stable for 20 frames
            if current_sign != "?" and current_sign == last_sign:
                stable_frames += 1
                if stable_frames > 20: # ~1 second
                    if self.builder.add(current_sign):
                        # Visual feedback
                        cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)
                    stable_frames = 0
            else:
                stable_frames = 0
                last_sign = current_sign

            # 4. UI Display
            # Top Bar (Current Sign)
            cv2.rectangle(frame, (0, 0), (640, 60), (50, 50, 50), -1)
            cv2.putText(frame, f"Sign: {current_sign} ({conf*100:.0f}%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Bottom Bar (Sentence)
            cv2.rectangle(frame, (0, 400), (640, 480), (255, 255, 255), -1)
            sentence_text = self.builder.get_text()
            cv2.putText(frame, sentence_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Instructions
            cv2.putText(frame, "SPACE: Speak | BACK: Delete | Q: Quit", (20, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            cv2.imshow("ASL to Speech", frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == 32: # Space
                self.tts.speak(sentence_text)
                self.builder.clear()
            elif key == 8: # Backspace
                self.builder.backspace()
                
            await asyncio.sleep(0.01)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FinalApp()
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass
