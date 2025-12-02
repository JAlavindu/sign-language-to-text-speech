import customtkinter as ctk
import cv2
import PIL.Image, PIL.ImageTk
import threading
import asyncio
import time
import sys
import os
import numpy as np
import tensorflow as tf
import joblib
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

# Paths
IMG_MODEL_PATH = os.path.join(current_dir, "models", "asl_model_final.h5")
SENS_MODEL_PATH = os.path.join(current_dir, "models", "sensor_model_final.h5")
SENS_SCALER_PATH = os.path.join(current_dir, "models", "sensor_scaler.pkl")
SENS_LABELS_PATH = os.path.join(current_dir, "models", "sensor_labels.pkl")
IMG_LABELS_PATH = os.path.join(current_dir, "datasets", "processed", "class_mapping.json")

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class BackendSystem:
    def __init__(self):
        self.running = False
        self.cap = None
        self.ble_connected = False
        self.sensor_buffer = []
        self.latest_sensor_probs = None
        self.latest_cam_probs = None
        
        # Components
        self.detector = HandDetector(max_num_hands=1)
        self.fusion = AdaptiveFusion()
        self.smoother = TemporalSmoother(buffer_size=10)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        
        # Sentence Building
        self.sentence = []
        self.last_added_time = 0
        self.stable_frames = 0
        self.last_sign = None
        
        # Load Models
        self.load_models()

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
            print("Models Loaded")
        except Exception as e:
            print(f"Error loading models: {e}")

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        # Start BLE in separate thread
        self.ble_thread = threading.Thread(target=self.run_ble_loop, daemon=True)
        self.ble_thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def run_ble_loop(self):
        asyncio.run(self.ble_task())

    async def ble_task(self):
        print(f"Scanning for {DEVICE_NAME}...")
        device = await BleakScanner.find_device_by_name(DEVICE_NAME)
        if device:
            async with BleakClient(device) as client:
                self.ble_connected = True
                await client.start_notify(CHARACTERISTIC_UUID_TX, lambda s, d: self.process_sensor_packet(d))
                while self.running:
                    await asyncio.sleep(0.1)
                self.ble_connected = False
        else:
            print("BLE Device not found")
            self.ble_connected = False

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

    def get_frame_and_prediction(self):
        if not self.cap: return None, "?", 0.0

        ret, frame = self.cap.read()
        if not ret: return None, "?", 0.0

        frame = cv2.flip(frame, 1) # Mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Camera Inference
        landmarks, bbox = self.detector.process(frame) # Note: detector expects BGR
        if bbox:
            hand_img = self.detector.crop_hand(frame, bbox)
            if hand_img is not None:
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(hand_img.astype(np.float32))
                img_array = np.expand_dims(img_array, axis=0)
                self.latest_cam_probs = self.img_model.predict(img_array, verbose=0)[0]
                self.detector.draw_landmarks(rgb_frame, landmarks) # Draw on RGB for display
        else:
            self.latest_cam_probs = None

        # 2. Fusion
        fused_probs, pred_idx, conf = self.fusion.fuse(self.latest_cam_probs, self.latest_sensor_probs)
        
        # 3. Smoothing
        current_sign = "?"
        if pred_idx is not None:
            self.smoother.add_prediction(pred_idx)
            smoothed_idx = self.smoother.get_smoothed_prediction()
            if smoothed_idx is not None:
                current_sign = self.labels.get(smoothed_idx, "?")

        # 4. Sentence Logic
        if current_sign != "?" and current_sign == self.last_sign:
            self.stable_frames += 1
            if self.stable_frames > 20: # ~1 sec hold
                if time.time() - self.last_added_time > 2.0:
                    self.sentence.append(current_sign)
                    self.last_added_time = time.time()
                    self.stable_frames = 0
        else:
            self.stable_frames = 0
            self.last_sign = current_sign

        return rgb_frame, current_sign, conf

    def speak_sentence(self):
        text = "".join(self.sentence)
        if text:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            self.sentence = []

    def clear_sentence(self):
        self.sentence = []

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sign Language to Speech - Pro")
        self.geometry("1100x700")
        
        self.backend = BackendSystem()
        
        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="ASL Glove", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.start_btn = ctk.CTkButton(self.sidebar, text="Start System", command=self.toggle_system)
        self.start_btn.grid(row=1, column=0, padx=20, pady=10)

        self.status_cam = ctk.CTkLabel(self.sidebar, text="Camera: OFF", text_color="gray")
        self.status_cam.grid(row=2, column=0, padx=20, pady=5)
        
        self.status_ble = ctk.CTkLabel(self.sidebar, text="Glove: OFF", text_color="gray")
        self.status_ble.grid(row=3, column=0, padx=20, pady=5)

        # Main Area
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Video Feed
        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew", pady=(0, 20))

        # Text Output Area
        self.text_frame = ctk.CTkFrame(self.main_frame, height=150)
        self.text_frame.grid(row=1, column=0, sticky="ew")
        
        self.current_sign_label = ctk.CTkLabel(self.text_frame, text="?", font=ctk.CTkFont(size=60, weight="bold"))
        self.current_sign_label.pack(side="left", padx=40)
        
        self.sentence_label = ctk.CTkLabel(self.text_frame, text="Waiting for input...", font=ctk.CTkFont(size=24))
        self.sentence_label.pack(side="left", padx=20, fill="x", expand=True)

        # Controls
        self.controls_frame = ctk.CTkFrame(self.main_frame, height=50, fg_color="transparent")
        self.controls_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        
        self.speak_btn = ctk.CTkButton(self.controls_frame, text="Speak 🔊", command=self.speak, fg_color="green")
        self.speak_btn.pack(side="right", padx=10)
        
        self.clear_btn = ctk.CTkButton(self.controls_frame, text="Clear ⌫", command=self.clear, fg_color="red")
        self.clear_btn.pack(side="right", padx=10)

        self.system_active = False
        self.update_gui()

    def toggle_system(self):
        if not self.system_active:
            self.backend.start()
            self.start_btn.configure(text="Stop System", fg_color="red")
            self.system_active = True
        else:
            self.backend.stop()
            self.start_btn.configure(text="Start System", fg_color="#1f538d")
            self.system_active = False
            self.video_label.configure(image=None)

    def speak(self):
        threading.Thread(target=self.backend.speak_sentence).start()

    def clear(self):
        self.backend.clear_sentence()

    def update_gui(self):
        if self.system_active:
            # Update Video
            frame, sign, conf = self.backend.get_frame_and_prediction()
            if frame is not None:
                # Resize for GUI
                h, w, _ = frame.shape
                ratio = 640 / w
                new_h = int(h * ratio)
                frame = cv2.resize(frame, (640, new_h))
                
                img = PIL.Image.fromarray(frame)
                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, new_h))
                self.video_label.configure(image=imgtk)
                self.video_label.image = imgtk # Keep reference

            # Update Labels
            self.current_sign_label.configure(text=sign)
            
            sentence = "".join(self.backend.sentence)
            self.sentence_label.configure(text=sentence if sentence else "...")

            # Update Status
            self.status_cam.configure(text="Camera: ON", text_color="green")
            if self.backend.ble_connected:
                self.status_ble.configure(text="Glove: CONNECTED", text_color="green")
            else:
                self.status_ble.configure(text="Glove: SEARCHING...", text_color="orange")
        else:
            self.status_cam.configure(text="Camera: OFF", text_color="gray")
            self.status_ble.configure(text="Glove: OFF", text_color="gray")

        self.after(30, self.update_gui)

if __name__ == "__main__":
    app = App()
    app.mainloop()
