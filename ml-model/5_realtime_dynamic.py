import cv2
import torch
import numpy as np
import mediapipe as mp
import torch.nn as nn
from collections import deque
import importlib

# Dynamic import for our NLP module
nlp_module = importlib.import_module("6_nlp_translation")
ASLTranslator = nlp_module.ASLTranslator

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

# --- Config ---
MAX_FRAMES = 50

# Feature Mapping Constants
POSE_START, POSE_END = 0, 132
FACE_START, FACE_END = 132, 1536
HANDS_START, HANDS_END = 1536, 1662

PHYSICAL_FEATURES = (POSE_END - POSE_START) + (HANDS_END - HANDS_START) # 258
FACE_FEATURES = FACE_END - FACE_START # 1404

PHYSICAL_HIDDEN_SIZE = 128
FACE_HIDDEN_SIZE = 64
NUM_LAYERS = 2

# Update this with the classes you actually trained on
CLASSES = ['hello', 'please', 'thank_you']  

# --- Updated Dual-Stream Model Definition ---
class DualStreamASLModel(nn.Module):
    def __init__(self, physical_input_size, face_input_size, physical_hidden, face_hidden, num_layers, num_classes):
        super(DualStreamASLModel, self).__init__()
        
        self.physical_lstm = nn.LSTM(physical_input_size, physical_hidden, num_layers, batch_first=True, dropout=0.2)
        self.face_lstm = nn.LSTM(face_input_size, face_hidden, num_layers, batch_first=True, dropout=0.2)
        
        combined_hidden = physical_hidden + face_hidden
        self.fc1 = nn.Linear(combined_hidden, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, physical_x, face_x):
        phys_out, _ = self.physical_lstm(physical_x)
        phys_last = phys_out[:, -1, :] 
        
        face_out, _ = self.face_lstm(face_x)
        face_last = face_out[:, -1, :] 
        
        combined = torch.cat((phys_last, face_last), dim=1)
        
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the new Dual-Stream Model
    model = DualStreamASLModel(
        physical_input_size=PHYSICAL_FEATURES, 
        face_input_size=FACE_FEATURES, 
        physical_hidden=PHYSICAL_HIDDEN_SIZE, 
        face_hidden=FACE_HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        num_classes=len(CLASSES)
    ).to(device)
    
    try:
        model.load_state_dict(torch.load('models/dual_stream_gesture_model.pth', map_location=device))
        model.eval()
        print("Dual-Stream Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}. Make sure you train the dual_stream model first!")
        return

    # 2. Load NLP Translator
    print("Initializing NLP Translator...")
    translator = ASLTranslator()

    cap = cv2.VideoCapture(0)
    
    sequence = deque(maxlen=MAX_FRAMES)
    current_prediction = "Waiting..."
    current_sentence = []       
    translated_display = ""     

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract full 1662 array
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # Once we have 50 frames, form the prediction
            if len(sequence) == MAX_FRAMES:
                # Shape is (50, 1662)
                seq_array = np.array(sequence) 
                
                # --- Split the data exactly as we did in the Dataset Class ---
                pose_data = seq_array[:, POSE_START:POSE_END]           # (50, 132)
                face_data = seq_array[:, FACE_START:FACE_END]           # (50, 1404)
                hands_data = seq_array[:, HANDS_START:HANDS_END]         # (50, 126)
                
                phys_data_np = np.concatenate([pose_data, hands_data], axis=1) # (50, 258)
                
                # Add batch dimension -> Shape becomes (1, 50, channels)
                phys_data_tensor = torch.FloatTensor(np.expand_dims(phys_data_np, axis=0)).to(device)
                face_data_tensor = torch.FloatTensor(np.expand_dims(face_data, axis=0)).to(device)
                
                with torch.no_grad():
                    # Pass the two streams into the model
                    output = model(phys_data_tensor, face_data_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    predicted_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_idx].item()
                    
                    if confidence > 0.85:  
                        word = CLASSES[predicted_idx]
                        current_prediction = f"{word} ({confidence*100:.1f}%)"
                        
                        if len(current_sentence) == 0 or current_sentence[-1] != word:
                            current_sentence.append(word)

            # Visual Display
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, current_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.rectangle(image, (0, 380), (640, 420), (50, 50, 50), -1)
            raw_text_display = " ".join(current_sentence)
            cv2.putText(image, f"ASL: {raw_text_display}", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

            cv2.rectangle(image, (0, 420), (640, 480), (100, 30, 30), -1)
            cv2.putText(image, f"Speech: {translated_display}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Dual-Stream ASL Engine', image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):         
                break
            elif key == 8:              
                current_sentence = []
                translated_display = ""
            elif key == 32:             
                if len(current_sentence) > 0:
                    translated_display = translator.translate_gloss_to_english(current_sentence)
                    cv2.imshow('Dual-Stream ASL Engine', image)
                    cv2.waitKey(1)
                    translator.speak(translated_display)
                    current_sentence = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()