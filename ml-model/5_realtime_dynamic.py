import cv2
import torch
import numpy as np
import mediapipe as mp
import torch.nn as nn
from collections import deque
import importlib

# Dynamic import for our NLP module (since the file starts with a number)
nlp_module = importlib.import_module("6_nlp_translation")
ASLTranslator = nlp_module.ASLTranslator

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

# --- Config ---
MAX_FRAMES = 50
INPUT_FEATURES = 1662
HIDDEN_SIZE = 128
NUM_LAYERS = 2
# Update this with the classes you actually trained on
CLASSES = ['hello', 'please', 'thank_you']  

# --- LSTM Model Definition ---
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load LSTM Gesture Model
    model = GestureLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, len(CLASSES)).to(device)
    try:
        model.load_state_dict(torch.load('models/dynamic_gesture_model.pth', map_location=device))
        model.eval()
        print("Gesture Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}. Train the model first!")
        return

    # 2. Load NLP & Speech System
    print("Initializing NLP Translator...")
    translator = ASLTranslator()

    cap = cv2.VideoCapture(0)
    
    # Tracking variables
    sequence = deque(maxlen=MAX_FRAMES)
    current_prediction = "Waiting..."
    current_sentence = []       # Holds the raw ASL glosses (e.g., ["store", "he", "go"])
    translated_display = ""     # Holds the final fluent sentence

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # Predict if we have enough frames
            if len(sequence) == MAX_FRAMES:
                input_data = np.expand_dims(np.array(sequence), axis=0) # [1, 50, 1662]
                input_tensor = torch.FloatTensor(input_data).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    predicted_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_idx].item()
                    
                    # High confidence threshold to avoid garbage data
                    if confidence > 0.85:  
                        word = CLASSES[predicted_idx]
                        current_prediction = f"{word} ({confidence*100:.1f}%)"
                        
                        # Add word to sentence if it's not the same as the last recorded word
                        if len(current_sentence) == 0 or current_sentence[-1] != word:
                            current_sentence.append(word)

            # --- Visual Display ---
            # 1. Top bar: Current real-time gesture prediction
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, current_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 2. Bottom bar: The accumulated ASL sentence
            cv2.rectangle(image, (0, 380), (640, 420), (50, 50, 50), -1)
            raw_text_display = " ".join(current_sentence)
            cv2.putText(image, f"ASL: {raw_text_display}", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

            # 3. Lowest bar: The translated English sentence
            cv2.rectangle(image, (0, 420), (640, 480), (100, 30, 30), -1)
            cv2.putText(image, f"Speech: {translated_display}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Dynamic ASL System (End-to-End)', image)

            # --- Key Press Logic ---
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):         # Quit
                break
            elif key == 8:              # Backspace: clear the sentence
                current_sentence = []
                translated_display = ""
            elif key == 32:             # Spacebar: Translate and Speak
                if len(current_sentence) > 0:
                    print(f"Translating: {current_sentence}")
                    translated_display = translator.translate_gloss_to_english(current_sentence)
                    print(f"Result: {translated_display}")
                    
                    # Force window update before freezing to speak
                    cv2.imshow('Dynamic ASL System (End-to-End)', image)
                    cv2.waitKey(1)
                    
                    translator.speak(translated_display)
                    # Optional: clear the sentence after speaking
                    current_sentence = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()