import cv2
import torch
import numpy as np
import mediapipe as mp
import torch.nn as nn
from collections import deque

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

# --- Config ---
MAX_FRAMES = 50
INPUT_FEATURES = 1662
HIDDEN_SIZE = 128
NUM_LAYERS = 2
# Update this with the classes you actually trained on
CLASSES = ['hello', 'please', 'thank_you']  

# --- LSTM Model Definition (Must match training script) ---
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
    model = GestureLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, len(CLASSES)).to(device)
    
    # Load Weights
    try:
        model.load_state_dict(torch.load('models/dynamic_gesture_model.pth', map_location=device))
        model.eval()
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}. Train the model first!")
        return

    cap = cv2.VideoCapture(0)
    
    # A rolling buffer of the last exactly 50 frames
    sequence = deque(maxlen=MAX_FRAMES)
    current_prediction = "Waiting..."

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Optional: Draw landmarks to see them on screen
            # mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # Only predict once we have a full sequence of 50 frames
            if len(sequence) == MAX_FRAMES:
                input_data = np.expand_dims(np.array(sequence), axis=0) # Add batch dimension -> [1, 50, 1662]
                input_tensor = torch.FloatTensor(input_data).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    predicted_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_idx].item()
                    
                    if confidence > 0.7:  # Confidence threshold
                        current_prediction = f"{CLASSES[predicted_idx]} ({confidence*100:.1f}%)"

            # Display prediction
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, current_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Dynamic ASL Recognition', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()