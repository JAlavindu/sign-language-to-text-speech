"""
Real-time ASL Recognition Script
"The Eyes" of the system - Detects hands and predicts signs (PyTorch)
"""

import cv2
import numpy as np
import os
import time
import json
import sys
import pyttsx3
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# Add current directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from utils.hand_detector import HandDetector
    from utils.temporal_smoother import TemporalSmoother
except ImportError:
    print("Error: Could not import HandDetector or TemporalSmoother.")
    print("Make sure you are running this script from the project root or ml-model directory.")
    sys.exit(1)

# Configuration
MODEL_PATH = os.path.join(current_dir, "models", "asl_model_final.pth")
CLASS_MAPPING_PATH = os.path.join(current_dir, "models", "class_mapping.json")
IMG_SIZE = 224

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_labels():
    """Load the trained model and class labels"""
    model = None
    labels = {}
    num_classes = 36  # Default to 36 (0-9, A-Z)
    
    # Load Labels first to know num_classes
    if os.path.exists(CLASS_MAPPING_PATH):
        try:
            with open(CLASS_MAPPING_PATH, 'r') as f:
                mapping = json.load(f)
            
            # Use idx_to_class if available (preferred)
            if 'idx_to_class' in mapping:
                labels = {int(k): v for k, v in mapping['idx_to_class'].items()}
            elif 'class_to_idx' in mapping:
                # Reverse the class_to_idx mapping
                labels = {v: k for k, v in mapping['class_to_idx'].items()}
            
            # Get num_classes from the mapping if available
            if 'num_classes' in mapping:
                num_classes = mapping['num_classes']
            else:
                num_classes = len(labels)
            
            print(f"✓ Loaded {num_classes} class labels")
            print(f"  Classes: {', '.join([labels[i] for i in sorted(labels.keys())[:10]])}...")
            
        except Exception as e:
            print(f"⚠ Error loading labels: {e}")
            print(f"  Using default num_classes = {num_classes}")
    else:
        print(f"⚠ Labels not found at {CLASS_MAPPING_PATH}")
        print(f"  Using default num_classes = {num_classes}")

    # Load Model
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading model from {MODEL_PATH}...")
            
            # Rebuild architecture with correct num_classes
            model = models.mobilenet_v2(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            
            # Load weights
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            print("✓ Model loaded successfully")
                
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            model = None
    else:
        print(f"⚠ Model not found at {MODEL_PATH}")
        print("Running in 'Detection Only' mode.")

    return model, labels

def main():
    print("Initializing Camera...")
    cap = cv2.VideoCapture(0)
    
    print(f"Camera opened: {cap.isOpened()}")
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        print("Trying camera index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("ERROR: No camera found!")
            return

    detector = HandDetector(max_num_hands=1)
    smoother = TemporalSmoother(buffer_size=5)
    model, labels = load_model_and_labels()
    
    # Text-to-speech
    engine = pyttsx3.init()
    
    # Preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    p_time = 0
    current_sentence = ""
    last_stable_sign = ""
    stable_frame_count = 0
    STABLE_THRESHOLD = 5
    has_added_current = False
    
    print("\nControls:")
    print("  [SPACE] - Speak sentence")
    print("  [BACKSPACE] - Clear sentence")
    print("  [F] - Toggle Camera Flip (View)")
    print("  [M] - Toggle Model Input Mirror (Fix 'Left' vs 'Right' hand issues)")
    print("  [R] - Toggle Crop Mode (Square/Rect)")
    print("  [Q] - Quit")
    print("\n*** Starting camera loop... ***\n")
    
    flip_camera = False 
    model_flip = False
    use_square_crop = True

    while True:
        success, img = cap.read()
        if not success:
            print("ERROR: Failed to read from camera!") 
            break
            
        if flip_camera:
            img = cv2.flip(img, 1)

        # Detect hands (using updated HandDetector API)
        landmarks, bbox = detector.process(img)
        if landmarks:
            detector.draw_landmarks(img, landmarks)
            lm_list = landmarks
        else:
            lm_list = None
        
        if lm_list and model:
            x, y, w, h = bbox
            
            # --- Crop Logic ---
            h_img, w_img, _ = img.shape
            
            if use_square_crop:
                # Smart Square Cropping (Preserve Aspect Ratio)
                offset = 20
                
                # Calculate center
                center_x, center_y = x + w // 2, y + h // 2
                
                # Make it square based on the largest dimension
                max_dim = max(w, h) + (offset * 2)
                half_dim = max_dim // 2
                
                x1 = max(center_x - half_dim, 0)
                y1 = max(center_y - half_dim, 0)
                x2 = min(center_x + half_dim, w_img)
                y2 = min(center_y + half_dim, h_img)
            else:
                # Rectangular Crop (Squashed Aspect Ratio)
                offset = 20
                x1 = max(x - offset, 0)
                y1 = max(y - offset, 0)
                x2 = min(x + w + offset, w_img)
                y2 = min(y + h + offset, h_img)
            
            img_crop = img[y1:y2, x1:x2]
            
            if img_crop.size > 0:
                try:
                    # Input Preparation
                    img_crop_input = img_crop
                    flip_info = ""

                    # Auto-Correction or Manual Flip
                    if model_flip:
                        img_crop_input = cv2.flip(img_crop, 1)
                        flip_info = "Manual Flip"
                    elif hasattr(detector, 'handedness') and detector.handedness:
                        # Adaptive: If Right Hand model, flip 'Left' hands (which MP calls 'Right')
                        if detector.handedness == "Right":
                            img_crop_input = cv2.flip(img_crop, 1)
                            flip_info = "Auto Flip (L->R)"
                        else:
                            flip_info = "Right Hand"

                    # Debug: Show what the model sees
                    debug_img = img_crop_input.copy()
                    cv2.putText(debug_img, flip_info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                    cv2.imshow("Model Input (Crop)", debug_img)

                    # Convert BGR to RGB
                    img_rgb = cv2.cvtColor(img_crop_input, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    
                    # Transform
                    input_tensor = transform(pil_img).unsqueeze(0).to(device)
                    
                    # Inference
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        
                        index = predicted.item()
                        conf_val = confidence.item()

                        # Debug: Print top 3 predictions
                        top3_prob, top3_idx = torch.topk(probabilities, 3)
                        print(f"Top 3: {[(labels.get(idx.item(), '?'), f'{prob.item():.2f}') for prob, idx in zip(top3_prob[0], top3_idx[0])]}")
                    
                    # Smooth prediction
                    smoother.add_prediction(index)
                    smoothed_index = smoother.get_smoothed_prediction()
                    
                    label = labels.get(smoothed_index, labels.get(str(smoothed_index), "?"))
                    
                    # Sentence Construction Logic
                    if label == last_stable_sign:
                        stable_frame_count += 1
                    else:
                        stable_frame_count = 0
                        last_stable_sign = label
                        has_added_current = False
                        
                    if stable_frame_count > STABLE_THRESHOLD and not has_added_current:
                        current_sentence += label
                        has_added_current = True
                    
                    # Display result
                    if conf_val > 0.7:
                        text = f"{label} ({conf_val*100:.0f}%)"
                        color = (0, 255, 0) # Green
                    else:
                        text = f"{label}? ({conf_val*100:.0f}%)"
                        color = (0, 255, 255) # Yellow
                        
                    # Draw label background
                    (t_w, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    cv2.rectangle(img, (x1, y1 - 35), (x1 + t_w, y1), color, cv2.FILLED)
                    cv2.putText(img, text, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                               
                except Exception as e:
                    # Ignore prediction errors (e.g. empty crop)
                    # print(f"Prediction error: {e}")
                    pass
        elif bbox:
             # Detection Only Mode
            x, y, w, h = bbox
            cv2.putText(img, "Hand Detected", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Display Sentence
        cv2.rectangle(img, (0, img.shape[0] - 60), (img.shape[1], img.shape[0]), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f"Sentence: {current_sentence}", (20, img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Calculate and show FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        
        cv2.putText(img, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        status_text = f"CamFlip: {'ON' if flip_camera else 'OFF'} | ModFlip: {'ON' if model_flip else 'OFF'} | Crop: {'Sqr' if use_square_crop else 'Rect'}"
        cv2.putText(img, status_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show frame
        cv2.imshow("ASL Real-time Recognition", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            flip_camera = not flip_camera
        elif key == ord('m'):
            model_flip = not model_flip
        elif key == ord('r'):
            use_square_crop = not use_square_crop
        elif key == ord(' '): # Space to speak
            if engine and current_sentence:
                engine.say(current_sentence)
                engine.runAndWait()
        elif key == 8: # Backspace to clear
            current_sentence = ""

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
