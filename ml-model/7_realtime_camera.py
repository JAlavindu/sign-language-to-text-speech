"""
Real-time ASL Recognition Script
"The Eyes" of the system - Detects hands and predicts signs
"""

import cv2
import numpy as np
import os
import time
import json
import sys
import pyttsx3

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
MODEL_PATH = os.path.join(current_dir, "models", "asl_model_final.h5")
CLASS_MAPPING_PATH = os.path.join(current_dir, "datasets", "processed", "class_mapping.json")
IMG_SIZE = 224

def load_model_and_labels():
    """Load the trained model and class labels"""
    model = None
    labels = {}
    
    # Load Model
    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            print("Loading model... (this might take a moment)")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
    else:
        print(f"⚠ Model not found at {MODEL_PATH}")
        print("Running in 'Detection Only' mode.")

    # Load Labels
    if os.path.exists(CLASS_MAPPING_PATH):
        try:
            with open(CLASS_MAPPING_PATH, 'r') as f:
                mapping = json.load(f)
                # Handle both formats of mapping
                if 'idx_to_class' in mapping:
                    labels = mapping['idx_to_class']
                else:
                    labels = mapping
                
                # Ensure keys are integers
                labels = {int(k): v for k, v in labels.items()}
            print(f"✓ Loaded {len(labels)} class labels")
        except Exception as e:
            print(f"⚠ Error loading labels: {e}")
    
    return model, labels

def main():
    print("\n" + "="*60)
    print("ASL REAL-TIME RECOGNITION")
    print("="*60)

    # Initialize Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize Hand Detector
    try:
        print("Initializing MediaPipe Hand Detector...")
        detector = HandDetector(max_hands=1, detection_con=0.7)
        smoother = TemporalSmoother(buffer_size=10)
    except Exception as e:
        print(f"\nError initializing MediaPipe: {e}")
        print("\nCRITICAL: MediaPipe is required for this script.")
        print("Please install it with: pip install mediapipe")
        print("Note: MediaPipe may not support Python 3.13 yet. Try Python 3.10 or 3.11.")
        return

    # Initialize Text-to-Speech
    try:
        print("Initializing Text-to-Speech...")
        engine = pyttsx3.init()
    except Exception as e:
        print(f"⚠ Warning: Could not initialize Text-to-Speech: {e}")
        engine = None

    # Load Model
    model, labels = load_model_and_labels()
    
    print("\nStarting camera feed...")
    print("Controls:")
    print("  [SPACE] - Speak sentence")
    print("  [BACKSPACE] - Clear sentence")
    print("  [Q] - Quit")
    
    p_time = 0
    
    # Sentence construction variables
    current_sentence = ""
    last_stable_sign = None
    stable_frame_count = 0
    STABLE_THRESHOLD = 20  # Number of frames to hold sign before adding
    has_added_current = False
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam")
            break

        img_output = img.copy()
        
        # Detect Hand
        img = detector.find_hands(img)
        lm_list, bbox = detector.find_position(img, draw=False)

        if bbox:
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Add padding for better crop
            offset = 20
            y1, y2 = max(0, y - offset), min(img.shape[0], h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], w + offset)
            
            # Draw bounding box on display image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            # Prediction (only if model exists)
            if model is not None:
                try:
                    # Crop hand
                    img_crop = img_output[y1:y2, x1:x2]
                    
                    if img_crop.size > 0:
                        # Preprocess
                        img_resize = cv2.resize(img_crop, (IMG_SIZE, IMG_SIZE))
                        img_norm = img_resize / 255.0
                        img_input = np.expand_dims(img_norm, axis=0)
                        
                        # Predict
                        prediction = model.predict(img_input, verbose=0)
                        index = np.argmax(prediction)
                        confidence = prediction[0][index]
                        
                        # Smooth prediction
                        smoother.add_prediction(index)
                        smoothed_index = smoother.get_smoothed_prediction()
                        
                        label = labels.get(smoothed_index, str(smoothed_index))
                        
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
                        if confidence > 0.7:
                            text = f"{label} ({confidence*100:.0f}%)"
                            color = (0, 255, 0) # Green
                        else:
                            text = f"{label}? ({confidence*100:.0f}%)"
                            color = (0, 255, 255) # Yellow
                            
                        # Draw label background
                        (t_w, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        cv2.rectangle(img, (x1, y1 - 35), (x1 + t_w, y1), color, cv2.FILLED)
                        cv2.putText(img, text, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                                   
                except Exception as e:
                    # Ignore prediction errors (e.g. empty crop)
                    pass
            else:
                # Detection Only Mode
                cv2.putText(img, "Hand Detected", (x1, y1 - 10), 
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

        # Show frame
        cv2.imshow("ASL Real-time Recognition", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
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
