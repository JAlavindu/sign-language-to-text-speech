from flask import Flask, render_template, Response, jsonify, request
import cv2
import sys
import os
import numpy as np
import base64
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Add ml-model to path so we can import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_model_dir = os.path.join(os.path.dirname(current_dir), 'ml-model')
sys.path.append(ml_model_dir)

from utils.hand_detector import HandDetector
from utils.temporal_smoother import TemporalSmoother

app = Flask(__name__)

# Initialize Global Models/Utils
detector = HandDetector(max_num_hands=1)
smoother = TemporalSmoother(buffer_size=10)
model = None
labels = {}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_PATH = os.path.join(ml_model_dir, "models", "asl_model_final.pth")
LABELS_PATH = os.path.join(ml_model_dir, "datasets", "processed", "class_mapping.json")

# Image Preprocessing (Must match training!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_resources():
    global model, labels
    try:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, 'r') as f:
                mapping = json.load(f)
                if 'idx_to_class' in mapping:
                    labels = {int(k): v for k, v in mapping['idx_to_class'].items()}
                else:
                    labels = {int(k): v for k, v in mapping.items()}
            print(f"Labels Loaded: {len(labels)} classes")
        
        if os.path.exists(MODEL_PATH):
            # 1. Recreate the Model Architecture
            model = models.mobilenet_v2(weights=None) # No need to download weights, we load our own
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(labels))
            )
            
            # 2. Load Weights
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval() # Set to evaluation mode
            print("Vision Model Loaded (PyTorch)")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")

    except Exception as e:
        print(f"Error loading resources: {e}")

load_resources()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from POST request
        data = request.json
        image_data = data['image']
        
        # Decode base64 image
        header, encoded = image_data.split(",", 1)
        binary = base64.b64decode(encoded)
        image_array = np.frombuffer(binary, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 1. Detect Hand
        hands, img_drawn = detector.find_hands(frame)
        
        prediction_text = "No Hand"
        confidence = 0.0

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Crop Hand 
            img_h, img_w, _ = frame.shape
            crop = frame[max(0,y-20):min(img_h, y+h+20), max(0,x-20):min(img_w, x+w+20)]
            
            if crop.size != 0:
                # Convert to PIL for PyTorch Transforms
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                input_tensor = transform(crop_pil).unsqueeze(0).to(device)

                # 2. Predict
                if model:
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        top_p, top_class = probabilities.topk(1, dim=1)
                        
                        predicted_idx = top_class.item()
                        confidence = top_p.item()
                    
                    raw_char = labels.get(predicted_idx, "?")
                    
                    # 3. Smooth
                    prediction_text = smoother.add_prediction(raw_char)
                else:
                    prediction_text = "Model Error"

        return jsonify({
            'prediction': prediction_text,
            'confidence': f"{confidence:.2f}"
        })

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the Flask app
    # host='0.0.0.0' allows access from other devices on the network
    # ssl_context='adhoc' is often required for webcam access on non-localhost
    app.run(debug=True, host='0.0.0.0', port=5000)
