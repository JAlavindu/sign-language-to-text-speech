import io 
import json
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from .model_loader import load_model_architecture
from .utils import get_transform
from fastapi.middleware.cors import CORSMiddleware
import os

# --- Configuration ---
# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "asl_model_final.pth")
MAPPING_PATH = os.path.join(ARTIFACTS_DIR, "class_mapping.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="ASL Recognition API")

# --- CORS Middleware ---
# Allows your React frontend (usually running on localhost:5173) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace specific URL like ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables for Model ---
model = None
class_names = []

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    global model, class_names
    print("Loading model and artifacts...")
    
    # 1. Load Class Mapping
    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(f"Mapping file not found at {MAPPING_PATH}")
    
    with open(MAPPING_PATH, 'r') as f:
        mapping_data = json.load(f)
        # Assuming mapping is like {"idx_to_class": {"0": "A", "1": "B"}} 
        # or a direct list depending on how you saved it. 
        # Based on typical PyTorch ImageFolder, we usually need the list of names.
        # Let's assume idx_to_class exists, otherwise we invert class_to_idx
        if 'idx_to_class' in mapping_data:
            idx_map = mapping_data['idx_to_class']
            # Ensure keys are sorted integers to get correct list order
            class_names = [idx_map[str(i)] for i in range(len(idx_map))]
        elif 'class_to_idx' in mapping_data:
             # Invert the dictionary
             inv_map = {v: k for k, v in mapping_data['class_to_idx'].items()}
             class_names = [inv_map[i] for i in range(len(inv_map))]
        else:
             # Fallback if just a raw list or dict
             print("Warning: unexpected JSON format, checking keys...")
             class_names = list(mapping_data.keys()) # simplistic fallback

    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes.")

    # 2. Load Model Architecture
    model = load_model_architecture(num_classes)
    
    # 3. Load Weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval() # Set to evaluation mode (disable dropout, etc)
    print("Model loaded successfully!")


@app.get("/")
def read_root():
    return {"status": "healthy", "model": "MobileNetV2_ASL"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image file, preprocesses it, and returns the ASL prediction.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        transform = get_transform()
        input_tensor = transform(image).unsqueeze(0) # Add batch dimension (1, 3, 224, 224)
        input_tensor = input_tensor.to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
            
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
            
            prediction = class_names[predicted_idx]
            
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "all_classes": class_names # Optional: useful for debugging
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))