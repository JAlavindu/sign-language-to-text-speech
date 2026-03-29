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

# --- Optimized Transform ---
# Initialize transform once to avoid overhead per request
TRANSFORM = get_transform()

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
    
    # 1. Load Weights (to determine correct number of classes)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    # Check the output size of the final layer explicitly
    # 'classifier.4.weight' corresponds to the last Linear layer in our MobileNetV2 head
    if 'classifier.4.weight' in state_dict:
        num_classes_model = state_dict['classifier.4.weight'].shape[0]
        print(f"Detected {num_classes_model} classes in model weights.")
    else:
        # Fallback if layer name differs (unlikely given our loader)
        num_classes_model = len(state_dict[list(state_dict.keys())[-1]]) 

    print("Loading model and artifacts...")
    
    # 2. Load Class Mapping
    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(f"Mapping file not found at {MAPPING_PATH}")
    
    with open(MAPPING_PATH, 'r') as f:
        mapping_data = json.load(f)
        if 'idx_to_class' in mapping_data:
            idx_map = mapping_data['idx_to_class']
            raw_class_names = [idx_map[str(i)] for i in range(len(idx_map))]
        elif 'class_to_idx' in mapping_data:
             inv_map = {v: k for k, v in mapping_data['class_to_idx'].items()}
             raw_class_names = [inv_map[i] for i in range(len(inv_map))]
        else:
             raw_class_names = list(mapping_data.keys())

    # 3. reconcile classes
    if len(raw_class_names) == num_classes_model:
        class_names = raw_class_names
    else:
        print(f"Warning: Mapping has {len(raw_class_names)} classes but model has {num_classes_model}.")
        print("Attempting to filter class names to match model...")
        
        # Heuristic: The model (36) is likely generic (0-9, A-Z). 
        # The mapping (40) likely has extras like 'DEL', 'SPACE', 'NOTHING'.
        # We filter for single-character alphanumeric classes.
        filtered_names = [c for c in raw_class_names if len(c) == 1 and c.isalnum()]
        filtered_names.sort() # Ensure strictly 0-9 then A-Z order which is standard for ImageFolder
        
        if len(filtered_names) == num_classes_model:
            print("Successfully filtered to standard alphanumeric classes (0-9, A-Z).")
            class_names = filtered_names
        else:
            print("Filter failed. Truncating list as fallback (Predictions may be wrong!).")
            class_names = raw_class_names[:num_classes_model]

    print(f"Final Class List: {class_names}")

    # 4. Load Model Architecture & Weights
    model = load_model_architecture(num_classes_model)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
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
        input_tensor = TRANSFORM(image).unsqueeze(0) # Add batch dimension (1, 3, 224, 224)
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