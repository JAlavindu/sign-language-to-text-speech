import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Configuration
KEYPOINTS_DIR = os.path.join("datasets", "processed_keypoints")
MAX_FRAMES = 50

# --- Feature Mapping Constants (Based on extraction script) ---
# Pose: 33*4 = 132
# Face: 468*3 = 1404
# LHand: 21*3 = 63
# RHand: 21*3 = 63
POSE_START, POSE_END = 0, 132
FACE_START, FACE_END = 132, 1536
HANDS_START, HANDS_END = 1536, 1662

# Feature counts for the split streams
PHYSICAL_FEATURES = (POSE_END - POSE_START) + (HANDS_END - HANDS_START) # 132 + 126 = 258
FACE_FEATURES = FACE_END - FACE_START # 1404

# Hyperparameters
PHYSICAL_HIDDEN_SIZE = 128
FACE_HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 16
EPOCHS = 30

class DynamicGestureDataset(Dataset):
    def __init__(self, data_dir, max_frames=MAX_FRAMES):
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.samples = []
        self.labels = []
        
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for file in os.listdir(class_dir):
                if file.endswith('.npy'):
                    self.samples.append(os.path.join(class_dir, file))
                    self.labels.append(class_name)
                    
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path = self.samples[idx]
        data = np.load(npy_path)
        
        frame_count = data.shape[0]
        # PADDING / TRUNCATING to 50 frames
        if frame_count < self.max_frames:
            padding = np.zeros((self.max_frames - frame_count, 1662))
            data = np.vstack((data, padding))
        else:
            data = data[:self.max_frames, :]
            
        # --- SPLIT THE DATA FOR THE DUAL STREAMS ---
        pose_data = data[:, POSE_START:POSE_END]
        face_data = data[:, FACE_START:FACE_END]
        hands_data = data[:, HANDS_START:HANDS_END]
        
        # Combine Pose and Hands into one "Physical/Motion" stream
        physical_stream = np.concatenate([pose_data, hands_data], axis=1)
        
        label = self.encoded_labels[idx]
        
        return torch.FloatTensor(physical_stream), torch.FloatTensor(face_data), torch.tensor(label, dtype=torch.long)

class DualStreamASLModel(nn.Module):
    def __init__(self, physical_input_size, face_input_size, physical_hidden, face_hidden, num_layers, num_classes):
        super(DualStreamASLModel, self).__init__()
        
        # Stream 1: Physical Motion (Body + Hands)
        self.physical_lstm = nn.LSTM(physical_input_size, physical_hidden, num_layers, batch_first=True, dropout=0.2)
        
        # Stream 2: Facial Expressions (Emotions/Grammar)
        self.face_lstm = nn.LSTM(face_input_size, face_hidden, num_layers, batch_first=True, dropout=0.2)
        
        # Fusion Layer
        # Concatenate the final hidden state of both LSTMs
        combined_hidden = physical_hidden + face_hidden
        
        self.fc1 = nn.Linear(combined_hidden, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, physical_x, face_x):
        # Stream 1 Forward
        phys_out, _ = self.physical_lstm(physical_x)
        phys_last = phys_out[:, -1, :] # Get last frame's output
        
        # Stream 2 Forward
        face_out, _ = self.face_lstm(face_x)
        face_last = face_out[:, -1, :] # Get last frame's output
        
        # Fusion
        combined = torch.cat((phys_last, face_last), dim=1)
        
        # Classification
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def train_model():
    print("Preparing dataset...")
    dataset = DynamicGestureDataset(KEYPOINTS_DIR)
    
    if len(dataset) == 0:
        print(f"No .npy files found in {KEYPOINTS_DIR}. Download WLASL data and extract keypoints first!")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualStreamASLModel(
        physical_input_size=PHYSICAL_FEATURES, 
        face_input_size=FACE_FEATURES, 
        physical_hidden=PHYSICAL_HIDDEN_SIZE, 
        face_hidden=FACE_HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        num_classes=dataset.num_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {device} for {dataset.num_classes} classes...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        model.train()
        for phys_data, face_data, batch_labels in dataloader:
            phys_data = phys_data.to(device)
            face_data = face_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass (now takes two inputs)
            outputs = model(phys_data, face_data)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {(correct/total)*100:.2f}%")

    # Save the model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/dual_stream_gesture_model.pth')
    print("Model saved to models/dual_stream_gesture_model.pth")

if __name__ == '__main__':
    train_model()