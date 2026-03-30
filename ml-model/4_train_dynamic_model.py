import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Configuration
KEYPOINTS_DIR = os.path.join("datasets", "processed_keypoints")
MAX_FRAMES = 50       # Standardize all clips to 50 frames
INPUT_FEATURES = 1662 # 132(Pose) + 1404(Face) + 63(LH) + 63(RH)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 16
EPOCHS = 30

class DynamicGestureDataset(Dataset):
    def __init__(self, data_dir, max_frames=MAX_FRAMES):
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.samples = []
        self.labels = []
        
        # Assumption: You organize .npy files in subfolders by class name
        # Example: processed_keypoints/hello/video1.npy
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for file in os.listdir(class_dir):
                if file.endswith('.npy'):
                    self.samples.append(os.path.join(class_dir, file))
                    self.labels.append(class_name)
                    
        # Encode string labels to integers
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load the keypoints array (Shape: [Frames, 1662])
        npy_path = self.samples[idx]
        data = np.load(npy_path)
        
        # PADDING / TRUNCATING
        frame_count = data.shape[0]
        if frame_count < self.max_frames:
            # Pad with zeros if the video is too short
            padding = np.zeros((self.max_frames - frame_count, INPUT_FEATURES))
            data = np.vstack((data, padding))
        else:
            # Truncate if the video is too long
            data = data[:self.max_frames, :]
            
        label = self.encoded_labels[idx]
        
        # Returns Tensor, wait...
        return torch.FloatTensor(data), torch.tensor(label, dtype=torch.long)

class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GestureLSTM, self).__init__()
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, 1662)
        out, (h_n, c_n) = self.lstm(x)
        
        # We only care about the output of the LSTM at the final time step
        # out[:, -1, :] gets the last frame's output for the whole batch
        last_out = out[:, -1, :] 
        
        # Classify
        return self.fc(last_out)

def train_model():
    print("Preparing dataset...")
    dataset = DynamicGestureDataset(KEYPOINTS_DIR)
    
    if len(dataset) == 0:
        print(f"No .npy files found in {KEYPOINTS_DIR}. Make sure they are inside class subfolders!")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GestureLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, dataset.num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {device} for {dataset.num_classes} classes...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        model.train()
        for batch_data, batch_labels in dataloader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_data)
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
    torch.save(model.state_dict(), 'models/dynamic_gesture_model.pth')
    print("Model saved to models/dynamic_gesture_model.pth")

if __name__ == '__main__':
    train_model()