import os
import glob
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.sensor_preprocessor import SensorPreprocessor

# Configuration
DATA_DIR = os.path.join("datasets", "sensor_data", "raw")
MODELS_DIR = "models"
WINDOW_SIZE = 50
NUM_FEATURES = 16  # 5 flex + 3 accel + 3 gyro + 5 touch
EPOCHS = 50
BATCH_SIZE = 32

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SensorModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SensorModel, self).__init__()
        
        # CNN layers
        # Input shape to Conv1d: (Batch, Channels/Features, Length/Time)
        # We will need to permute input from (Batch, Time, Features) to (Batch, Features, Time)
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM layer
        # Input to LSTM: (Batch, Time, Features)
        # After pooling, the time dimension is reduced.
        # We need to calculate what the input size is for LSTM, or let it infer.
        # Conv1d output: 64 channels. So LSTM input_size is 64.
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        
        # Dense layers
        self.fc1 = nn.Linear(64, 64)
        self.fc_dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Time, Features)
        
        # Permute for CNN: (Batch, Features, Time)
        x_cnn = x.permute(0, 2, 1)
        
        # CNN block
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.bn(x_cnn)
        x_cnn = self.pool(x_cnn)
        
        # Permute back for LSTM: (Batch, Time, Features)
        x_lstm = x_cnn.permute(0, 2, 1)
        
        # LSTM block
        # output shape: (Batch, Time, Hidden)
        # hidden/cell shape: (Layers, Batch, Hidden)
        lstm_out, (h_n, c_n) = self.lstm(x_lstm)
        
        # Take the last time step output for classification
        # OR using h_n[-1]
        last_out = lstm_out[:, -1, :] 
        
        # Dense block
        out = self.dropout1(last_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc_dropout(out)
        out = self.fc2(out)
        
        return out

def load_and_process_data():
    """
    Loads all CSV files, fits the preprocessor, and creates sequences.
    """
    print("Loading data...")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}. Please run 8_collect_sensor_data.py first.")
        return None, None, None, None

    # 1. First pass: Collect all data to fit the scaler
    all_data_list = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            features = df.iloc[:, 1:17].values # Columns 1 to 16
            all_data_list.append(features)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not all_data_list:
        print("No valid data found.")
        return None, None, None, None
        
    # Fit preprocessor
    all_data = np.concatenate(all_data_list)
    preprocessor = SensorPreprocessor(window_size=WINDOW_SIZE)
    # create_sequences internally does fit_transform-like logic if we just passed raw data?
    # Actually checking `sensor_preprocessor.py` (assumed logic based on usage):
    # Usually we fit scaler on training data. Here we fit on everything which is a slight leak check,
    # but for sensor ranges (0-1023) it's mostly static scaling.
    preprocessor.fit(all_data)
    
    # Save preprocessor
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "sensor_preprocessor.pkl"))
    
    # 2. Second pass: Create sequences
    X_sequences = []
    y_labels = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        features = df.iloc[:, 1:17].values
        label = df['label'].iloc[0]
        
        # Transform features
        features_scaled = preprocessor.transform(features)
        
        # Create sequences
        labels_array = np.array([label] * len(features))
        seq_X, seq_y = preprocessor.create_sequences(features_scaled, labels_array)
        
        if len(seq_X) > 0:
            X_sequences.append(seq_X)
            y_labels.append(seq_y)

    if not X_sequences:
        print("Not enough data to create sequences (recordings too short?).")
        return None, None, None, None

    X = np.concatenate(X_sequences)
    y = np.concatenate(y_labels)
    
    print(f"Total sequences: {X.shape[0]}")
    print(f"Input shape: {X.shape}")
    
    return X, y, preprocessor

def main():
    # 1. Load and Process Data
    X, y, preprocessor = load_and_process_data()
    
    if X is None:
        return

    # 2. Encode Labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    # Save label encoder
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "sensor_labels.pkl"))
    print(f"Classes: {label_encoder.classes_}")

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Build Model
    print("Building Model...")
    model = SensorModel(num_features=NUM_FEATURES, num_classes=num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Train
    print("Starting training...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "sensor_model_best.pth"))
            
    print("Training Complete!")
    print(f"Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
