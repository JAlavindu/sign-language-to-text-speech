import os
import glob
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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
            # Columns: timestamp, flex1..5, accel_x..z, gyro_x..z, touch1..5, label
            # Features are columns 1 to 16 (0-indexed) -> indices 1 to 17 in python slice?
            # Let's select by name to be safe if possible, or index.
            # Assuming standard format from collector script:
            # timestamp, flex1, flex2, flex3, flex4, flex5, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, touch1, touch2, touch3, touch4, touch5, label
            
            features = df.iloc[:, 1:17].values # Columns 1 to 16
            all_data_list.append(features)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not all_data_list:
        print("No valid data found.")
        return None, None, None, None

    # Fit preprocessor
    print("Fitting preprocessor...")
    all_data_concat = np.vstack(all_data_list)
    preprocessor = SensorPreprocessor(window_size=WINDOW_SIZE, num_features=NUM_FEATURES)
    preprocessor.fit(all_data_concat)
    
    # Save preprocessor immediately
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    preprocessor.save(os.path.join(MODELS_DIR, "sensor_scaler.pkl"))
    print("Preprocessor saved.")

    # 2. Second pass: Create sequences per file
    print("Creating sequences...")
    X_sequences = []
    y_labels = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        features = df.iloc[:, 1:17].values
        
        # Get label from the file (assuming one label per file as per collector script)
        # The collector script adds a 'label' column.
        # We can take the mode of the label column or just the first one.
        label = df['label'].iloc[0]
        
        # Transform features
        features_scaled = preprocessor.transform(features)
        
        # Create sequences
        # We create a dummy label array for the sequence generator
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

def build_model(input_shape, num_classes):
    """
    Builds the CNN-LSTM model.
    """
    model = Sequential([
        # 1D CNN for spatial feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # LSTM for temporal dependency
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        # Dense layers for classification
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # 1. Load and Process Data
    X, y, preprocessor = load_and_process_data()
    
    if X is None:
        return

    # 2. Encode Labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save label encoder
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "sensor_labels.pkl"))
    print(f"Classes: {label_encoder.classes_}")

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # 4. Build Model
    model = build_model(input_shape=(WINDOW_SIZE, NUM_FEATURES), num_classes=len(label_encoder.classes_))
    model.summary()

    # 5. Train
    print("Starting training...")
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODELS_DIR, 'sensor_model_best.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5
        )
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    # 6. Evaluate
    print("\nEvaluating model...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    # Save final model
    model.save(os.path.join(MODELS_DIR, 'sensor_model_final.h5'))
    print("Training complete. Models saved to 'models/' directory.")

if __name__ == "__main__":
    main()
