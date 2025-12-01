import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
import glob
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.sensor_preprocessor import SensorPreprocessor

# Configuration
REPORTS_DIR = "reports"
IMG_MODEL_PATH = os.path.join("models", "asl_model_final.h5")
SENS_MODEL_PATH = os.path.join("models", "sensor_model_final.h5")
SENS_DATA_DIR = os.path.join("datasets", "sensor_data", "raw")
IMG_TEST_DIR = os.path.join("datasets", "processed", "test")

def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    """Generates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, filename))
    plt.close()
    print(f"Saved {filename}")

def evaluate_image_model():
    print("\n" + "="*50)
    print("Evaluating Image Model...")
    print("="*50)
    
    if not os.path.exists(IMG_MODEL_PATH):
        print("Image model not found. Skipping.")
        return

    # Load Model
    model = tf.keras.models.load_model(IMG_MODEL_PATH)
    
    # Data Generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    if not os.path.exists(IMG_TEST_DIR):
        print(f"Test directory {IMG_TEST_DIR} not found. Skipping.")
        return

    test_generator = test_datagen.flow_from_directory(
        IMG_TEST_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Predict
    print("Running predictions on test set...")
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    # Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, class_labels, "Image Model Confusion Matrix", "image_confusion_matrix.png")
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(REPORTS_DIR, "image_classification_report.csv"))
    print("Saved image_classification_report.csv")

def evaluate_sensor_model():
    print("\n" + "="*50)
    print("Evaluating Sensor Model...")
    print("="*50)
    
    if not os.path.exists(SENS_MODEL_PATH):
        print("Sensor model not found. Skipping.")
        return

    # Load Model & Artifacts
    model = tf.keras.models.load_model(SENS_MODEL_PATH)
    label_encoder = joblib.load(os.path.join("models", "sensor_labels.pkl"))
    
    # Load Data (Replicating logic from 9_train_sensor_model.py)
    csv_files = glob.glob(os.path.join(SENS_DATA_DIR, "*.csv"))
    if not csv_files:
        print("No sensor data found. Skipping.")
        return

    # Load and Preprocess
    # We need to fit the preprocessor on ALL data first to match training
    all_data_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            features = df.iloc[:, 1:17].values
            all_data_list.append(features)
        except: pass
        
    if not all_data_list: return

    preprocessor = SensorPreprocessor(window_size=50, num_features=16)
    preprocessor.fit(np.vstack(all_data_list))
    
    # Create Sequences
    X_sequences = []
    y_labels = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        features = df.iloc[:, 1:17].values
        label = df['label'].iloc[0]
        
        features_scaled = preprocessor.transform(features)
        labels_array = np.array([label] * len(features))
        seq_X, seq_y = preprocessor.create_sequences(features_scaled, labels_array)
        
        if len(seq_X) > 0:
            X_sequences.append(seq_X)
            y_labels.append(seq_y)
            
    if not X_sequences: return

    X = np.concatenate(X_sequences)
    y = np.concatenate(y_labels)
    
    # Encode Labels
    y_encoded = label_encoder.transform(y)
    
    # Split (Same random_state as training to get the same test set)
    _, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Predict
    print("Running predictions on sensor test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_, "Sensor Model Confusion Matrix", "sensor_confusion_matrix.png")
    
    # Classification Report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(REPORTS_DIR, "sensor_classification_report.csv"))
    print("Saved sensor_classification_report.csv")

if __name__ == "__main__":
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        
    evaluate_image_model()
    evaluate_sensor_model()
    print("\nDone! Check the 'reports' folder.")
