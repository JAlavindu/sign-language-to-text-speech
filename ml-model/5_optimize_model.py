import tensorflow as tf
import os
import numpy as np

# Configuration
MODEL_DIR = "models"
IMG_MODEL_NAME = "asl_model_final.h5"
SENS_MODEL_NAME = "sensor_model_final.h5"

def convert_to_tflite(model_path, output_path, quantize=False):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Converting {model_path}...")
    
    # Load Keras model
    model = tf.keras.models.load_model(model_path)
    
    # Convert
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("Applying default optimization (quantization)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"✓ Saved to {output_path}")
    
    # Compare sizes
    orig_size = os.path.getsize(model_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Original: {orig_size:.2f} MB")
    print(f"  TFLite:   {new_size:.2f} MB")
    print(f"  Reduction: {(1 - new_size/orig_size)*100:.1f}%")

def main():
    # 1. Convert Image Model
    img_path = os.path.join(MODEL_DIR, IMG_MODEL_NAME)
    img_out = os.path.join(MODEL_DIR, "asl_model.tflite")
    convert_to_tflite(img_path, img_out, quantize=True)
    
    # 2. Convert Sensor Model
    sens_path = os.path.join(MODEL_DIR, SENS_MODEL_NAME)
    sens_out = os.path.join(MODEL_DIR, "sensor_model.tflite")
    convert_to_tflite(sens_path, sens_out, quantize=False) # Sensors are already small

if __name__ == "__main__":
    main()
