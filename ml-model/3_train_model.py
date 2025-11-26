"""
Main Training Script for ASL Recognition Model
Uses Transfer Learning with MobileNetV2
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
# Get paths from env or use defaults
DATASET_PATH = os.getenv("PROCESSED_DATASET_PATH", r"e:\UNI sub\ICT\3rd yr\HCI\sign-language-glove\ml-model\datasets\processed")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", r"e:\UNI sub\ICT\3rd yr\HCI\sign-language-glove\ml-model\models")
LOGS_PATH = os.getenv("LOGS_PATH", r"e:\UNI sub\ICT\3rd yr\HCI\sign-language-glove\ml-model\logs")

# Resolve relative paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.isabs(DATASET_PATH):
    DATASET_PATH = os.path.join(project_root, DATASET_PATH)
if not os.path.isabs(MODEL_SAVE_PATH):
    MODEL_SAVE_PATH = os.path.join(project_root, MODEL_SAVE_PATH)
if not os.path.isabs(LOGS_PATH):
    LOGS_PATH = os.path.join(project_root, LOGS_PATH)

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
FINE_TUNE_AT = 100  # Unfreeze last 30 layers after initial training

# Create directories
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

def load_class_mapping():
    """Load class names and mapping"""
    with open(os.path.join(DATASET_PATH, 'class_mapping.json'), 'r') as f:
        mapping = json.load(f)
    return mapping

def create_data_generators():
    """Create data generators with augmentation"""
    print(f"\n{'='*60}")
    print("Creating Data Generators")
    print(f"{'='*60}")
    
    # Training data augmentation (realistic for hand gestures)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,           # ±20° rotation
        width_shift_range=0.2,       # 20% horizontal shift
        height_shift_range=0.2,      # 20% vertical shift
        shear_range=0.15,            # Perspective distortion
        zoom_range=0.2,              # Zoom in/out
        horizontal_flip=False,        # DON'T flip (signs are directional!)
        brightness_range=[0.8, 1.2], # Lighting variation
        fill_mode='nearest'
    )
    
    # Validation and test data (no augmentation, only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load datasets
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'validation'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {val_generator.samples}")
    print(f"✓ Test samples: {test_generator.samples}")
    print(f"✓ Number of classes: {train_generator.num_classes}")
    print(f"✓ Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"✓ Batch size: {BATCH_SIZE}")
    
    return train_generator, val_generator, test_generator

def build_model(num_classes):
    """Build transfer learning model with MobileNetV2"""
    print(f"\n{'='*60}")
    print("Building Model Architecture")
    print(f"{'='*60}")
    
    # Load pre-trained MobileNetV2 (without top classification layer)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    print(f"\nBase model: MobileNetV2")
    print(f"✓ Pre-trained on ImageNet")
    print(f"✓ Total layers: {len(base_model.layers)}")
    print(f"✓ Trainable: {base_model.trainable}")
    
    # Build custom top layers
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Preprocessing
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = keras.Model(inputs, outputs)
    
    print(f"\nModel architecture:")
    print(f"  Input: {IMG_SIZE}x{IMG_SIZE}x3")
    print(f"  → MobileNetV2 (frozen)")
    print(f"  → GlobalAveragePooling2D")
    print(f"  → Dense(256) + Dropout(0.5)")
    print(f"  → Dense(128) + Dropout(0.3)")
    print(f"  → Dense({num_classes}) [softmax]")
    
    return model, base_model

def compile_model(model, learning_rate):
    """Compile model with optimizer and loss"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    print(f"\n✓ Model compiled")
    print(f"  Optimizer: Adam (lr={learning_rate})")
    print(f"  Loss: Categorical Crossentropy")
    print(f"  Metrics: Accuracy, Top-3 Accuracy")

def create_callbacks(model_name):
    """Create training callbacks"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_PATH, f'{model_name}_best.h5'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join(LOGS_PATH, f'{model_name}_{timestamp}'),
            histogram_freq=1,
            write_graph=True
        ),
        
        # CSV logging
        CSVLogger(
            filename=os.path.join(LOGS_PATH, f'{model_name}_{timestamp}.csv'),
            append=True
        )
    ]
    
    print(f"\n✓ Callbacks configured:")
    print(f"  • ModelCheckpoint (save best model)")
    print(f"  • EarlyStopping (patience=10)")
    print(f"  • ReduceLROnPlateau (factor=0.5, patience=5)")
    print(f"  • TensorBoard logging")
    print(f"  • CSV logging")
    
    return callbacks

def plot_training_history(history, model_name):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(LOGS_PATH, f'{model_name}_training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training history plot saved: {plot_path}")
    plt.close()

def train_phase1(model, train_gen, val_gen, callbacks, initial_epochs):
    """Phase 1: Train only top layers (base frozen)"""
    print(f"\n{'='*60}")
    print("PHASE 1: Training Top Layers (Base Frozen)")
    print(f"{'='*60}")
    print(f"Epochs: {initial_epochs}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    history = model.fit(
        train_gen,
        epochs=initial_epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def train_phase2(model, base_model, train_gen, val_gen, callbacks, total_epochs, initial_epochs):
    """Phase 2: Fine-tune with unfrozen layers"""
    print(f"\n{'='*60}")
    print("PHASE 2: Fine-Tuning (Unfreezing Layers)")
    print(f"{'='*60}")
    
    # Unfreeze base model layers
    base_model.trainable = True
    
    # Freeze early layers, unfreeze last layers
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    
    trainable_layers = sum([layer.trainable for layer in base_model.layers])
    print(f"\nUnfrozen {trainable_layers} layers (from layer {FINE_TUNE_AT})")
    
    # Re-compile with lower learning rate
    fine_tune_lr = LEARNING_RATE / 10
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    print(f"New learning rate: {fine_tune_lr}")
    print(f"Fine-tuning for {total_epochs - initial_epochs} more epochs")
    
    history_fine = model.fit(
        train_gen,
        epochs=total_epochs,
        initial_epoch=initial_epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return history_fine

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("ASL SIGN LANGUAGE RECOGNITION - MODEL TRAINING")
    print("="*60)
    
    # GPU check
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"\n✓ GPU available: {physical_devices}")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print(f"\n⚠ No GPU detected, using CPU (training will be slower)")
    
    # Load class mapping
    class_mapping = load_class_mapping()
    num_classes = class_mapping['num_classes']
    print(f"\n✓ Loaded class mapping: {num_classes} classes")
    
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Build model
    model, base_model = build_model(num_classes)
    
    # Compile model
    compile_model(model, LEARNING_RATE)
    
    # Print model summary
    print(f"\n{'='*60}")
    print("Model Summary")
    print(f"{'='*60}")
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks('asl_model')
    
    # Calculate epochs for each phase
    initial_epochs = int(EPOCHS * 0.6)  # 60% for phase 1
    
    # Phase 1: Train top layers
    history_phase1 = train_phase1(model, train_gen, val_gen, callbacks, initial_epochs)
    
    # Phase 2: Fine-tune
    history_phase2 = train_phase2(
        model, base_model, train_gen, val_gen, callbacks, EPOCHS, initial_epochs
    )
    
    # Combine histories
    combined_history = {
        'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
        'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
        'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
        'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
    }
    
    # Create history object for plotting
    class HistoryObject:
        def __init__(self, history_dict):
            self.history = history_dict
    
    combined_hist = HistoryObject(combined_history)
    
    # Plot training history
    plot_training_history(combined_hist, 'asl_model')
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Evaluating on Test Set")
    print(f"{'='*60}")
    
    test_loss, test_acc, test_top3 = model.evaluate(test_gen, verbose=1)
    print(f"\n✓ Test Results:")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Test Top-3 Accuracy: {test_top3*100:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(MODEL_SAVE_PATH, 'asl_model_final.h5')
    model.save(final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")
    
    # Save model in SavedModel format (for TFLite conversion)
    saved_model_path = os.path.join(MODEL_SAVE_PATH, 'asl_model_saved')
    model.save(saved_model_path)
    print(f"✓ SavedModel format saved: {saved_model_path}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Run 4_evaluate_model.py for detailed evaluation")
    print(f"  2. Run 5_optimize_model.py to create TFLite versions")
    print(f"  3. Run 6_test_realtime.py for real-time testing")
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
