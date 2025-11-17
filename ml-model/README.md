# ASL Sign Language Recognition - ML Training Pipeline

## 📊 Your Datasets

You have two excellent ASL datasets ready to use:

### Dataset 1: SignAlphaSet

- **Location**: `e:\UNI sub\ICT\3rd yr\HCI\SignAlphaSet`
- **Classes**: 26 (A-Z)
- **Images per class**: ~1,000
- **Total images**: ~26,000
- **Format**: Clean, consistent hand gesture images

### Dataset 2: asl_dataset

- **Location**: `e:\UNI sub\ICT\3rd yr\HCI\asl_dataset`
- **Classes**: 36 (a-z + 0-9)
- **Images per class**: ~70
- **Total images**: ~2,500
- **Format**: Multiple angles (top, bottom, left, right, different hands)
- **Special**: Adds viewpoint diversity

### Combined Power

- **Total unique classes**: 36 (A-Z + 0-9)
- **Total images**: ~28,500
- **Coverage**: Excellent for training a robust model!

---

## 🎯 Training Strategy

We'll use **Transfer Learning** with **MobileNetV2**:

1. **Base Model**: MobileNetV2 (pre-trained on ImageNet)

   - Already knows general visual features (edges, shapes, textures)
   - We'll adapt it specifically for ASL hand shapes

2. **Two-Phase Training**:

   - **Phase 1** (30 epochs): Train only top layers, freeze base
   - **Phase 2** (20 epochs): Fine-tune last layers of base model

3. **Data Augmentation**: Realistic variations
   - Rotation (±20°)
   - Position shifts (20%)
   - Zoom (20%)
   - Brightness (±20%)
   - NO horizontal flip (signs are directional!)

---

## 📋 Training Scripts Overview

### 1. Data Exploration (`1_data_exploration.py`)

**What it does**:

- Counts images in each class
- Checks for corrupted images
- Visualizes sample images
- Generates distribution plots

**Run time**: 2-3 minutes

**Output**: Reports in `ml-model/reports/`

---

### 2. Dataset Preparation (`2_prepare_dataset.py`)

**What it does**:

- Merges both datasets
- Splits into train (70%), validation (15%), test (15%)
- Organizes into clean folder structure
- Creates metadata CSV files
- Generates class mapping JSON

**Run time**: 5-10 minutes

**Output**: Processed dataset in `ml-model/datasets/processed/`

---

### 3. Model Training (`3_train_model.py`)

**What it does**:

- Builds MobileNetV2-based model
- Trains with data augmentation
- Uses smart callbacks (early stopping, learning rate reduction)
- Saves best model automatically
- Logs everything to TensorBoard and CSV

**Run time**:

- GPU: 2-4 hours
- CPU: 8-16 hours

**Output**:

- Trained models in `ml-model/models/`
- Training logs in `ml-model/logs/`
- Accuracy/loss plots

---

## 🚀 Complete Training Workflow

### Prerequisites

```powershell
# 1. Navigate to project
cd "e:\UNI sub\ICT\3rd yr\HCI\sign-language-glove"

# 2. Create virtual environment (if not already created)
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r ml-model\requirements.txt
```

### The 3-Step Training Process

#### Step 1: Explore Data

```powershell
cd ml-model
python 1_data_exploration.py
```

**What to check**:

- Are all 36 classes present?
- Is distribution balanced?
- Any corrupted images?

**Look at**: `ml-model/reports/` for visualizations

---

#### Step 2: Prepare Dataset

```powershell
python 2_prepare_dataset.py
```

**What happens**:

- Both datasets merged intelligently
- Images copied to train/val/test folders
- ~19,000 training images
- ~4,200 validation images
- ~4,200 test images

**Verify**: Check `ml-model/datasets/processed/` has train/validation/test folders

---

#### Step 3: Train Model

```powershell
python 3_train_model.py
```

**Monitor progress**:

- Watch accuracy increase in terminal
- Open TensorBoard (see below)
- Check CSV logs

**Training phases**:

1. **Phase 1** (Epochs 1-30): Base frozen, train top layers
   - Expect: 70% → 90% accuracy
2. **Phase 2** (Epochs 31-50): Fine-tune base model
   - Expect: 90% → 96%+ accuracy

**Final output**:

- `asl_model_best.h5` - Best model (highest validation accuracy)
- `asl_model_final.h5` - Final model after all epochs
- Training history plots

---

## 📊 Monitor Training with TensorBoard

While training is running, open a **new terminal**:

```powershell
cd "e:\UNI sub\ICT\3rd yr\HCI\sign-language-glove\ml-model"
tensorboard --logdir logs
```

Then open browser to: **http://localhost:6006**

You'll see real-time:

- 📈 Training vs validation accuracy
- 📉 Training vs validation loss
- 🔢 Learning rate changes
- 🏗️ Model architecture

---

## 📈 Expected Training Progress

### Typical Accuracy Progression

```
Epoch 1:   ~45% accuracy (random guessing better than 2.7%)
Epoch 5:   ~70% accuracy (learning basic patterns)
Epoch 10:  ~85% accuracy (recognizing most signs)
Epoch 20:  ~92% accuracy (fine details improving)
Epoch 30:  ~94% accuracy (phase 1 complete)
Epoch 35:  ~95% accuracy (fine-tuning begins)
Epoch 45:  ~96% accuracy (near optimal)
Epoch 50:  ~97% accuracy (final result)
```

### Good Training Signs ✅

- Validation accuracy follows training accuracy (within 2-3%)
- Loss steadily decreases
- No sudden spikes or divergence
- Top-3 accuracy >98%

### Warning Signs ⚠️

- Large gap between train and val accuracy (overfitting)
- Validation accuracy stuck or decreasing (data issues)
- Loss increasing (learning rate too high)

---

## 🎯 Target Metrics

### Success Criteria

- ✅ **Test Accuracy**: >95%
- ✅ **Test Top-3 Accuracy**: >98%
- ✅ **All classes**: >90% individual accuracy
- ✅ **Model size**: <15MB
- ✅ **Inference time**: <100ms on mobile

### Real-World Performance

With your 28,500 images, expect:

- **Alphabet (A-Z)**: 96-98% accuracy
- **Numbers (0-9)**: 94-97% accuracy
- **Overall**: 95-97% accuracy

---

## 🔧 Hyperparameters Explained

These are set in `3_train_model.py`:

```python
IMG_SIZE = 224          # Input image size (MobileNetV2 standard)
BATCH_SIZE = 32         # Images per training step
EPOCHS = 50             # Total training epochs
LEARNING_RATE = 0.001   # Initial learning rate
```

### When to adjust:

**Reduce BATCH_SIZE** (32 → 16) if:

- Out of memory errors
- Computer freezing

**Reduce IMG_SIZE** (224 → 128) if:

- Training too slow
- Out of memory
- Trade-off: slightly lower accuracy

**Increase EPOCHS** (50 → 70) if:

- Accuracy still improving at epoch 50
- Want to squeeze out extra performance

**Adjust LEARNING_RATE** if:

- Too high (0.001 → 0.0001): Training unstable, accuracy jumps around
- Too low (0.001 → 0.01): Training too slow, stuck at low accuracy

---

## 💾 Output Files

After training, your `ml-model/` folder structure:

```
ml-model/
├── models/
│   ├── asl_model_best.h5           ← Use this for evaluation
│   ├── asl_model_final.h5          ← Final model
│   └── asl_model_saved/            ← For TFLite conversion
│
├── logs/
│   ├── asl_model_20251117_143022/  ← TensorBoard logs
│   ├── asl_model_20251117_143022.csv  ← Training metrics
│   └── asl_model_training_history.png ← Plots
│
├── datasets/
│   └── processed/
│       ├── train/          ← 19,000 images
│       ├── validation/     ← 4,200 images
│       ├── test/           ← 4,200 images
│       └── class_mapping.json
│
└── reports/
    ├── SignAlphaSet_samples.png
    ├── SignAlphaSet_distribution.png
    ├── asl_dataset_samples.png
    └── asl_dataset_distribution.png
```

---

## 🚨 Common Issues & Solutions

### Issue 1: "No module named tensorflow"

```powershell
pip install tensorflow==2.15.0
```

### Issue 2: Training stuck at low accuracy (~30-40%)

**Possible causes**:

- Data not loaded correctly
- Learning rate too low

**Solution**:

```powershell
# Re-run data preparation
python 2_prepare_dataset.py

# Check dataset folders exist
dir datasets\processed\train
```

### Issue 3: Out of memory error

**Solution**:
Edit `3_train_model.py`:

```python
BATCH_SIZE = 16  # Reduce from 32
IMG_SIZE = 128   # Reduce from 224
```

### Issue 4: Overfitting (train 98%, val 85%)

**Solution**:
Edit `3_train_model.py`:

```python
# Increase dropout in Dense layers
x = layers.Dropout(0.6)(x)  # Increase from 0.5

# Or reduce model complexity
x = layers.Dense(128, activation='relu')(x)  # Reduce from 256
```

### Issue 5: Training too slow (>1 min per epoch)

**Expected**:

- GPU: 10-30 seconds per epoch
- CPU: 2-5 minutes per epoch

**Solutions**:

- Reduce IMG_SIZE to 128
- Increase BATCH_SIZE to 64 (if RAM allows)
- Accept slower training (it's normal on CPU!)

---

## 📚 What Happens During Training

### Phase 1: Transfer Learning (Epochs 1-30)

```
MobileNetV2 Base (FROZEN) → Custom Layers (TRAINING)
```

- Pre-trained features stay locked
- Only top layers learn ASL-specific patterns
- Fast, prevents overfitting
- Gets to ~90-94% accuracy

### Phase 2: Fine-Tuning (Epochs 31-50)

```
MobileNetV2 Base (LAST 30 LAYERS UNFROZEN) → Custom Layers (TRAINING)
```

- Fine-tune base model for ASL
- Lower learning rate (more careful)
- Slower but reaches 95-97%

### Why this works:

- MobileNetV2 already knows edges, textures, shapes
- We just teach it "this edge pattern = letter A"
- Much faster than training from scratch
- Better results with less data

---

## ✨ After Training Succeeds

Once you see:

```
Test Accuracy: 96.23%
Test Top-3 Accuracy: 98.47%
✓ Final model saved: ml-model/models/asl_model_final.h5
```

**You're ready for next steps**:

1. **Evaluate in detail**: Run confusion matrix analysis
2. **Optimize for mobile**: Convert to TFLite (4MB)
3. **Test real-time**: Use webcam to test live recognition
4. **Deploy to app**: Integrate with smartphone app
5. **Connect to glove**: Add camera module to capture signs

---

## 🎓 Understanding Your Model

### Architecture Breakdown

```
Input Image (224x224x3)
        ↓
   Preprocessing (normalize to [-1, 1])
        ↓
   MobileNetV2 (1280 features)
        ↓
   Global Average Pooling
        ↓
   Dense(256) + ReLU + Dropout(0.5)
        ↓
   Dense(128) + ReLU + Dropout(0.3)
        ↓
   Dense(36) + Softmax
        ↓
   Output (probability for each class)
```

### Model Statistics

- **Total parameters**: ~3.5 million
- **Trainable parameters**: ~1.2 million (after phase 1)
- **Model size**: 14 MB (full), 4 MB (quantized)
- **Inference time**: 20-50ms (mobile), 10-20ms (GPU)

---

## 🏆 Success Checklist

Before moving to deployment:

- [ ] Training completed without errors
- [ ] Test accuracy >95%
- [ ] Validation accuracy within 2% of training accuracy
- [ ] Training history plots show convergence
- [ ] Model files saved successfully
- [ ] No class below 90% accuracy
- [ ] Confusion matrix mostly diagonal

**All checked? Congratulations! You have a production-ready ASL recognition model! 🎉**

---

## 🎯 Quick Command Reference

```powershell
# Setup (one time)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r ml-model\requirements.txt

# Training workflow (in order)
cd ml-model
python 1_data_exploration.py      # 2-3 min
python 2_prepare_dataset.py       # 5-10 min
python 3_train_model.py           # 2-4 hours (GPU) / 8-16 hours (CPU)

# Monitor (optional, while training)
tensorboard --logdir logs         # Open http://localhost:6006
```

---

## 📞 Need Help?

Check these in order:

1. Error message in terminal
2. Training logs in `ml-model/logs/`
3. TensorBoard visualizations
4. This README troubleshooting section

Most issues are:

- Wrong paths → Check dataset locations
- Missing packages → Run `pip install -r requirements.txt`
- Out of memory → Reduce batch size or image size
- Low accuracy → Check data quality and distribution

---

**Ready to train? Start with Step 1! 🚀**
