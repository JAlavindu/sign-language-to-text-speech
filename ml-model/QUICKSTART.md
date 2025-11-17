# Quick Start Guide - ML Model Training

## 🚀 Get Started in 5 Steps

### Step 1: Activate Virtual Environment

```powershell
cd "e:\UNI sub\ICT\3rd yr\HCI\sign-language-glove\ml-model"
..\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Explore Your Data

```powershell
python 1_data_exploration.py
```

**Expected output**: Reports in `ml-model/reports/` folder

- Check class distributions
- View sample images
- Identify any issues

### Step 4: Prepare Dataset

```powershell
python 2_prepare_dataset.py
```

**Expected output**: Organized dataset in `ml-model/datasets/processed/`

- train/ folder (~70% of data)
- validation/ folder (~15% of data)
- test/ folder (~15% of data)
- metadata CSV files

**⏱ Estimated time**: 5-10 minutes

### Step 5: Train the Model

```powershell
python 3_train_model.py
```

**Expected output**: Trained model in `ml-model/models/`

- asl_model_best.h5 (best checkpoint)
- asl_model_final.h5 (final model)
- Training logs in `ml-model/logs/`

**⏱ Estimated time**:

- With GPU: 2-4 hours
- CPU only: 8-16 hours

---

## 📊 Monitor Training Progress

### Watch Training in Real-Time with TensorBoard

```powershell
tensorboard --logdir ml-model/logs
```

Then open browser to: http://localhost:6006

You'll see:

- Accuracy curves (training vs validation)
- Loss curves
- Learning rate changes
- Model graph

### Check CSV Logs

Training metrics are also saved to CSV files in `ml-model/logs/` folder.

---

## 🎯 What to Expect

### Training Progress Timeline

**Epochs 1-10** (Initial Learning)

```
Epoch 1/50
accuracy: 0.45 - val_accuracy: 0.52
Epoch 5/50
accuracy: 0.75 - val_accuracy: 0.78
Epoch 10/50
accuracy: 0.85 - val_accuracy: 0.87
```

**Epochs 11-30** (Steady Improvement)

```
Epoch 20/50
accuracy: 0.92 - val_accuracy: 0.91
Epoch 30/50
accuracy: 0.95 - val_accuracy: 0.93
```

**Epochs 31-50** (Fine-Tuning)

```
Epoch 40/50
accuracy: 0.97 - val_accuracy: 0.96
Epoch 50/50
accuracy: 0.98 - val_accuracy: 0.97
```

### Final Target Metrics

✅ **Test Accuracy**: >95%
✅ **Test Top-3 Accuracy**: >98%
✅ **Training Time**: 2-4 hours (GPU) / 8-16 hours (CPU)
✅ **Model Size**: ~14MB

---

## 🔍 Troubleshooting

### Problem: "No module named tensorflow"

**Solution**:

```powershell
pip install tensorflow==2.15.0
```

### Problem: Training is very slow

**Reasons**:

- No GPU (expected, CPU training takes longer)
- Batch size too small

**Solutions**:

- Reduce image size (edit 3_train_model.py: IMG_SIZE = 128)
- Increase batch size if you have enough RAM
- Be patient - CPU training works, just takes longer!

### Problem: Validation accuracy stuck at 70-80%

**Reasons**:

- Learning rate too high/low
- Not enough data augmentation
- Data quality issues

**Solutions**:

1. Check data distribution (run 1_data_exploration.py again)
2. Adjust learning rate in 3_train_model.py (try 0.0001 or 0.01)
3. Let training continue - sometimes it improves suddenly

### Problem: Overfitting (train acc >> val acc)

**Example**: Train: 98%, Val: 85%

**Solutions**:

- Edit 3_train_model.py:
  - Increase dropout: 0.5 → 0.6
  - Add more data augmentation
  - Reduce epochs if it starts overfitting early

### Problem: Out of memory error

**Solutions**:

```python
# Edit 3_train_model.py
BATCH_SIZE = 16  # Reduce from 32
IMG_SIZE = 128   # Reduce from 224
```

---

## 📁 Output Files Explained

After training completes, you'll have:

```
ml-model/
├── models/
│   ├── asl_model_best.h5          # Best model (highest val accuracy)
│   ├── asl_model_final.h5         # Final model after all epochs
│   └── asl_model_saved/           # SavedModel format (for TFLite)
│
├── logs/
│   ├── asl_model_TIMESTAMP/       # TensorBoard logs
│   ├── asl_model_TIMESTAMP.csv    # Training metrics CSV
│   └── asl_model_training_history.png  # Accuracy/Loss plots
│
├── datasets/
│   └── processed/
│       ├── train/
│       ├── validation/
│       ├── test/
│       └── class_mapping.json     # Class names and indices
│
└── reports/
    ├── SignAlphaSet_samples.png
    ├── SignAlphaSet_distribution.png
    ├── asl_dataset_samples.png
    └── asl_dataset_distribution.png
```

---

## ✨ Next Steps After Training

### 1. Evaluate Model Performance

```powershell
python 4_evaluate_model.py
```

Generates:

- Confusion matrix
- Per-class accuracy
- Misclassification analysis

### 2. Optimize for Mobile

```powershell
python 5_optimize_model.py
```

Creates:

- TFLite model (14MB)
- Quantized TFLite model (4MB)
- ESP32-optimized model (3MB)

### 3. Test Real-Time

```powershell
python 6_test_realtime.py
```

Test with:

- Webcam
- Single images
- Batch predictions

---

## 💡 Tips for Best Results

### Data Quality Matters

✅ Clean, well-lit images
✅ Consistent hand positions
✅ Diverse backgrounds
✅ Multiple people/hand sizes

### Training Best Practices

✅ Monitor validation accuracy (not just training!)
✅ Stop early if overfitting
✅ Save checkpoints regularly
✅ Use TensorBoard to visualize

### Hardware Recommendations

- **Minimum**: 8GB RAM, CPU training
- **Recommended**: 16GB RAM, GPU (NVIDIA GTX 1050+)
- **Optimal**: 32GB RAM, GPU (NVIDIA RTX 3060+)

---

## 📞 Need Help?

Check:

1. Training logs in `ml-model/logs/`
2. TensorBoard visualizations
3. Error messages in terminal
4. This README's troubleshooting section

Common issues are usually:

- Wrong file paths
- Missing dependencies
- Insufficient RAM
- Data format problems

Most errors are solved by:

- Re-running with smaller batch size
- Checking dataset paths
- Reinstalling dependencies

---

## 🎓 Understanding Your Model

### What the model learns:

**Phase 1 (Epochs 1-30)**: Frozen Base

- Learns to classify ASL signs using pre-trained features
- Fast training, prevents overfitting
- Gets you to ~90% accuracy quickly

**Phase 2 (Epochs 31-50)**: Fine-Tuning

- Adapts pre-trained features specifically for ASL
- Slower, more careful adjustments
- Pushes accuracy to 95-97%

### Why MobileNetV2?

- Small size (good for phones/embedded devices)
- Fast inference (~50ms on phone)
- Pre-trained on millions of images
- Perfect balance of speed and accuracy

---

## 🏁 Success Checklist

Before deploying your model:

- [ ] Test accuracy >95%
- [ ] No class below 90%
- [ ] Confusion matrix mostly diagonal
- [ ] Real-time testing successful
- [ ] Model size acceptable (<15MB)
- [ ] Inference speed acceptable (<100ms)

**If all checked - you're ready to deploy! 🎉**
