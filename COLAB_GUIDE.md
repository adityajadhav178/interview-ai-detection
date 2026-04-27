# Google Colab Training Guide (NO DRIVE UPLOAD)

## 📋 Quick Start

This script trains your audio models **directly in Google Colab** without needing Google Drive.

---

## 🚀 How to Use

### **Step 1: Open Google Colab**
- Go to [https://colab.research.google.com](https://colab.research.google.com)
- Create a new notebook

### **Step 2: Run Each Cell in Order**

#### **CELL 1:** Install Dependencies
```python
import subprocess
subprocess.run(['pip', 'install', '-q', 'scikit-learn', 'pandas', 'numpy', 
                'matplotlib', 'seaborn', 'xgboost', 'lightgbm', 'joblib'], 
               check=True)
print("✓ Dependencies installed")
```

---

#### **CELL 2:** Import Libraries & Configuration
```python
import pandas as pd
import numpy as np
import joblib
import warnings
import json
from io import StringIO
# ... (rest of imports - copy from the script)
```

---

#### **CELL 3:** Define Models & Hyperparameters
```python
def _base_models():
    # ... copy from colab_train_audio.py
```

---

#### **CELL 4:** Load Your Features Data
**MOST IMPORTANT STEP!**

**Option A: Paste CSV directly**

1. Export your `features_balanced.csv` to text format
2. Open it in a text editor
3. Copy ALL content (including headers)
4. Paste into the `csv_data` variable:

```python
csv_data = """patient_id,condition,label,feature_1,feature_2,feature_3,...
patient_001,depression,1,0.5,0.3,0.8...
patient_002,normal,0,0.2,0.1,0.3...
...
"""
```

**Option B: Upload file from computer**

Uncomment this code:
```python
from google.colab import files
print("📁 Select your features_balanced.csv file:")
uploaded = files.upload()
csv_filename = list(uploaded.keys())[0]
with open(csv_filename, 'r') as f:
    csv_data = f.read()
```

Then run the cell to load the data.

---

#### **CELL 5-7:** Training Cells
Just run them! They'll:
- ✓ Train 6 base models
- ✓ Perform hyperparameter tuning
- ✓ Build 2 ensemble models
- ✓ Show metrics table

---

#### **CELL 8:** Save Models
Prepares all models for download

---

#### **CELL 9:** Download Files
**Automatically downloads:**
- ✅ `best_audio_model.joblib` — Use this for predictions
- ✅ `scaler.joblib` — Scale new features before predicting
- ✅ `training_results.json` — Performance metrics
- ✅ All individual models as `.joblib` files

---

#### **CELL 10:** Visualizations
Creates and downloads:
- 📊 Model comparison chart
- 📊 Confusion matrix

---

## 📥 What You'll Get

After running all cells:

```
Downloaded Files:
  └─ best_audio_model.joblib      (400 KB - your best model)
  └─ scaler.joblib                (30 KB - scaling)
  └─ training_results.json        (1 KB - metrics)
  └─ model_comparison.png         (visualization)
  └─ confusion_matrix.png         (visualization)
  └─ random_forest.joblib         (alternative models)
  └─ xgboost.joblib
  └─ lightgbm.joblib
  └─ [and more...]
```

---

## 🔮 Using Your Trained Model

### Load the model locally:

```python
import joblib

# Load model and scaler
model = joblib.load('best_audio_model.joblib')
scaler = joblib.load('scaler.joblib')

# Use for predictions
new_features = [[feature1, feature2, feature3, ...]]  # 1D array of 93 features
scaled_features = scaler.transform(new_features)
prediction = model.predict(scaled_features)
probability = model.predict_proba(scaled_features)

print(f"Prediction: {prediction[0]}")
print(f"Confidence: {probability[0]}")
```

---

## 💡 Tips

### If your CSV is large:
- Use **Option B (upload file)** instead to avoid copy-paste issues
- Colab has ~50 GB storage, so any CSV will fit

### If you have data in Google Drive:
Uncomment this at the top:
```python
from google.colab import drive
drive.mount('/content/drive')
# Then adjust paths in CELL 4
```

### Common Issues:

| Issue | Solution |
|-------|----------|
| "pandas not found" | Run CELL 1 first |
| CSV paste error | Make sure you copied ALL rows including header |
| Memory error | Your CSV might be too large - run on GPU instead (Runtime → Change runtime type → GPU) |
| Model download fails | Just run CELL 9 again - it retries |

---

## 🎯 Next Steps

1. **Download** all files after training
2. **Test** best model with your own data
3. **Compare** models using `training_results.json`
4. **Retrain** with better hyperparameters if needed

---

## 📞 Questions?

The Colab script has comments for every step. Check them if something seems unclear!

---

**Happy training! 🚀**
