# Audio Processing Pipeline Steps

All pipeline step implementations organized in this dedicated folder.

**Important:** All step files are ONLY in this folder. Run from here.

## 📂 Structure

```
pipeline_steps/
├── __init__.py                      Module interface
├── README.md                        This documentation
├── step1_preprocess_audio.py        [1] Audio segmentation & normalization
├── step2_extract_features.py        [2] Acoustic feature extraction
└── step3_train_models.py            [3] Model training & evaluation
```

## 🚀 How to Run

### Option 1: From This Folder (Recommended)
```bash
cd pipeline_steps

# Run individual steps
python step1_preprocess_audio.py
python step2_extract_features.py
python step3_train_models.py
```

### Option 2: From Python Code
```python
# Import from pipeline_steps module
from pipeline_steps import (
    run_step1_preprocess,
    run_step2_extract_features,
    run_step3_train_models
)

# Run steps
run_step1_preprocess()
run_step2_extract_features()
run_step3_train_models()
```

### Option 3: Run All Steps Sequentially
```python
from pipeline_steps import (
    run_step1_preprocess,
    run_step2_extract_features,
    run_step3_train_models
)

# Run sequentially
if run_step1_preprocess():
    if run_step2_extract_features():
        run_step3_train_models()
```

## 📊 What Each Step Does

### Step 1: Preprocess Audio (2-3 minutes)
**File:** `step1_preprocess_audio.py`

**Input:** `data/raw/audio/{condition}/*.wav` (1,689 files, variable length)

**Process:**
- Pad short audio (< 10s) with silence to 10s
- Keep medium audio (10-25s) as-is
- Segment long audio (> 25s) with 50% overlap

**Output:** `data/raw/audio_segmented/{condition}/*.wav` (~1,705 normalized audio files)

---

### Step 2: Extract Features (5-10 minutes)
**File:** `step2_extract_features.py`

**Input:** `data/raw/audio_segmented/` (normalized audio files)

**Process:**
- Extract 20+ acoustic features:
  - Prosody (pitch, contours, vibrato)
  - Voice quality (jitter, shimmer, HNR)
  - Temporal (speaking rate, voice activity)
  - Spectral (MFCC, centroids)
  - Energy (intensity, loudness)
- Aggregate to 100 features per patient

**Output:** `data/final_features/features.csv` (1,705 rows × 103 columns)

---

### Step 3: Train Models (10-15 minutes)
**File:** `step3_train_models.py`

**Input:** `data/final_features/features.csv`

**Process:**
- Load features
- Apply SMOTE for class balancing
- Split: Train 70% / Val 15% / Test 15%
- Train 4 models:
  - Random Forest
  - XGBoost
  - SVM
  - LightGBM
- 5-Fold Stratified Cross-Validation
- Evaluate on test set

**Output:**
- Models: `models/audio_global/` (best_audio_model.joblib + scaler)
- Evaluation: `evaluation/audio_global/` (plots, metrics, report)

## ⚡ Quick Reference

```bash
# Navigate to pipeline folder
cd pipeline_steps

# Run individual steps
python step1_preprocess_audio.py
python step2_extract_features.py
python step3_train_models.py

# Check progress
ls -la data/raw/audio_segmented/  # Step 1 output
ls -la data/final_features/        # Step 2 output
ls -la models/audio_global/        # Step 3 output
```

## 🔄 Dependency Chain

```
Raw Audio
    ↓
Step 1 (Preprocessing) → data/raw/audio_segmented/
    ↓
Step 2 (Feature Extraction) → data/final_features/features.csv
    ↓
Step 3 (Model Training) → models/audio_global/
    ↓
Ready for Predictions!
```

**Important:** Must run steps in order (1 → 2 → 3)

## 🛠️ Troubleshooting

**"ModuleNotFoundError"?**
- Make sure you're in pipeline_steps folder: `cd pipeline_steps`
- All step files are ONLY in this folder

**"File not found"?**
- Check previous step completed successfully
- Verify input files exist

**Step 1 slow?**
- Expected: ~1,700 audio files takes 2-3 minutes
- Normal processing speed

**Step 2 stuck?**
- Expected: Feature extraction takes time
- Number of features × number of files = processing time

**Step 3 long?**
- Expected: Training 4 models with 5-Fold CV takes 10-15 minutes
- Model complexity and data size determine time

## 📝 Environment Setup

Ensure virtual environment is active:
```bash
# Windows
.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## ✅ Success Indicators

**Step 1 Success:**
- `data/raw/audio_segmented/` folder has files
- Console shows: "✓ Audio preprocessing complete!"

**Step 2 Success:**
- `data/final_features/features.csv` exists
- CSV has 1,700+ rows and 100+ columns
- Console shows: "✓ Feature extraction complete!"

**Step 3 Success:**
- `models/audio_global/best_audio_model.joblib` exists
- Evaluation plots created in `evaluation/audio_global/`
- Console shows: "✓ Model training complete!"

---

**All scripts ready to run. Start with Step 1!** 🚀
