# 🎯 CLEAN PROJECT STRUCTURE - AUDIO PIPELINE

**Last Updated: April 12, 2026**

---

## 📂 Directory Structure (Clean & Organized)

```
interview-ai-detection/
│
├── 📊 DATA FOLDERS
│   └── data/
│       ├── raw/
│       │   ├── audio/                    ✓ Raw audio files (1,690 total)
│       │   │   ├── normal/               (1,092 files)
│       │   │   ├── depression/           (391 files)
│       │   │   ├── adhd/                 (41 files)
│       │   │   ├── ocd/                  (55 files)
│       │   │   └── anxiety/              (110 files - NEW! MP3→WAV)
│       │   │
│       │   └── audio_segmented/          [Will be created]
│       │       └── (segmented 25s audio)
│       │
│       ├── processed/
│       │   └── audio/                    [Intermediate processing]
│       │
│       └── final_features/
│           └── features.csv              [Final feature dataset]
│
├── 💻 CODE FOLDERS
│   └── src/audio/
│       ├── preprocessing/                [Audio preprocessing code]
│       │   ├── __init__.py               ✓ Clean interface
│       │   ├── segment_audio.py          ✓ Main segmentation logic
│       │   ├── audio_cleaner.py          (audio cleaning)
│       │   └── preprocess.py             (temp - can delete)
│       │
│       ├── feature_extraction/           [Feature extraction code]
│       │   ├── __init__.py               ✓ Clean interface
│       │   ├── extract_features.py       ✓ Feature extraction logic
│       │   ├── build_dataset.py          ✓ Dataset building
│       │
│       ├── training/                     [Model training code]
│       │   ├── __init__.py               ✓ Clean interface
│       │   ├── balance_data.py           ✓ SMOTE balancing
│       │   ├── train_models.py           ✓ Model training & eval
│       │   └── train.py                  (temp - can delete)
│       │
│       ├── interface/                    ✓ NEW! High-level interface
│       │   └── __init__.py               (AudioPreprocessor, etc)
│       │
│       └── inference/                    [Prediction code]
│           ├── __init__.py
│           └── predict.py
│
├── 📚 EXECUTION SCRIPTS (ROOT)
│   ├── run_audio_pipeline.py             ✓ Run full pipeline (1-3)
│   ├── step1_preprocess_audio.py         ✓ Step 1 only
│   ├── step2_extract_features.py         ✓ Step 2 only
│   ├── step3_train_models.py             ✓ Step 3 only
│   │
│   └── (Other scripts)
│       ├── setup_project_structure.py    ✓ Setup (already run)
│       ├── analyze_audio_sizes.py        (optional - analysis only)
│       └── ... (other non-essential)
│
├── 📋 DOCUMENTATION
│   ├── AUDIO_ANALYSIS_DECISION.md        ✓ Analysis & decisions
│   ├── FILE_REFERENCE_GUIDE.md           ✓ Complete reference
│   ├── GLOBAL_MODEL_PLAN_SUMMARY.md      ✓ Architecture overview
│   ├── PIPELINE_IMPLEMENTATION_PLAN.md   ✓ Detailed steps
│   ├── QUICK_START_PIPELINE.md           ✓ Command reference
│   └── README.md                         (main project readme)
│
├── 🤖 MODELS & EVALUATION (created during training)
│   ├── models/
│   │   └── audio_global/
│   │       ├── best_audio_model.joblib
│   │       ├── standard_scaler.joblib
│   │       └── label_mapping.json
│   │
│   └── evaluation/
│       └── audio_global/
│           ├── confusion_matrix.png
│           ├── classification_report.txt
│           └── metrics_summary.csv
│
└── ⚙️ CONFIG & UTILITIES
    ├── requirements.txt                  ✓ Python dependencies
    ├── configs/
    │   └── audio_config.yaml            (optional - config)
    └── .gitignore                       (ensure data/ excluded)
```

---

## 🔥 FILES THAT CAN BE DELETED (Not needed anymore)

```bash
# These files are temp duplicates or redundant:
rm src/audio/preprocessing/preprocess.py          # Duplicate
rm src/audio/preprocessing/extract_audio.py       # Unused
rm src/audio/training/train.py                    # Duplicate
rm analyze_audio_sizes.py                         # Temp analysis
rm prepare_data_for_training.py                   # Old format
rm combine_features.py                            # Not used
rm step2_extract_features.py                      # Old version
rm convert_video_to_audio.py                      # Not for audio
rm segment_audio_28sec.py                         # Old segment script
```

---

## ✨ KEY IMPROVEMENTS

✅ **Audio Data Complete**
- Normal: 1,092 files
- Depression: 391 files
- ADHD: 41 files
- OCD: 55 files
- Anxiety: 110 files ← **NEW (MP3→WAV converted)**
- **Total: 1,690 files**

✅ **Code Organized Properly**
- All preprocessing in `src/audio/preprocessing/`
- All feature extraction in `src/audio/feature_extraction/`
- All training in `src/audio/training/`
- Clean interface in `src/audio/interface/` for easy usage

✅ **Data Organized Properly**
- Raw audio: `data/raw/audio/`
- Segmented audio: `data/raw/audio_segmented/`
- Processed audio: `data/processed/audio/`
- Features: `data/final_features/features.csv`

✅ **Execution Clear**
- `step1_preprocess_audio.py` - Segment audio
- `step2_extract_features.py` - Extract features
- `step3_train_models.py` - Train models
- OR use `run_audio_pipeline.py` for complete run

---

## 🚀 HOW TO RUN (3 Options)

### Option 1: Run Complete Pipeline
```bash
python run_audio_pipeline.py
```

### Option 2: Run Step by Step
```bash
python step1_preprocess_audio.py    # 2-3 minutes
python step2_extract_features.py    # 5-10 minutes
python step3_train_models.py        # 10-15 minutes
```

### Option 3: Use Programmatically
```python
from src.audio.interface import run_full_pipeline

run_full_pipeline(
    raw_audio_dir="data/raw/audio",
    segmented_audio_dir="data/raw/audio_segmented",
    feature_csv="data/final_features/features.csv",
    model_dir="models/audio_global",
    eval_dir="evaluation/audio_global"
)
```

---

## 📊 EXPECTED OUTPUTS

After running all steps, you should have:

✓ **Segmented Audio** (Step 1)
```
data/raw/audio_segmented/
├── normal/           (~1,092 files, 10-25s each)
├── depression/       (~391 files, 10-25s each)
├── adhd/             (~57 segments from 41 files)
├── ocd/              (~55 files, 10-25s each)
└── anxiety/          (~110 files, 10-25s each)
Total: ~1,705 segments
```

✓ **Feature Dataset** (Step 2)
```
data/final_features/features.csv
├── 1,705 rows (samples)
├── 103 columns (patient_id + label + 100 features)
├── All numeric values
└── No NaN values
```

✓ **Trained Models** (Step 3)
```
models/audio_global/
├── best_audio_model.joblib        ← Use this for predictions!
├── standard_scaler.joblib         ← Required for preprocessing features
└── label_mapping.json             ← Class mappings

evaluation/audio_global/
├── confusion_matrix.png
├── classification_report.txt
└── metrics_summary.csv
```

---

## 🎯 NEXT STEPS AFTER TRAINING

1. **Make Predictions on New Audio**
   ```python
   from src.audio.interface import AudioModelPredictor
   
   prediction = AudioModelPredictor.predict(
       audio_file="path/to/new_audio.wav",
       model_dir="models/audio_global"
   )
   ```

2. **Deploy Model**
   - Copy `models/audio_global/` to production
   - Use `src/audio/inference/predict.py` for inference

3. **Optimize & Improve**
   - Try individual disease models if global model accuracy < 75%
   - Add more features if needed
   - Adjust SMOTE parameters for better balance

---

## 💡 IMPORTANT NOTES

**Keep in data/ folder:**
- `data/raw/audio/` - Required (raw audio)
- `data/raw/audio_segmented/` - Created (segmented audio)
- `data/final_features/features.csv` - Created (features)

**Keep in models/ folder:**
- `models/audio_global/best_audio_model.joblib`
- `models/audio_global/standard_scaler.joblib`
- These are essential for predictions!

**Don't delete:**
- `src/audio/` folder (all code)
- Any `__init__.py` files
- `requirements.txt`

---

## ✅ VERIFICATION CHECKLIST

Run these commands to verify structure:

```bash
# Check raw audio files
ls -1 data/raw/audio/*/*.wav | wc -l  # Should be ~1,690

# Check code files
ls -1 src/audio/preprocessing/*.py    # Should be 4-5 files
ls -1 src/audio/feature_extraction/*.py  # Should be 3-4 files
ls -1 src/audio/training/*.py         # Should be 3-4 files

# Ready to run?
python step1_preprocess_audio.py --help  # Should work
```

---

**Status: ✓ READY TO RUN PIPELINE**

All code is organized, all raw audio files are ready (including converted anxiety files), and all execution scripts are in place.

Next: Run `python step1_preprocess_audio.py` to begin!
