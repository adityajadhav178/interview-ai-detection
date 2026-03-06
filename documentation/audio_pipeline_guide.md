# Audio Mental Health Detection Pipeline Guide

> **Last Updated:** March 2026  
> **Scope:** Full 10-step classical ML pipeline for mental health detection from interview audio.

---

## Pipeline Architecture

```
Raw Videos → Audio Extraction → Preprocessing → Feature Extraction (20 features)
                                                        ↓
                              Aggregation (100 features per patient)
                                                        ↓
                       SMOTE Balancing → Model Training (RF / XGB / SVM / LightGBM)
                                                        ↓
                                        Saved Model → Inference / Predict
```

---

## Data Folder Structure

Organize all data by condition. **The folder name IS the label — no CSV required.**

```
data/raw/
├── video/
│   ├── depression/   ← put .mp4 videos here
│   ├── anxiety/
│   ├── ocd/
│   ├── adhd/
│   └── normal/       ← healthy control subjects (no condition)
└── audio/
    ├── depression/   ← auto-filled by extract_audio.py
    ├── anxiety/
    ├── ocd/
    ├── adhd/
    └── normal/
```

Each condition folder contains **patient subfolders**, each with up to 5 answer recordings:
```
data/raw/audio/depression/
    patient_001/
        answer_1.wav
        answer_2.wav
        ...
    patient_002/
        ...
```

**Label mapping (auto-derived from folder name):**
| Folder | Label |
|--------|-------|
| normal | 0 |
| depression | 1 |
| anxiety | 2 |
| adhd | 3 |
| ocd | 4 |

---

## Step-by-Step Guide

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Extract Audio from Videos

Converts `.mp4` videos → mono, 16kHz `.wav` files, preserving the folder structure.

```bash
python src/audio/preprocessing/extract_audio.py \
    --input_dir data/raw/video \
    --output_dir data/raw/audio
```

**Script:** `src/audio/preprocessing/extract_audio.py`

---

### Step 3: Audio Preprocessing

**Script:** `src/audio/preprocessing/audio_cleaner.py`  
**Function:** `preprocess_audio(file_path)`

Each file goes through this pipeline in order:
1. **Load** — Resample to 16kHz, convert to mono.
2. **Noise Reduction** — Suppress stationary background noise (`noisereduce`, `prop_decrease=0.8`).
3. **Voice Activity Detection (VAD)** — Remove silence using `webrtcvad` (aggressiveness=3, 30ms frames).
4. **RMS Normalization** — Bring all files to a consistent volume (`target_rms=0.1`).
5. **Pre-emphasis Filter** — Amplify high frequencies for better speech clarity (`alpha=0.97`).

---

### Step 4: Feature Extraction

**Script:** `src/audio/feature_extraction/extract_features.py`  
**Function:** `extract_audio_features(audio_array, sample_rate)`

Extracts exactly **20 features** per audio file:

| Category | Feature | Count | Library |
|----------|---------|-------|--------|
| MFCC | mfcc_mean, mfcc_std, delta_mfcc_mean, delta_mfcc_std, delta2_mfcc_mean, delta2_mfcc_std | 6 | librosa |
| Pitch | pitch_mean, pitch_std, pitch_min, pitch_max | 4 | **Parselmouth (Praat F0)** |
| Energy | energy_mean, energy_std, energy_min, rms_energy | 4 | librosa |
| Speech Patterns | speech_rate, pause_duration_mean, pause_frequency, zero_crossing_rate_mean | 4 | librosa / scipy |
| Voice Quality | jitter, shimmer | 2 | Parselmouth (Praat) |

> **Pitch note:** Pitch is extracted using `parselmouth.Sound.to_pitch()` (Praat's autocorrelation-based F0), which is more accurate for clinical speech than `librosa.piptrack`. Unvoiced frames (0 Hz) are filtered out automatically.

---

### Step 5 & 6: Aggregation & Dataset Building

**Script:** `src/audio/feature_extraction/build_dataset.py`

- Scans `data/raw/audio/` folder structure — **no labels.csv needed**.
- For each patient, processes all answer WAV files and extracts 20 features each.
- Aggregates across answers: **mean, std, min, max, median** → final **100 features per patient**.
- Labels are auto-derived from condition folder names.

```bash
python src/audio/feature_extraction/build_dataset.py \
    --data_dir data/raw/audio \
    --output_csv data/processed/features.csv
```

---

### Step 7: Handle Class Imbalance

**Script:** `src/audio/training/balance_data.py`

Mental health datasets are rarely balanced. This script:
1. Visualizes class distribution before and after.
2. Applies **SMOTE** to synthetically oversample minority classes.
3. Also prints `class_weight` values as an alternative.

```bash
python src/audio/training/balance_data.py \
    --input_csv data/processed/features.csv \
    --output_csv data/processed/features_balanced.csv
```

> **Note:** SMOTE needs at least 6 samples per class. If you have fewer, use `class_weight='balanced'` in your model instead.

---

### Step 8 & 9: Model Training & Feature Importance

**Script:** `src/audio/training/train_models.py`

Trains 4 models with 5-Fold Stratified Cross-Validation:
- **Random Forest** — Good baseline, supports feature importance.
- **XGBoost** — Strong gradient boosted trees.
- **SVM** — Good for small datasets with clear margins.
- **LightGBM** — Fast and accurate, good with class imbalance.

```bash
python src/audio/training/train_models.py \
    --input_csv data/processed/features_balanced.csv \
    --model_dir models/audio \
    --eval_dir evaluation/audio
```

**Primary metric: F1-weighted** (more reliable than accuracy for imbalanced mental health data).

For each model, training prints:
```
  Model: Random Forest
  ├── Accuracy  : 0.82
  ├── F1 Score  : 0.79   ← primary metric
  ├── Precision : 0.81
  ├── Recall    : 0.77
  └── AUC-ROC   : 0.85
```

**Model selection:** The model with the **highest F1 score** is saved as the best model.

**Outputs:**
- `models/audio/best_audio_model.joblib` — Best model (selected by F1).
- `models/audio/standard_scaler.joblib` — Scaler for inference.
- `evaluation/audio/` — Confusion matrices and feature importance plots per model.

---

### Step 10: Inference on a New Patient

**Script:** `src/audio/inference/predict.py`  
**Class:** `AudioMentalHealthPredictor`

Runs the full pipeline on raw WAV files for a new patient.

```bash
python src/audio/inference/predict.py \
    --patient_folder data/raw/audio/depression/patient_NEW \
    --model_dir models/audio
```

Output:
```
✅ Final Prediction: Class 1

Confidence Scores:
  - Class 0 (normal): 12.50%
  - Class 1 (depression): 79.30%
  - Class 2 (anxiety): 8.20%
```

---

## Known Limitations & Honest Assessment

> **Be aware of these limitations before relying on results.**

### ⚠️ Feature Concerns

- **MFCC features are over-compressed**: Collapses all 13 coefficients into a single mean/std. Per-coefficient stats (26 features) would preserve more spectral detail.
- ~~**Pitch via `piptrack` is noisy**~~ ✅ **Fixed** — Now uses `parselmouth.to_pitch()` (Praat F0) for accurate clinical-grade pitch.
- **Speech Rate is approximate**: Uses RMS energy peaks as a proxy for syllable rate — not true phoneme boundary detection.

### ⚠️ Training Concerns

- **No hyperparameter tuning**: All models use default parameters.
- ~~**Model selection by accuracy only**~~ ✅ **Fixed** — Now uses **F1-weighted** as the primary metric with full Precision, Recall, and AUC-ROC reporting.
- **SMOTE on high-dimensional data**: Can introduce noise; consider `class_weight='balanced'` for very small datasets.

### ⚠️ Data Concerns

- **Very small datasets**: Mental health audio datasets are typically small (<200 patients). With 100 features and 5-fold CV, overfitting is likely. Regularized models (SVM, LightGBM) will hold up better than Random Forest in this case.
- **"Normal" means Healthy Control**: Any patient in `normal/` should be confirmed healthy — no conditions whatsoever. If a patient in `normal/` has an undiagnosed condition, it directly poisons the model.

### ✅ What is Done Well

- Solid preprocessing pipeline (VAD + noise reduction + normalization).
- Parselmouth used for both Pitch (F0) and Jitter/Shimmer — clinically validated.
- **F1-weighted** is the primary evaluation metric with full Accuracy/Precision/Recall/AUC-ROC reporting.
- Modular, clean code structure — easy to extend.
- Folder-based labelling — no CSV management overhead.
- SMOTE + class weight support for imbalanced data.

---

## Recommended Next Steps

1. **Expand MFCC features** — Extract per-coefficient mean/std for all 13 MFCCs (26 features total instead of 6).
2. ~~**Replace `piptrack` pitch with Parselmouth F0**~~ ✅ Done.
3. ~~**Add F1/AUC-ROC as primary metric**~~ ✅ Done.
4. **Hyperparameter tuning** — Use `GridSearchCV` or `Optuna` for XGBoost/LightGBM.
5. **Binary classification mode** — Train one model per condition (e.g., Depression vs. All) for higher per-condition accuracy.
