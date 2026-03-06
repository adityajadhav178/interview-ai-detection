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

| Category | Feature | Count |
|----------|---------|-------|
| MFCC | mfcc_mean, mfcc_std, delta_mfcc_mean, delta_mfcc_std, delta2_mfcc_mean, delta2_mfcc_std | 6 |
| Pitch | pitch_mean, pitch_std, pitch_min, pitch_max | 4 |
| Energy | energy_mean, energy_std, energy_min, rms_energy | 4 |
| Speech Patterns | speech_rate, pause_duration_mean, pause_frequency, zero_crossing_rate_mean | 4 |
| Voice Quality | jitter, shimmer (via Praat/parselmouth) | 2 |

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

**Outputs:**
- `models/audio/best_audio_model.joblib` — Best model saved automatically.
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

- **MFCC features are over-compressed**: The current pipeline collapses all 13 MFCC coefficients into a single mean/std. Standard practice is to keep per-coefficient statistics (13 means + 13 stds = 26 features just for MFCC). This means the model loses spectral detail.
- **Pitch via `piptrack` is noisy**: `librosa.piptrack` is known to produce inaccurate pitch for speech. Parselmouth (already used for jitter/shimmer) would give more reliable fundamental frequency (F0) estimates.
- **Speech Rate is approximate**: The current "syllable rate" uses RMS energy peaks, not true phoneme/syllable boundary detection. This is a rough proxy.

### ⚠️ Training Concerns

- **No hyperparameter tuning**: All models use default parameters. Real performance will be higher after tuning.
- **Model selection by accuracy only**: Accuracy is misleading for imbalanced classes. Use **F1-score** or **AUC-ROC** to select the best model instead.
- **SMOTE on high-dimensional 100-feature space**: SMOTE can introduce artificial noise; consider alternatives like `ADASYN` or simply `class_weight='balanced'` for small datasets.

### ⚠️ Data Concerns

- **Very small datasets**: Mental health audio datasets are typically small (<200 patients). With 100 features and 5-fold CV, overfitting is likely. Regularized models (SVM, LightGBM) will hold up better than Random Forest in this case.
- **"Normal" means Healthy Control**: Any patient in `normal/` should be confirmed healthy — no conditions whatsoever. If a patient in `normal/` has an undiagnosed condition, it directly poisons the model.

### ✅ What is Done Well

- Solid preprocessing pipeline (VAD + noise reduction + normalization).
- Parselmouth integration for Jitter/Shimmer (clinically validated voice quality markers).
- Modular, clean code structure — easy to extend.
- Folder-based labelling — no CSV management overhead.
- SMOTE + class weight support for imbalanced data.

---

## Recommended Next Steps

1. **Expand MFCC features** — Extract per-coefficient mean/std for all 13 MFCCs (adds 20 more features but much richer).
2. **Replace `piptrack` pitch with Parselmouth F0** — Much more accurate for clinical speech.
3. **Add F1/AUC-ROC** as primary evaluation metric in `train_models.py`.
4. **Hyperparameter tuning** — Use `GridSearchCV` or `Optuna` for XGBoost/LightGBM.
5. **Binary classification mode** — Train one model per condition (e.g., Depression vs. All) for higher per-condition accuracy.
