# Interview AI Detection (Multimodal)

Research-oriented Python project for detecting mental health conditions from **four modalities**:
- **MCQ** questionnaire data
- **Audio** features extracted from interview responses
- **Text** transcripts generated from speech
- **Video** facial behaviour analysis

The repository is structured so **four developers can work independently** on one modality under `src/{audio,video,text,mcq}/` with consistent `preprocessing/`, `feature_extraction/` (or `feature_engineering/`), `training/`, and `inference/` subpackages. Fusion lives in `src/fusion/`.

---

## Repository Layout

```
interview-ai-detection/
├── data/
│   ├── raw/
│   │   ├── video/                  ← Drop raw .mp4 interview videos here
│   │   │   ├── depression/
│   │   │   ├── anxiety/
│   │   │   ├── adhd/
│   │   │   ├── ocd/
│   │   │   └── normal/             ← Healthy controls (no diagnosis)
│   │   └── audio/                  ← Auto-populated by extract_audio.py
│   │       ├── depression/
│   │       │   └── patient_001/
│   │       │       ├── answer_1.wav
│   │       │       └── answer_2.wav
│   │       ├── anxiety/
│   │       ├── adhd/
│   │       ├── ocd/
│   │       └── normal/
│   ├── processed/
│   │   ├── audio/                  ← Cleaned audio segments (per condition)
│   │   │   ├── depression/
│   │   │   ├── anxiety/
│   │   │   ├── adhd/
│   │   │   ├── ocd/
│   │   │   └── normal/
│   │   ├── video/                  ← Processed video frames (per condition)
│   │   │   ├── depression/
│   │   │   ├── anxiety/
│   │   │   ├── adhd/
│   │   │   ├── ocd/
│   │   │   └── normal/
│   │   ├── features.csv            ← Audio features (auto-built, no labels.csv)
│   │   └── features_balanced.csv   ← After SMOTE balancing
│   └── annotations/                ← Optional: timestamps, segments, alignments
├── src/
│   ├── audio/
│   │   ├── preprocessing/          ← extract_audio.py, audio_cleaner.py
│   │   ├── feature_extraction/     ← extract_features.py, build_dataset.py
│   │   ├── training/               ← train_models.py, balance_data.py
│   │   └── inference/              ← predict.py
│   ├── video/
│   ├── text/
│   ├── mcq/
│   └── fusion/
├── models/
│   └── audio/                      ← best_audio_model.joblib, standard_scaler.joblib
├── evaluation/
│   └── audio/                      ← Confusion matrices, feature importance plots
├── checkpoints/
├── configs/
│   └── audio_config.yaml
├── scripts/
│   └── train_audio.py
├── documentation/
│   └── audio_pipeline_guide.md     ← Detailed audio pipeline guide
├── notebooks/
├── requirements.txt
└── main_pipeline.py
```

**Label mapping — folder name = label (auto-derived, no CSV needed):**

| Folder | Label | Meaning |
|--------|-------|---------|
| `normal` | 0 | Healthy control (no diagnosis) |
| `depression` | 1 | Major Depressive Disorder |
| `anxiety` | 2 | Anxiety Disorder |
| `adhd` | 3 | ADHD |
| `ocd` | 4 | OCD |

---

## Quickstart

Create an environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Audio Pipeline (Step-by-Step)

```bash
# 1. Extract audio from videos (preserves condition folder structure)
python src/audio/preprocessing/extract_audio.py \
    --input_dir data/raw/video \
    --output_dir data/raw/audio

# 2. Build feature dataset (labels auto-derived from folder names)
python src/audio/feature_extraction/build_dataset.py \
    --data_dir data/raw/audio \
    --output_csv data/processed/features.csv

# 3. Balance classes using SMOTE
python src/audio/training/balance_data.py \
    --input_csv data/processed/features.csv \
    --output_csv data/processed/features_balanced.csv

# 4. Train models (RF, XGBoost, SVM, LightGBM) — best saved by F1 score
python src/audio/training/train_models.py \
    --input_csv data/processed/features_balanced.csv \
    --model_dir models/audio \
    --eval_dir evaluation/audio

# 5. Predict on a new patient
python src/audio/inference/predict.py \
    --patient_folder data/raw/audio/depression/patient_NEW \
    --model_dir models/audio
```

---

## Conventions

- **Data immutability**: treat `data/raw/` as read-only; write derived artifacts to `data/processed/`.
- **No manual labelling**: folder name determines the label — just drop files in the right folder.
- **Reproducibility**: keep experiment parameters in YAML configs; log runs + random seeds.
- **Separation of concerns**:
  - `preprocessing/`: cleaning, resampling, segmentation, normalization
  - `feature_extraction/` / `feature_engineering/`: transform into model-ready tensors/features
  - `training/`: training loops, model definitions, losses, optimizers
  - `inference/`: loading artifacts and producing predictions
- **Fusion**:
  - **Early fusion**: concatenate/merge representations before a unified model
  - **Late fusion**: combine modality-specific predictions (e.g., stacking, weighted averaging)

---

## Audio Pipeline — Current Status

| Component | Status |
|-----------|--------|
| Video → Audio extraction | ✅ Done |
| Audio preprocessing (VAD, noise, RMS, pre-emphasis) | ✅ Done |
| Feature extraction — 20 features/file (Parselmouth pitch + Jitter/Shimmer) | ✅ Done |
| Feature aggregation — 100 features/patient | ✅ Done |
| SMOTE class balancing | ✅ Done |
| Model training (RF, XGB, SVM, LightGBM) + F1 evaluation | ✅ Done |
| Inference pipeline | ✅ Done |
| Documentation | ✅ `documentation/audio_pipeline_guide.md` |
| Binary per-condition models | 🔲 Planned |
| Hyperparameter tuning | 🔲 Planned |

---

## Next Steps

- Connect each modality pipeline to real datasets under `data/raw/`.
- Implement binary classifiers per condition (Depression vs. All, Anxiety vs. All, etc.).
- Add experiment tracking (MLflow / W&B) and tests for data contracts.
