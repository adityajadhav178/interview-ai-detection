# 🧠 Audio Modality v2.0 — Technical Documentation

## Overview
The audio modality pipeline classifies mental health conditions (Normal, Depression, Anxiety, ADHD, OCD) from interview audio. It has been restructured from a fragmented directory structure into a high-performance, centralized 8-file module in `src/audio/`.

## 📁 Project Structure
```
src/audio/
├── __init__.py       # Public API (run_pipeline, AudioPredictor)
├── config.py         # Centralized constants, paths, and ML settings
├── preprocess.py     # Load -> Denoise -> VAD -> Normalize -> Segment
├── features.py       # 88 clinical features (Jitter, Shimmer, HNR, Formants)
├── dataset.py        # Feature CSV building + SMOTE class balancing
├── train.py          # 9 ML models + RandomizedSearchCV tuning
├── predict.py        # End-to-end inference (Single file / Folder voting)
└── pipeline.py       # 4-step CLI orchestrator
```

## 🛠️ Pipeline Steps

### 1. Preprocessing (`preprocess.py`)
- **Noise Reduction**: Spectral-gated denoising.
- **VAD**: WebRTC Voice Activity Detection to remove silence.
- **Normalization**: RMS amplitude matching to -20dB.
- **Segmentation**: Normalizes audio lengths.
  - < 10s: Padded with silence.
  - 10-25s: Kept as-is.
  - > 25s: Split into 25s segments with 50% overlap.

### 2. Feature Extraction (`features.py`)
Extracts **88 features** including:
- **MFCC (39)**: 13 coefficients + stds + deltas.
- **Voice Quality (7)**: Jitter (Local, RAP, PPQ5), Shimmer (Local, APQ3, APQ5), and HNR.
- **Formants (5)**: F1-F3 frequencies + F1-F2 stds (Vowel space analysis).
- **Pitch (5)**: Fundamental frequency stats via Parselmouth.
- **Spectral (10)**: Centroid, Bandwidth, Contrast, Flatness, Rolloff.
- **Temporal (5)**: Zero-crossing rate, Pause frequency/duration, Speech rate.
- **Chroma (12)**: Tonal characteristics.

### 3. Model Training (`train.py`)
Features 9 diverse models with automated hyperparameter tuning:
- **XGBoost & LightGBM**: Primary gradient boosting models.
- **SVM (RBF)**: For high-dimensional feature separation.
- **Stacking Ensemble**: Meta-learner combining the top-3 base models.
- **Random Forest & Extra Trees**: Bagging-based baselines.

## 🚀 Usage

### Command Line Interface
```bash
# Run the entire pipeline (Segment -> Extract -> Balance -> Train)
python -m src.audio.pipeline

# Run specific steps
python -m src.audio.pipeline --step 1      # Preprocessing only
python -m src.audio.pipeline --step 4      # Training only
```

### Python Implementation
```python
from src.audio import run_pipeline, AudioPredictor

# Orchestrate training
run_pipeline()

# Run inference
predictor = AudioPredictor()
result = predictor.predict("path/to/voice.wav")
print(f"Condition: {result['condition']} (Conf: {result['confidence']:.2%})")
```

## 📊 Key Enhancements in v2.0
1. **Clinical Features**: Added Jitter and Shimmer which were missing in v1.0 but are critical for mental health detection.
2. **Structural Consolidation**: Reduced codebase complexity from 20+ files to 8.
3. **Advanced ML**: Moved from default parameters to a tuned Stacking Ensemble architecture.
4. **Majority Voting**: Inference now supports folder-level prediction for multi-answer interviews.
