# 🧠 Audio Modality — Master Technical Documentation

This document provides a comprehensive deep-dive into the consolidated audio mental health classification pipeline (v2.0). It covers everything from signal processing and clinical feature extraction to hyper-parameter tuning and majority-voting inference.

---

## 1. ARCHITECTURE OVERVIEW

The audio module is located in `src/audio/`. It has been optimized from a fragmented structure into 7 high-performance, flat files.

| File | Purpose |
|---|---|
| `config.py` | Centralized constants, paths, and training parameters. |
| `preprocess.py` | Signal cleaning (denoising, VAD, RMS) and segmentation. |
| `features.py` | 88-feature extractor including clinical voice markers. |
| `dataset.py` | Feature extraction orchestrator + SMOTE class balancing. |
| `train.py` | Multi-model training loop with 9 classifiers and tuning. |
| `predict.py` | End-to-end inference for files and folders. |
| `pipeline.py` | Full automation orchestrator (CLI). |

---

## 2. SIGNAL PROCESSING (`preprocess.py`)

To ensure the model learns mental health markers rather than recording noise, we apply a strict 5-stage cleaning pipeline:

1.  **Noise Reduction**: Uses spectral gating to subtract stationary noise.
2.  **Voice Activity Detection (VAD)**: Uses WebRTC-VAD to strip all non-speech segments.
3.  **RMS Normalization**: Standardizes volume to -20dB to prevent loudness bias.
4.  **Pre-Emphasis**: Boosts high frequencies ($y[t] = x[t] - 0.97 \cdot x[t-1]$) to clarify consonants.
5.  **Segmentation**: Breaks long interviews into 25s overlapping windows (50% overlap).

---

## 3. CLINICAL FEATURE SET (88 Features)

While the previous version used 65 general audio features, v2.0 introduces **clinically-validated markers** essential for mental health detection:

### 🎙️ The "Gold Standard" Markers
- **Jitter (3 types)**: Measures frequency instability. High jitter often indicates vocal strain or anxiety.
- **Shimmer (3 types)**: Measures amplitude instability. Common in depressed, "breathy" voices.
- **HNR (Harmonics-to-Noise Ratio)**: Measures voice clarity. Low HNR correlates with vocal dysphonia in clinical depression.
- **Formants (F1, F2, F3)**: Measures vocal tract configuration. Vowel space reduction (monotone speech) is a key depression signature.

### 📐 Feature Breakdown
- **Spectral**: Centroid, Bandwidth, Contrast, Flatness, Rolloff.
- **Timbre**: 13 MFCCs + 13 Stds + 13 Deltas (39 total).
- **Temporal**: Pause frequency, Mean pause duration, Speech rate, ZCR.
- **Tonal**: 12 Chroma bin means.

---

## 4. MACHINE LEARNING ENGINE (`train.py`)

We train **9 distinct classifiers** to find the best boundary between Normal, Depression, Anxiety, ADHD, and OCD.

### 🏆 Model Lineup
1.  **XGBoost**: Gradient boosting with heavy regularization (Industry standard).
2.  **LightGBM**: Fast, leaf-wise boosting.
3.  **CatBoost**: Handles feature interactions exceptionally well.
4.  **SVM (RBF)**: Non-linear kernel, excellent for small, high-dimensional data.
5.  **Stacking Ensemble**: A meta-classifier that learns how to best weigh the top-3 models.
6.  **Random Forest & Extra Trees**: Bagging models for feature importance stability.

### ⚙️ Automation
- **Hyperparameter Tuning**: `RandomizedSearchCV` runs 60 iterations per model to find optimal depth, learning rates, and regularization.
- **Class Balancing**: SMOTE generates synthetic samples for the minority classes (ADHD/OCD) to prevent majority-class bias.

---

## 5. FULL SOURCE CODE (Master Reference)

Below is the code for the 3 most critical modules.

### A. Feature Extraction (`features.py`)
```python
# Extracting Jitter, Shimmer, HNR, and Formants via Parselmouth
def _voice_quality_features(snd: parselmouth.Sound) -> dict:
    pitch = call(snd, "To Pitch", 0.0, C.PITCH_FLOOR, C.PITCH_CEILING)
    pp    = call(snd, "To PointProcess (periodic, cc)", C.PITCH_FLOOR, C.PITCH_CEILING)

    feats = {
        "jitter_local":   _safe(lambda: call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)),
        "shimmer_local":  _safe(lambda: call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)),
        "hnr_mean":       _safe(lambda: call(call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0), "Get mean", 0, 0)),
    }
    return feats
```

### B. Prediction Engine (`predict.py`)
```python
# Folder-level prediction via Majority Voting
def predict_folder(self, folder: str | Path) -> dict | None:
    preds = [self.predict(wav) for wav in Path(folder).glob("*.wav")]
    avg_probs = {c: np.mean([p["probabilities"][c] for p in preds]) for c in C.LABEL_NAMES.values()}
    best_cond = max(avg_probs, key=avg_probs.get)
    return {"condition": best_cond, "confidence": avg_probs[best_cond], "probabilities": avg_probs}
```

### C. Pipeline Runner (`pipeline.py`)
```bash
# How to trigger the full world
python -m src.audio.pipeline --step 1-4
```

---

## 6. USAGE GUIDE

### Step 1: Segmentation
```bash
python -m src.audio.pipeline --step 1
```
*Turns messy raw data into neat 25s chunks.*

### Step 2: Training
```bash
python -m src.audio.pipeline --step 2-4
```
*Extracts 88 features, balances classes, and trains 9 models.*

### Step 3: Deployment
```python
from src.audio import AudioPredictor
predictor = AudioPredictor()
result = predictor.predict("patient_interview.wav")
```

---

## 7. COMPLETE FEATURE DICTIONARY (88 Features)

Below is the exhaustive list of every feature extracted by the pipeline, categorized by acoustic group.

### 📊 Group A: MFCCs & Dynamics (39 Features)
*Timbre and Vocal Tract Shape*
- **mfcc_0 to mfcc_12 (Means)**: The first 13 Mel-Frequency Cepstral Coefficients. Represent the broad spectral envelope (vocal tract shape).
- **mfcc_0 to mfcc_12 (Stds)**: Standard deviation of each MFCC over time. Measures timber variability.
- **delta_mfcc_0 to delta_mfcc_12 (Means)**: The first-order derivative of MFCCs. Represents the *change* in timbre (speech dynamics).
    - *Clinical Note: Depressed patients often exhibit lower variance in MFCCs, reflecting "monotimbral" speech.*

### 📈 Group B: Pitch & Prosody (5 Features)
*Fundamental Frequency (F0) Statistics*
- **pitch_mean**: Average frequency of vocal fold vibration.
- **pitch_std**: Variation in pitch.
- **pitch_min / pitch_max**: The range of the voice.
- **pitch_range**: Difference between max and min pitch ($F0_{max} - F0_{min}$).
    - *Clinical Note: Reduced pitch range and variability (flat prosody) is a leading indicator of clinical depression.*

### 🎙️ Group C: Voice Quality - Glottal (7 Features)
*Clinical Gold Standards for Vocal Strain*
- **jitter_local**: Cycle-to-cycle variation in fundamental frequency.
- **jitter_rap**: Relative Average Perturbation (variability over 3 cycles).
- **jitter_ppq5**: Five-point Period Perturbation Quotient.
- **shimmer_local**: Cycle-to-cycle variation in amplitude (loudness).
- **shimmer_apq3**: Three-point Amplitude Perturbation Quotient.
- **shimmer_apq5**: Five-point Amplitude Perturbation Quotient.
- **hnr_mean**: Harmonics-to-Noise Ratio.
    - *Clinical Note: High Jitter/Shimmer + Low HNR = breathy, unstable, or "creaky" voice, often found in anxiety (mechanical strain) or depression (vocal cord weakness).*

### 🌊 Group D: Formants & Articulation (5 Features)
*Resonance and Vowel Space*
- **f1_mean**: First formant (correlates with jaw opening).
- **f2_mean**: Second formant (correlates with tongue position).
- **f3_mean**: Third formant (correlates with cavity size).
- **f1_std / f2_std**: Variation in formants during speech.
    - *Clinical Note: "Vowel Space Reduction"—where F1 and F2 cluster together—indicates precise articulation loss, common in psychomotor retardation.*

### 🔉 Group E: Energy & Intensity (5 Features)
*Loudness and Effort*
- **energy_mean**: Average power of the signal.
- **energy_std**: Variability in loudness.
- **energy_max**: Peak intensity.
- **rms_mean / rms_std**: Root-Mean-Square energy statistics.
    - *Clinical Note: Lower overall energy and lower energy variance are symptoms of the "flat affect" in depression.*

### 🌈 Group F: Spectral Shape (10 Features)
*Frequency Domain Descriptors*
- **spec_centroid_mean / std**: The "center of mass" of the spectrum. (Brightness).
- **spec_bandwidth_mean / std**: The width of the spectral range used.
- **spec_contrast_mean / std**: Difference between spectral peaks and valleys (Clarity).
- **spec_flatness_mean / std**: How "noise-like" the signal is (Smoothness).
- **spec_rolloff_mean / std**: The frequency below which 85% of power resides.

### ⏱️ Group G: Speech Patterns (5 Features)
*Temporal and Rhythmic Markers*
- **zcr_mean / std**: Zero-Crossing Rate (High-frequency content/breathiness).
- **pause_frequency**: Number of silences > 50ms per second.
- **pause_duration_mean**: Average length of silences.
- **speech_rate**: Number of speech-bursts per second.
    - *Clinical Note: ADHD often presents as high speech rate / low pause duration. Depression/Anxiety present as low speech rate / high pause frequency.*

### 🎹 Group H: Chroma (12 Features)
*Tonal Content*
- **chroma_0 to chroma_11 (Means)**: 12-bin representation of the energy distributed across semitones. Captures harmonic richness and "musicality" of the voice.

---
*End of Complete Feature Dictionary*

