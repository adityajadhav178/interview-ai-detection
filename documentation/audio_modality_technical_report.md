# Technical Documentation Report: Audio Modality

This document serves as the comprehensive technical reference for the audio modality in the mental health diagnosis AI detection suite. It provides a detailed, end-to-end overview of audio preprocessing, feature extraction, training, and inference, sufficient for replication by a new developer.

---

## 1. OVERVIEW
- **What this modality does**: Processes patient speech recordings, extracts clinically relevant diagnostic acoustic features, and uses classical ML models to predict mental health conditions.
- **What problem it solves**: Automates the extraction of acoustic markers for mental states (such as psychomotor retardation or anxiety), providing an objective indicator to supplement subjective questionnaires and visual cues.
- **Input & Output**: 
  - *Input*: Raw video (`.mp4`) or audio (`.wav`) files. In clinical testing, up to 5 speech response files per patient are expected. 
  - *Output*: A predicted class label (e.g., `normal`, `depression`, `anxiety`, `adhd`, `ocd`) alongside class-wise confidence probabilities.

---

## 2. DEPENDENCIES & LIBRARIES
Below are the key dependencies required for the audio pipeline:

- **Core & Data Handling**: `numpy`, `pandas`, `scipy`
- **Machine Learning**: `scikit-learn` (metrics, CV, scaling, models), `xgboost`, `lightgbm`, `imbalanced-learn` (for SMOTE), `joblib` (for saving models)
- **Audio/Video Processing**: 
  - `librosa`: Loading audio, extracting MFCCs, energy, and speech patterns.
  - `noisereduce`: Spectral noise gating.
  - `moviepy`: Extracting audio tracks from video files.
  - `webrtcvad` / `webrtcvad-wheels`: Robust Voice Activity Detection (silence removal).
  - `praat-parselmouth`: Clinical-grade pitch extraction (widely used in linguistic/medical research).
- **Visualization Utilities**: `matplotlib`, `seaborn`, `tqdm`

**Environment Initialization Command**:
```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm imbalanced-learn joblib librosa noisereduce moviepy webrtcvad-wheels praat-parselmouth matplotlib seaborn tqdm
```

---

## 3. PREPROCESSING PIPELINE
Preprocessing cleans raw environmental recordings into noise-free, standardized speech signals. Located in `src/audio/preprocessing/`.

**Step 0: Video to Audio (`extract_audio.py` -> `extract_audio_from_video`)**
- Reads `.mp4` using `moviepy`.
- Extracts audio and forces: `fps=16000`, 1 channel (`mono`), `16-bit PCM`.
- Skips gracefully if no audio track exists.

**Step 1: Load Audio (`audio_cleaner.py` -> `preprocess_audio`)**
- `librosa.load(file_path, sr=16000, mono=True)` reads the file. Returns `None` if empty or corrupted.

**Step 2: Noise Reduction**
- `noisereduce.reduce_noise(y=y, sr=sr, prop_decrease=0.8)`. Reduces steady-state background noise by 80%.

**Step 3: Voice Activity Detection (`run_vad_on_audio`)**
- Strips non-speech (silence) segments.
- Segments audio into `30ms` frames.
- Pads array with zeros if length is not perfectly divisible by the frame size.
- Uses `webrtcvad.Vad(aggressiveness=3)` (Maximum strictness for filtering noise).
- Only concatenates frames identified as speech.
- *Edge case*: If VAD flags the entire file as silence, it safely **returns the original un-VADed array** to prevent crashing.

**Step 4: RMS Amplitude Normalization (`normalize_rms`)**
- Standardizes loudness. Computes overall RMS: $\sqrt{mean(audio^2)}$.
- Multiplies audio by `(0.1 / current_rms)` to set a target RMS of `0.1`.

**Step 5: Pre-emphasis Filter (`apply_preemphasis`)**
- Amplifies high frequencies to balance the frequency spectrum (essential for MFCC).
- Formula: $y(t) = x(t) - 0.97 \cdot x(t-1)$, where `0.97` is the pre-emphasis coefficient.

---

## 4. FEATURE EXTRACTION
Extracts **18 features** across four categories per audio slice. Defined in `src/audio/feature_extraction/extract_features.py`. Returns a dictionary of float values.

| Category | Features | Method / Library | Parameters |
| :--- | :--- | :--- | :--- |
| **MFCC** (6) | `mfcc_mean`, `mfcc_std`, `delta_mfcc_mean`, `delta_mfcc_std`, `delta_delta_mfcc_mean`, `delta_delta_mfcc_std` | `librosa.feature.mfcc` & `librosa.feature.delta` | `n_mfcc=13`, `sr=16000`, delta order 1 & 2 |
| **Pitch** (4) | `pitch_mean`, `pitch_std`, `pitch_min`, `pitch_max` | `parselmouth.Sound(...).to_pitch()` | Filters out frequency <= 0. *Edge Case: defaults to 0.0 if failed/none.* |
| **Energy** (4) | `energy_mean`, `energy_std`, `energy_min`, `rms_energy` | `librosa.feature.rms` (frames) & `np.sqrt(np.mean(y**2))` | Default frame sizes. |
| **Speech** (4) | `zero_crossing_rate_mean`, `pause_frequency`, `pause_duration_mean`, `speech_rate` | `librosa.feature.zero_crossing_rate`, manual pause gap tracking, `scipy.signal.find_peaks` | Pause defined as amp < 10% peak AND gap > `0.05s`. `distance=10` for peaks. |

*Output Type*: Dictionary of 18 Keys -> float mappings.

---

## 5. DATASET CREATION
Defined in `src/audio/feature_extraction/build_dataset.py` -> `build_dataset`.

**Workflow**:
1. Crawls `data/raw/audio/` looking for condition subfolders matching the map: `{"normal": 0, "depression": 1, "anxiety": 2, "adhd": 3, "ocd": 4}`.
2. Identifies nested (`patient_001/*.wav`) or flat (`*.wav`) structure. 
3. Iterates over `.wav` files, processing them through `audio_cleaner` -> `extract_features`.
4. **Aggregation** (`aggregate_features`): For a patient with multiple responses, we calculate the `mean`, `std`, `min`, `max`, and `median` of *each* of the 18 extracted features across their files.
5. **Final Matrix Shape**: 18 features $\times$ 5 mathematical aggregations = **90 aggregate features** per patient row.
6. **Dataframe Layout**: `['patient_id', 'condition', 'label', 'mfcc_mean_mean', 'mfcc_mean_std', ... 88 more columns]`.
7. Saves outputs grouped by class as `features_{condition}.csv`.

---

## 6. CLASS BALANCING
Defined in `src/audio/training/balance_data.py`. Mental health datasets suffer from severe class imbalance.

- **Technique**: **SMOTE** (Synthetic Minority Over-sampling Technique).
- **Parameters**: `random_state=42`, default `k_neighbors=5`.
- **Implementation**: Drops `patient_id` during synthesis. Real patients keep their IDs, but synthetic samples appended get a `SYNTHETIC_n` id tag.
- **Reporting**: Prints distribution counts to console, plots bar charts `class_distribution.png`, and outputs `data/processed/features_balanced.csv`.
- *(Alternative Provided in Code)*: Computes inverse class weights `compute_class_weight('balanced')` in case researchers wish to run XGBoost/RF natively on imbalanced data.

---

## 7. MODEL TRAINING
Defined in `src/audio/training/train_models.py`.

- **Splitting**: `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
- **Normalization**: `StandardScaler`. Fit only on training data, transformed on both. Saved to disk for inference.
- **Models Evaluated**:
  1. `RandomForestClassifier` (`n_estimators=100`, `n_jobs=-1`)
  2. `XGBClassifier` (`eval_metric='mlogloss'`, `use_label_encoder=False`)
  3. `SVC` (`kernel='rbf'`, `probability=True`)
  4. `LGBMClassifier` (`n_jobs=-1`)
- **Evaluation Criteria**: Main target metric is **Weighted F1-Score** evaluated via `StratifiedKFold(n_splits=5)`. Also reports Accuracy, Precision, Recall, AUC-ROC (OVR weighted). 
- **Artifacts Generated**:
  - `*_confusion_matrix.png` per model.
  - `*_top_features.png` and `*_category_importance.png` for tree-based interpretability mapping features back to their clinical categories (e.g. MFCC vs Pitch).
  - Best model is saved based strictly on F1 Test Score as `best_audio_model.joblib`. Contains `{"model": <obj>, "label_mapping": {0: 1 ...}}`.

---

## 8. INFERENCE / PREDICTION
Defined in `src/audio/inference/predict.py`. Predicts class for a new patient from scratch.

**Pipeline**:
1. Class `AudioMentalHealthPredictor` instantiated. Loads `standard_scaler.joblib` and `best_audio_model.joblib`.
2. Passed a directory containing patient data: `predict("path/to/patient/wavs/")`.
3. Selects up to `5` `.wav` files (cuts off excess). 
4. Executes: `preprocess_audio` -> `extract_audio_features` on each file.
5. A list of feature dictionaries is aggregated using `aggregate_features()`.
6. Converted to a single-row Pandas DataFrame. Missing/mismatching columns are safely aligned with `scaler.feature_names_in_` and filled with `0.0`.
7. Multiplies data through `scaler.transform()`.
8. Obtains prediction using `model.predict()` and confidence via `model.predict_proba()`. Probabilities are mapped back to their original semantic labels.
9. **Final Output Format**: `{'predicted_label': 1, 'probabilities': {0: 0.12, 1: 0.88}}`

---

## 9. FILE & FOLDER STRUCTURE
**`src/audio/`** - Core Logic:
- `preprocessing/extract_audio.py` : Video mp4 to Audio wav conversion script.
- `preprocessing/audio_cleaner.py` : SNR enhancement, noise gating, VAD.
- `feature_extraction/extract_features.py` : 18-acoustic-feature mathematical extraction logic.
- `feature_extraction/build_dataset.py` : Aggregates individual files into tabular datasets.
- `training/balance_data.py` : SMOTE class rebalancing & plotting.
- `training/train_models.py` : ML Pipeline, 5-Fold Evaluation, Feature Importance interpretation.
- `inference/predict.py` : Re-usable class to wrap the entire flow for new, unseen user requests.

**Data & Models**:
- `data/raw/audio/{condition}/{patient}/` : Input data structure expected.
- `data/processed/` : Processed CSV files.
- `models/audio/` : Pickled ML estimators and scalers.
- `evaluation/audio/` : Confusion matrices and interpretability plots.

---

## 10. CONFIGURATION & CONSTANTS
Key hyperparameters hardcoded into the pipeline governing the audio logic:
- `SAMPLE_RATE`: `16000 Hz` (Industry standard for Voice Activity Detection).
- `VAD_FRAME_DURATION`: `30 ms` (Chunk length for speech classification).
- `VAD_AGGRESSIVENESS`: `3` (Most aggressive cutoff for silence).
- `NOISE_PROP_DECREASE`: `0.80` (Preserves 20% harmonic noise to avoid artifacting).
- `RMS_TARGET`: `0.1` (Constant energy baseline for uniformity).
- `PRE_EMPHASIS_COEFF`: `0.97` (Standard acoustic pre-emphasis coefficient).
- `N_MFCC`: `13` (Discarding upper bands unnecessary for speech).
- `PAUSE_THRESHOLD`: `0.1 * peak_amplitude` lasting > `0.05 seconds`.
- `TEST_SIZE`: `0.20` for train-test split.
- `RANDOM_STATE`: `42` globally ensuring reproducibility across SKLearn, SMOTE, and Boosting libraries.

---

## 11. KNOWN LIMITATIONS & ASSUMPTIONS
- **VAD Failure**: If a clip contains only deep sighs or crying without distinct speech patterns, `webrtcvad` may incorrectly strip the entire file. The pipeline safely catches this by bypassing the VAD and returning the noisy original file.
- **Pitch Fallback**: Parselmouth fails to extract F0 intervals on severely glottalised, "creaky" voices or very short utterances. In these failure cases, Pitch features (`pitch_min`, `pitch_max`, `pitch_mean`, `pitch_std`) safely default to constant `0.0`.
- **SMOTE Limitation**: SMOTE algorithm requires at least 6 records minimum in the smallest target class due to its default `k_neighbors=5` setup during interpolation. 
- **Time Constraints**: Inference currently relies on the standard structure of `5 replies` per patient to establish reliable aggregative patterns (min/max variance over a dialogue). Applying inference with only `1` extremely short audio file will lead to standard deviations = 0.0, which severely limits model confidence.
