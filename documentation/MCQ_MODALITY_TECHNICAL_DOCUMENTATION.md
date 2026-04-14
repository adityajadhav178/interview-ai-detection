# MCQ Modality — Technical Documentation

Complete technical documentation for the MCQ (Multiple Choice Questionnaire) modality in the Interview AI Detection multimodal mental health system.

---

## 1. OVERVIEW

### What this modality does
The MCQ modality uses psychometric questionnaire data to predict whether a respondent is classified as a real patient (positive for mental health condition) or not. It ingests MCQ answers from MongoDB, converts them into fixed-length feature vectors, and trains binary classifiers (Logistic Regression, Random Forest, XGBoost) to predict the `isRealPatientData` label.

### Problem it solves
- Automated screening of mental health conditions (depression, anxiety, OCD, ADHD) from self-reported questionnaire responses
- Scalable processing of psychometric data stored in MongoDB
- Reproducible ML pipeline from raw data to trained models

### Input
- **Source**: MongoDB documents from collection `fake_aditya_tests` (database: `mental_health_db`)
- **Document shape**:
  ```json
  {
    "_id": ObjectId,
    "userId": ObjectId,
    "testType": "depression" | "anxiety" | "ocd" | "adhd",
    "isRealPatientData": true | false,
    "mcqAnswers": [
      { "questionId": 1, "answer": "Often", "score": 2 },
      { "questionId": 2, "answer": "Always", "score": 3 }
    ],
    "mcqCompleted": true,
    "subjectiveCompleted": false,
    "createdAt": ISODate
  }
  ```

### Output
- **Unified CSV**: `data/processed/mcq/all_diseases_dataset.csv` — single table with q1–q15, one-hot disease columns, and label
- **Trained models**: `models/mcq_model/mcq_logreg.joblib`, `mcq_rf.joblib`, `mcq_xgb.joblib`
- **Label**: Binary (0 = not real patient, 1 = real patient)

---

## 2. DEPENDENCIES & LIBRARIES

| Library | Purpose |
|---------|---------|
| `pandas` | DataFrame ops, CSV I/O |
| `numpy` | Numeric ops, median imputation |
| `scikit-learn` | Train/test split, LogisticRegression, RandomForest, cross_val_score, StratifiedKFold, metrics |
| `xgboost` | XGBClassifier |
| `pymongo` | MongoDB client, `MongoClient` |
| `joblib` | Model serialization (`joblib.dump` / `joblib.load`) |
| `pyyaml` | Load `mcq_config.yaml` (scripts) |

### Versions (from `requirements.txt`)
Versions are not pinned. Current project `requirements.txt` lists:
- numpy, pandas, scipy
- scikit-learn, xgboost, joblib
- pymongo
- (plus other modalities: torch, librosa, opencv, etc.)

### Full pip install command
```bash
pip install numpy pandas scipy scikit-learn xgboost joblib pymongo pyyaml
```

---

## 3. PREPROCESSING PIPELINE

### 3.1 MongoDB fetch (`fetch_from_mongo.py`)

#### Step 1: `connect_mongo(mongo_uri: str | None = None) -> MongoClient`
- **Input**: Optional `mongo_uri`. Falls back to env var `MONGODB_URI`.
- **Output**: `pymongo.MongoClient`.
- **Edge case**: Raises `ValueError` if no URI is provided.

#### Step 2: `_load_disease_question_ids(config_path: Path, disease: str) -> list[int]`
- **Input**: Path to `configs/mcq_questions.json`, disease name (e.g. `"depression"`).
- **Output**: Sorted list of question IDs for that disease (e.g. `[1, 2, ..., 15]`).
- **Behavior**: Supports `{"disease": [1,2,3]}` or `{"disease": [{"id": 1, ...}, ...]}`.
- **Edge case**: Raises `KeyError` if disease not in config; `ValueError` if format unsupported.

#### Step 3: `fetch_mcq_data(client, disease, db_name, collection_name) -> Iterable[dict]`
- **Input**: MongoClient, disease name, db name (`mental_health_db`), collection (`fake_aditya_tests`).
- **Query**: `{"testType": disease, "mcqCompleted": True}`.
- **Projection**: `_id`, `testType`, `mcqAnswers`, `isRealPatientData`.
- **Output**: Cursor of documents.
- **Edge case**: Skips docs with `mcqCompleted != True`.

#### Step 4: `build_feature_vector(mcq_answers: list[dict], question_ids: list[int]) -> list[int]`
- **Input**: `mcqAnswers` array, list of question IDs.
- **Output**: Fixed-length vector `[score_q1, score_q2, ..., score_q15]` (15 ints).
- **Rules**:
  - Uses `score` only; no encoding of answer text.
  - Missing questions → `-1`.
  - Ignores `None` entries, invalid types (TypeError/ValueError).
- **Edge cases**: Empty `mcqAnswers` → all `-1`; invalid qid/score → skip that answer.

#### Step 5: `save_unified_dataset(rows, question_ids, disease_names, output_path, include_label=True) -> Path`
- **Input**: Rows as `(vec, disease, label)` tuples; question_ids; disease_names; output path.
- **Output**: CSV with columns `q1..q15`, `depression`, `anxiety`, `ocd`, `adhd`, `label`.
- **One-hot**: For each row, exactly one disease column is 1, others 0.

### 3.2 Notebook preprocessing (`preprocessing.ipynb`)

| Step | Function/Logic | Input | Output |
|------|----------------|-------|--------|
| 1 | `pd.read_csv(all_diseases_path)` | Path to `all_diseases_dataset.csv` | `df`: (N, 20) DataFrame |
| 2 | Missing value check | `df` | NaN per col, `-1` count per feature |
| 3 | Replace `-1` with NaN, impute with column median | `df`, `feature_cols` | `df_clean`: no NaN |
| 4 | `df_clean.drop(columns=["label"])` | `df_clean` | `X` (N, 19), `y` (N,) |
| 5 | `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` | `X`, `y` | `X_train`, `X_test`, `y_train`, `y_test` |

### Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `test_size` | 0.2 | 80% train, 20% test |
| `random_state` | 42 | Reproducibility |
| `stratify` | y | Preserve label distribution |
| Missing imputation | Column median | Robust to outliers; only for `-1`/NaN in feature cols |

### Edge cases
- **Empty mcqAnswers**: All question scores set to `-1`; later imputed by median.
- **Missing `isRealPatientData`**: Treated as falsy → `label=0`.
- **Invalid score types**: Ignored; question stays `-1`.

---

## 4. FEATURE EXTRACTION

The MCQ modality does not use classical audio/video feature extraction. “Features” are the questionnaire scores and one-hot disease columns.

### 4.1 MCQ scores (q1–q15)
- **Source**: `score` field in each `mcqAnswers` entry.
- **Computation**: Direct mapping `questionId → score`.
- **Library**: None (Python dict lookup).
- **Value range**: 0–3 (ordinal psychometric scale, e.g. Never=0, Rarely=1, Sometimes=2, Often/Always=3).
- **Missing**: `-1`, later imputed by median.

### 4.2 One-hot disease (depression, anxiety, ocd, adhd)
- **Computation**: `1 if row.disease == col else 0` per disease.
- **Value range**: 0 or 1.
- **Constraint**: Exactly one 1 per row.

### 4.3 Final feature set
| Column | Type | Range |
|--------|------|-------|
| q1 | int | 0–3 (or median-imputed) |
| q2 | int | 0–3 |
| … | … | … |
| q15 | int | 0–3 |
| depression | int | 0 or 1 |
| anxiety | int | 0 or 1 |
| ocd | int | 0 or 1 |
| adhd | int | 0 or 1 |

**Total features**: 19 (15 MCQ + 4 one-hot).

---

## 5. DATASET CREATION

### 5.1 Raw data source
- **MongoDB**: `clusterproject.ijackeq.mongodb.net` (Atlas)
- **Database**: `mental_health_db`
- **Collection**: `fake_aditya_tests`

### 5.2 Folder structure expected
```
data/
  raw/mcq/          # Not used (data comes from MongoDB)
  processed/mcq/
    all_diseases_dataset.csv   # Primary output
    {disease}_dataset.csv      # Legacy per-disease (with --disease)
```

### 5.3 Aggregation per sample
- One row per document.
- No aggregation (mean/std/min/max) across multiple files.
- Each row: 15 question scores + 4 one-hot + 1 label.

### 5.4 Label assignment
- `label = 1` if `isRealPatientData == True`, else `0`.

### 5.5 Final CSV structure (`all_diseases_dataset.csv`)

| Column | Represents |
|--------|------------|
| q1 | Score for question 1 (disease-specific) |
| q2 | Score for question 2 |
| … | … |
| q15 | Score for question 15 |
| depression | 1 if depression test, else 0 |
| anxiety | 1 if anxiety test, else 0 |
| ocd | 1 if OCD test, else 0 |
| adhd | 1 if ADHD test, else 0 |
| label | 1 = real patient, 0 = not |

**Example row**: `2,3,3,3,3,2,2,3,2,2,2,3,2,2,2,1,0,0,0,1`  
(depression test, real patient).

### 5.6 Normal vs depressed
- No separate “normal” vs “depressed” CSVs.
- Single CSV; label column encodes class (0/1).

---

## 6. CLASS BALANCING

**Technique used**: None.

- **Before**: Depends on MongoDB data; typically balanced (e.g. 800 vs 800 in test set).
- **After**: Unchanged.
- **Stratify**: `stratify=y` in `train_test_split` keeps class ratio in train/test.
- **Note**: `imbalanced-learn` is in `requirements.txt` but not used in MCQ pipeline.

---

## 7. MODEL TRAINING

### 7.1 Models

| Model | Class | Hyperparameters |
|-------|-------|-----------------|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | `max_iter=1000`, `n_jobs=-1` |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | `n_estimators=200`, `random_state=42`, `n_jobs=-1` |
| XGBoost | `xgboost.XGBClassifier` | `n_estimators=300`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.9`, `colsample_bytree=0.9`, `objective="binary:logistic"`, `eval_metric="logloss"`, `n_jobs=-1`, `random_state=42` |

### 7.2 Train/test split
- **Ratio**: 80% train, 20% test.
- **Random state**: 42.
- **Stratification**: By `y`.

### 7.3 Normalization
- **Scaler**: None. Raw scores (0–3) and one-hot (0/1) used as-is.

### 7.4 Evaluation
- **5-fold StratifiedKFold**: `n_splits=5`, `shuffle=True`, `random_state=42`.
- **Metrics**: `accuracy`, `f1_macro` (per fold; mean ± std reported).
- **Final eval**: Accuracy, macro-F1, classification report, confusion matrix on held-out test set.

### 7.5 Model saving
- **Format**: `joblib`
- **Paths**: `models/mcq_model/mcq_logreg.joblib`, `mcq_rf.joblib`, `mcq_xgb.joblib`
- **Method**: `joblib.dump(model, model_path)`

---

## 8. INFERENCE / PREDICTION

No dedicated inference script exists for MCQ. The following describes how inference should work with the current pipeline:

### 8.1 Input
- **Format**: Same as training: 19-dimensional feature vector (q1..q15, depression, anxiety, ocd, adhd).
- **Source**: Either MongoDB document (run `build_feature_vector` + one-hot) or prebuilt CSV row.

### 8.2 Steps
1. Load model: `model = joblib.load("models/mcq_model/mcq_logreg.joblib")`
2. Ensure input has 19 features in order: `q1..q15`, `depression`, `anxiety`, `ocd`, `adhd`.
3. Missing values: Replace `-1` with training median (or precompute and store median at train time).
4. Scaling: None (not used in training).
5. Predict: `y_pred = model.predict(X)` or `y_proba = model.predict_proba(X)`.

### 8.3 Output
- **Class**: `0` or `1`.
- **Probability**: `model.predict_proba(X)[:, 1]` for class 1.
- **Confidence**: Same as probability for binary case.

---

## 9. FILE & FOLDER STRUCTURE

```
interview-ai-detection/
├── configs/
│   ├── mcq_config.yaml          # Data paths, training defaults (skeleton)
│   └── mcq_questions.json       # Disease → question configs
├── data/
│   └── processed/
│       └── mcq/
│           └── all_diseases_dataset.csv
├── models/
│   └── mcq_model/
│       ├── mcq_logreg.joblib
│       ├── mcq_rf.joblib
│       └── mcq_xgb.joblib
├── scripts/
│   └── train_mcq.py             # Skeleton; imports src.mcq.training.train (may not exist)
└── src/
    └── mcq/
        ├── __init__.py
        └── preprocessing/
            ├── fetch_from_mongo.py   # MongoDB → CSV
            └── preprocessing.ipynb   # Full ML pipeline
```

### Naming conventions
- Datasets: `all_diseases_dataset.csv` or `{disease}_dataset.csv`.
- Models: `mcq_{algo}.joblib` (e.g. `mcq_logreg.joblib`).
- Question columns: `q1`–`q15`.
- Disease one-hot: `depression`, `anxiety`, `ocd`, `adhd`.

---

## 10. CONFIGURATION & CONSTANTS

| Constant | Value | Location |
|----------|-------|----------|
| `DEFAULT_MONGO_HOST` | `clusterproject.ijackeq.mongodb.net` | `fetch_from_mongo.py` |
| `DEFAULT_DB_NAME` | `mental_health_db` | `fetch_from_mongo.py` |
| `DEFAULT_COLLECTION` | `fake_aditya_tests` | `fetch_from_mongo.py` |
| `test_size` | 0.2 | `preprocessing.ipynb` |
| `random_state` | 42 | `preprocessing.ipynb`, StratifiedKFold |
| `n_splits` | 5 | StratifiedKFold |
| `max_iter` (LogReg) | 1000 | `preprocessing.ipynb` |
| `n_estimators` (RF) | 200 | `preprocessing.ipynb` |
| `n_estimators` (XGB) | 300 | `preprocessing.ipynb` |
| `learning_rate` (XGB) | 0.05 | `preprocessing.ipynb` |
| `max_depth` (XGB) | 4 | `preprocessing.ipynb` |
| `subsample` (XGB) | 0.9 | `preprocessing.ipynb` |
| `colsample_bytree` (XGB) | 0.9 | `preprocessing.ipynb` |
| Missing value placeholder | -1 | `build_feature_vector` |
| Questions per disease | 15 | `mcq_questions.json` |
| Score range | 0–3 | Ordinal scale in config |

### Paths (from notebook `base_path`)
- `base_path`: `Path("../../..")` from `src/mcq/preprocessing/` → project root
- `data_dir`: `base_path / "data" / "processed" / "mcq"`
- `all_diseases_path`: `data_dir / "all_diseases_dataset.csv"`
- `models_dir`: `base_path / "models" / "mcq_model"`

---

## 11. KNOWN LIMITATIONS & ASSUMPTIONS

### Assumptions
1. MongoDB documents have `mcqCompleted == True`; others are excluded.
2. `mcqAnswers` entries have `questionId` and `score`; invalid entries are skipped.
3. Each document has exactly one `testType` (depression, anxiety, ocd, adhd).
4. `isRealPatientData` is present; missing → treated as false (label 0).
5. `mcq_questions.json` defines all diseases and their question IDs.
6. All diseases use 15 questions; column layout is q1..q15.

### Limitations
1. No dedicated inference module; inference must be implemented by the user.
2. `scripts/train_mcq.py` imports `src.mcq.training.train`, which may not exist; actual training is in `preprocessing.ipynb`.
3. No scaling; performance may depend on score distribution.
4. No class balancing; may be an issue if data is imbalanced.
5. One-hot disease columns give away test type; model may over-rely on them.

### Missing / corrupted data
- Empty `mcqAnswers`: All scores `-1`, imputed by median.
- Invalid `score`: Entry skipped, question remains `-1`.
- No MongoDB connection: `connect_mongo` raises `ValueError`.
- Unknown disease: `_load_disease_question_ids` raises `KeyError`.

---

*Document generated from codebase scan. Last updated: 2025-03-06.*
