# Text Modality — Complete Technical Documentation Report

**Project:** interview-ai-detection  
**Modality:** Text (mental health screening)  
**Last Updated:** March 2026  
**Scope:** Text-based model only (video, audio, MCQ excluded)

---

## 1. OVERVIEW

### 1.1 What This Modality Does

The text modality performs **multi-label binary classification** for four mental health conditions: depression, anxiety, OCD, and ADHD. It takes a patient's written interview responses (5 questions × 5 answers) and outputs a vector of 4 independent probabilities, one per disease.

### 1.2 What Problem It Solves

- **Screening:** Identifies whether a patient shows signs of depression, anxiety, OCD, and/or ADHD based on text responses.
- **Multi-label:** A patient can have high probability for multiple diseases simultaneously (e.g., depression + anxiety).
- **Indian context:** Dataset reflects Indian English, cultural references (UPSC, dowry, BPO, joint family, puja rituals, etc.), and socioeconomic diversity.

### 1.3 Input and Output

| Aspect | Details |
|--------|---------|
| **Input** | JSON file with `sample_id`, `label` (0/1), `disease`, and `responses` array of 5 Q&A pairs. Each response has `question_id`, `question`, `answer`. |
| **Input format** | `mental_health_training_dataset.json` in `data/raw/text/` |
| **Output (training)** | `text_model.pt` — PyTorch saved `state_dict` of `BertMultiDiseaseClassifier` |
| **Output (inference)** | `probabilities`: `(batch, 4)` tensor in [0, 1]; `logits`: `(batch, 4)`; `attention_weights`: dict of `(batch, seq_len)` per disease; `shared_features`: `(batch, 768)` |
| **Human-readable output** | `get_disease_probabilities()` returns list of dicts: `{"depression": 0.82, "anxiety": 0.61, "ocd": 0.12, "adhd": 0.34, "predictions": ["depression", "anxiety"]}` |

---

## 2. DEPENDENCIES & LIBRARIES

### 2.1 Libraries Used (Text Modality)

| Library | Version (min) | Purpose |
|---------|---------------|---------|
| `torch` | >=2.0.0 | PyTorch tensors, model, training loop |
| `torchvision` | >=0.15.0 | (Optional) — not used by text modality |
| `transformers` | >=4.35.0 | `BertModel`, `BertTokenizerFast` |
| `tokenizers` | >=0.15.0 | BERT tokenizer backend |
| `accelerate` | >=0.24.0 | HuggingFace utilities |
| `numpy` | >=1.24.0 | Arrays, random seed |
| `pandas` | >=2.0.0 | DataFrames, loading/saving splits |
| `scikit-learn` | >=1.3.0 | `train_test_split`, `TfidfVectorizer`, `cosine_similarity` |
| `matplotlib` | >=3.7.0 | Visualization (EDA) |
| `seaborn` | >=0.12.0 | Visualization (EDA) |
| `wordcloud` | >=1.9.0 | Word clouds (EDA) |
| `jupyter` | >=1.0.0 | Preprocessing notebook |
| `notebook` | >=7.0.0 | Preprocessing notebook |
| `tqdm` | >=4.65.0 | Progress bars |
| `pyyaml` | — | Config loading (`configs/text_config.yaml`) |
| `json` | stdlib | JSON parsing, config |
| `re` | stdlib | Regex for text cleaning |
| `collections` | stdlib | `Counter`, `defaultdict` |

### 2.2 Full pip Install Command

```bash
pip install torch>=2.0.0 torchvision>=0.15.0
pip install transformers>=4.35.0 tokenizers>=0.15.0 accelerate>=0.24.0
pip install numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0
pip install matplotlib>=3.7.0 seaborn>=0.12.0 wordcloud>=1.9.0
pip install jupyter>=1.0.0 notebook>=7.0.0 ipykernel>=6.25.0
pip install tqdm>=4.65.0 pyyaml packaging>=23.0
```

Or install from project root:

```bash
pip install -r requirements.txt
```

---

## 3. PREPROCESSING PIPELINE

### 3.1 Pipeline Overview

All preprocessing is implemented in `src/text/preprocessing/mental_health_preprocessing.ipynb`. There is no standalone `preprocess.py` script for text (the placeholder in `src/text/preprocessing/preprocess.py` is empty).

### 3.2 Processing Steps (in Order)

| Step | Function/Code | Description |
|------|---------------|-------------|
| 1 | Environment setup | `SEED=42`, `PROJECT_ROOT`, `PROCESSED_TEXT_DIR`, `TOKENIZER_DIR` |
| 2 | Load JSON | `json.load()` on `mental_health_training_dataset.json` |
| 3 | Flatten | Loop over `depression`, `anxiety`, `ocd`, `adhd`; build `records` list |
| 4 | Build DataFrame | `pd.DataFrame(records)` with columns: `sample_id`, `disease`, `label`, `answer_1`–`answer_5`, `question_1`–`question_5` |
| 5 | Validation | `df.isnull().sum()`, `df[answer_cols].notnull().all()`, `label in {0,1}`, `disease in DISEASES`, `sample_id.is_unique` |
| 6 | EDA | Class balance, text length, vocab, TF-IDF similarity, linguistic features |
| 7 | Text cleaning | `clean_text()` on each `answer_1`–`answer_5` |
| 8 | Text construction | `text_v1`, `text_v2`, `text_v3` |
| 9 | Split | `train_test_split`, stratified per disease, 70/15/15 |
| 10 | Tokenization | `BertTokenizerFast.from_pretrained("bert-base-uncased")` |
| 11 | Save | `train.pkl`, `val.pkl`, `test.pkl`, `*_encodings.pt`, `label_mappings.json`, `dataset_stats.json`, `tokenizer_config/` |

### 3.3 Key Functions and Parameters

#### `clean_text(text: str) -> str`

**Location:** `mental_health_preprocessing.ipynb` (Section 4)

**Steps:**
1. `text = str(text)` if not string; else `""`
2. `text = text.lower()`
3. Expand abbreviations: `dont`→`don't`, `cant`→`can't`, `wont`→`won't` (via `ABBREVIATIONS` dict)
4. `normalize_repeated_chars(text, max_repeats=2)` — collapse `soooo` → `soo`
5. `re.sub(r"[^a-z0-9\s\.,\?!']+", " ", text)` — remove special chars except `. , ? ! '`
6. `re.sub(r"\s+", " ", text).strip()` — collapse whitespace
7. If empty: `text = "[NO RESPONSE]"`

**Parameters:**
- `max_repeats=2` in `normalize_repeated_chars` — keeps at most 2 repeated chars

#### `normalize_repeated_chars(text: str, max_repeats: int = 2) -> str`

**Regex:** `r"(.)\1{max_repeats,}"` — matches any char repeated more than `max_repeats` times; replaces with `char * max_repeats`.

#### `build_qa_pairs(row) -> str`

**Location:** `mental_health_preprocessing.ipynb` (Section 5)

**Format:** `"Q: {question_1} A: {answer_1} [SEP] Q: {question_2} A: {answer_2} [SEP] ..."`

**SEP_TOKEN:** `" [SEP] "` (space-separated)

#### `tokenize(text: str) -> list[str]`

**Location:** `mental_health_preprocessing.ipynb` (Section 3d)

**Pattern:** `re.compile(r"\b\w+\b", re.UNICODE)` — extracts word tokens for EDA/vocab analysis only. Not used for BERT tokenization.

### 3.4 Input/Output Shapes

| Stage | Input | Output |
|-------|-------|--------|
| Raw JSON | `mental_health_training_dataset.json` | — |
| Flattened DataFrame | — | `(116, 13)` — `sample_id`, `disease`, `label`, `question_1`–`question_5`, `answer_1`–`answer_5` |
| After cleaning | `df[answer_cols]` | `df[answer_cols]` overwritten; `answers_combined_clean` added |
| Text construction | `df` | `text_v1`, `text_v2`, `text_v3` columns |
| Split | `df` | `df_train` (80), `df_val` (16), `df_test` (20) |
| Tokenization | `df_train[BEST_TEXT_COL].tolist()` | `enc_train`: `input_ids` (80×512), `attention_mask` (80×512), `token_type_ids` (80×512) |

### 3.5 Edge Cases Handled

| Edge Case | Handling |
|-----------|----------|
| Non-string answer | `clean_text(text)` converts to `""`; `[NO RESPONSE]` if empty |
| Missing/null responses | `fillna("")` before tokenization; validation checks for exactly 5 non-null answers |
| Empty string after cleaning | Replaced with `[NO RESPONSE]` |
| Unknown disease | `disease_to_id.get(disease, -1)` — target index -1 yields zeros for other diseases |
| Tokenizer missing `token_type_ids` | Training `MentalHealthDataset` provides zeros: `torch.zeros_like(item["input_ids"])` |
| Dataset not found | `FileNotFoundError` if `train.pkl` missing in `processed_dir` |

---

## 4. FEATURE EXTRACTION

**Note:** The text modality does **not** use traditional feature extraction (e.g., MFCCs, hand-crafted features). Instead, it uses BERT tokenization and learned representations.

### 4.1 Tokenization (BERT)

| Parameter | Value | Source |
|-----------|-------|--------|
| Tokenizer | `bert-base-uncased` | `BertTokenizerFast.from_pretrained("bert-base-uncased")` |
| `max_length` | 512 | `recommended_max_length` from `dataset_stats.json` (based on QA pairs) |
| `padding` | `"max_length"` | Padding to `max_length` |
| `truncation` | `True` | Truncate sequences exceeding `max_length` |
| `add_special_tokens` | `True` | [CLS], [SEP] added |
| `return_tensors` | `"pt"` | PyTorch tensors |

### 4.2 EDA Features (Not Used for Training)

| Feature | Computation | Library |
|---------|-------------|---------|
| Word count | `len(str(x).split())` | Python |
| Char count | `len(str(x))` | Python |
| Vocabulary | `re.compile(r"\b\w+\b").findall(text)` | `re` |
| TF-IDF | `TfidfVectorizer(max_features=5000)` | `sklearn` |
| Cosine similarity | `cosine_similarity(mat)` | `sklearn.metrics.pairwise` |
| Indian markers | Count of `umm`, `hmm`, `only`, `itself`, `na`, `no`, `actually`, `basically` | Custom |
| First-person pronouns | Count of `i`, `me`, `my`, `myself` | Custom |
| Negative words | Count of `hopeless`, `useless`, `tired`, etc. | Custom |
| Hedging words | Count of `maybe`, `perhaps`, `sometimes`, etc. | Custom |

### 4.3 Final Model Input

| Column | Type | Shape | Value Range |
|--------|------|-------|-------------|
| `input_ids` | `torch.LongTensor` | `(batch, 512)` | 0–30522 (BERT vocab) |
| `attention_mask` | `torch.LongTensor` | `(batch, 512)` | 0 or 1 |
| `token_type_ids` | `torch.LongTensor` | `(batch, 512)` | 0 (single segment) |

**Total features:** 512 tokens per sample (BERT input dimension).

---

## 5. DATASET CREATION

### 5.1 Raw File Structure

**Path:** `data/raw/text/mental_health_training_dataset.json`

**Structure:**
```json
{
  "metadata": {
    "description": "...",
    "diseases": ["depression", "anxiety", "ocd", "adhd"],
    "label_schema": {"1": "affected / positive", "0": "not affected / negative"},
    "total_samples": 116,
    "per_disease_counts": {...},
    "version": "v4"
  },
  "depression": [
    {
      "sample_id": "DEP_001",
      "label": 1,
      "disease": "depression",
      "responses": [
        {"question_id": 1, "question": "...", "answer": "..."},
        ...
      ]
    }
  ],
  "anxiety": [...],
  "ocd": [...],
  "adhd": [...]
}
```

### 5.2 Folder Structure Expected

```
data/
  raw/
    text/
      mental_health_training_dataset.json   # or mental_health_training_dataset_v4.json
  processed/
    text/   # or src/text/preprocessing/data/processed/text/ (notebook output)
      train.pkl
      val.pkl
      test.pkl
      train_encodings.pt
      val_encodings.pt
      test_encodings.pt
      label_mappings.json
      dataset_stats.json
      tokenizer_config/
        tokenizer_config.json
        tokenizer.json
```

### 5.3 Aggregation Per Sample

- **No aggregation** — each sample is one row per disease (one sample = one patient's 5 Q&A responses for one disease).
- **Label:** Binary 0 or 1 per sample (per disease).

### 5.4 Final CSV/DataFrame Structure

**DataFrame columns (after preprocessing):**

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | str | e.g. `DEP_001`, `ANX_013` |
| `disease` | str | `depression`, `anxiety`, `ocd`, `adhd` |
| `label` | int | 0 or 1 |
| `answer_1` | str | Cleaned answer to question 1 |
| `answer_2` | str | Cleaned answer to question 2 |
| `answer_3` | str | Cleaned answer to question 3 |
| `answer_4` | str | Cleaned answer to question 4 |
| `answer_5` | str | Cleaned answer to question 5 |
| `question_1` | str | Question 1 |
| `question_2` | str | Question 2 |
| `question_3` | str | Question 3 |
| `question_4` | str | Question 4 |
| `question_5` | str | Question 5 |
| `text_v1` | str | `answer_1 [SEP] answer_2 [SEP] ... answer_5` |
| `text_v2` | str | `Q: q1 A: a1 [SEP] Q: q2 A: a2 [SEP] ...` (used for training) |
| `text_v3` | str | `answer_1 answer_2 ... answer_5` (space-joined) |

### 5.5 Normal vs Depressed

- **Per-disease:** Each sample is labeled for one disease only (0 or 1).
- **No separate CSVs** — single DataFrame with `disease` and `label` columns.
- **Combined:** All 4 diseases flattened into one DataFrame; rows are independent per disease.

---

## 6. CLASS BALANCING

### 6.1 Technique Used

**No explicit balancing** (e.g., SMOTE, undersampling) is applied. **Class imbalance is handled via `pos_weight`** in `MultiDiseaseLoss`:

- `pos_weight[d] = num_negative / num_positive` for disease `d`
- Computed from `dataset_stats.json` → `label_counts_per_disease`

### 6.2 Class Distribution (Before/After)

| Disease | Label 0 (not affected) | Label 1 (affected) | Minority % |
|---------|------------------------|--------------------|------------|
| depression | 23 | 27 | 46% |
| anxiety | 9 | 13 | 41% |
| ocd | 9 | 13 | 41% |
| adhd | 9 | 13 | 41% |
| **Overall** | **50** | **66** | **43%** |

### 6.3 pos_weight Values

From `compute_pos_weights(dataset_stats.json)`:

- depression: `23/27 ≈ 0.85`
- anxiety: `9/13 ≈ 0.69`
- ocd: `9/13 ≈ 0.69`
- adhd: `9/13 ≈ 0.69`

### 6.4 Random State

- `SEED = 42` (notebook)
- `SKLEARN_RANDOM_STATE = 42` (splits)
- `cfg.seed = 42` (training)

---

## 7. MODEL TRAINING

### 7.1 Model Architecture

**Model:** `BertMultiDiseaseClassifier` (`src/model.py`)

| Component | Configuration |
|-----------|----------------|
| Base | `BertModel.from_pretrained("bert-base-uncased")` |
| Frozen | Embeddings + first 8 encoder layers |
| Trainable | Last 4 encoder layers + shared extractor + disease heads |
| Shared extractor | `Dropout(0.3) → Linear(768→768) → GELU → LayerNorm → Dropout(0.3)` |
| Disease heads | 4× `DiseaseAttentionHead` (8 heads, Q/K/V, attention over sequence, pooled → logit) |
| Output | Sigmoid → 4 probabilities (multi-label) |

### 7.2 Train/Val/Test Split

| Split | Ratio | Count | Random State |
|-------|-------|-------|--------------|
| Train | 70% | 80 | 42 |
| Val | 15% | 16 | 42 |
| Test | 15% | 20 | 42 |

### 7.3 Normalization

- **No feature scaling** — BERT uses raw token IDs; no StandardScaler/MinMaxScaler.
- Tokenization is deterministic (same tokenizer, same preprocessing).

### 7.4 Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `bert_model_name` | `bert-base-uncased` | `ModelConfig` |
| `num_diseases` | 4 | `ModelConfig` |
| `dropout_rate` | 0.3 | `ModelConfig` |
| `freeze_bert_layers` | 8 | `ModelConfig` |
| `use_disease_attention` | True | `ModelConfig` |
| `hidden_size` | 768 | `ModelConfig` |
| `num_attention_heads` | 8 | `ModelConfig` |
| `batch_size` | 16 | `TextTrainConfig` |

| Parameter | Value | Source |
|-----------|-------|--------|
| `learning_rate` | 2e-5 | `TextTrainConfig` |
| `num_epochs` | 3 | `TextTrainConfig` |
| `optimizer` | AdamW | `train.py` |

### 7.5 Evaluation Metrics

- **Training:** Average loss per epoch (BCE with logits, weighted by `pos_weight`).
- **Validation:** Average validation loss per epoch.
- **Best model:** Saved when validation loss improves.

**No explicit accuracy/F1/confusion matrix** in the current training loop.

### 7.6 Model Saving

- **Format:** `torch.save(model.state_dict(), path)`
- **Path:** `checkpoints/text/text_model.pt` (or `--checkpoints-dir` value)
- **Content:** PyTorch state dict only (weights). Architecture must be reconstructed from `ModelConfig` when loading.

---

## 8. INFERENCE / PREDICTION

### 8.1 Step-by-Step (New Sample)

1. **Load model:** `BertMultiDiseaseClassifier` + `model.load_state_dict(torch.load(path))`
2. **Load tokenizer:** `BertTokenizerFast.from_pretrained("bert-base-uncased")` or from `tokenizer_config/`
3. **Input format:** Dict or list of 5 Q&A pairs: `[{"question": "...", "answer": "..."}, ...]`
4. **Preprocessing:** Apply `clean_text()` to each answer; build `text_v2` (QA pairs with `[SEP]`.
5. **Tokenization:** `tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")`
6. **Forward:** `model(input_ids, attention_mask, token_type_ids=None)`
7. **Output:** `outputs["probabilities"]` → `(batch, 4)` in [0, 1]
8. **Human-readable:** `model.get_disease_probabilities(probabilities, threshold=0.5)`

### 8.2 Exact Input Format Expected

**Option A — Dict:**
```python
{
  "question_1": "...", "answer_1": "...",
  "question_2": "...", "answer_2": "...",
  ...
  "question_5": "...", "answer_5": "..."
}
```

**Option B — List of dicts:**
```python
[
  {"question": "...", "answer": "..."},
  {"question": "...", "answer": "..."},
  ...
]
```

### 8.3 Preprocessing at Inference

- Same `clean_text()` as training: lowercase, expand abbreviations, normalize repeats, remove special chars, strip.
- Same `build_qa_pairs()` format: `Q: {q} A: {a} [SEP] ...`
- Same tokenizer: `bert-base-uncased`, `max_length=512`, `padding="max_length"`, `truncation=True`.

### 8.4 Scaling

- No scaling applied.

### 8.5 Output Format

| Type | Format |
|------|--------|
| `probabilities` | `torch.Tensor` `(batch, 4)` — values in [0, 1] |
| `logits` | `torch.Tensor` `(batch, 4)` |
| `get_disease_probabilities()` | List of dicts: `{"depression": 0.82, "anxiety": 0.61, "ocd": 0.12, "adhd": 0.34, "predictions": ["depression", "anxiety"]}` |
| `predict(probabilities, threshold=0.5)` | Returns `(binary, human_readable)` — binary `(batch, 4)`, human_readable list of dicts |

### 8.6 Standalone Inference Script

**There is no dedicated inference script** (`src/text/inference/predict.py` does not exist). Inference is done via model methods; a script would need to be implemented separately.

---

## 9. FILE & FOLDER STRUCTURE

### 9.1 Text Modality Files

| File | Purpose |
|------|---------|
| `src/model.py` | `DiseaseAttentionHead`, `BertMultiDiseaseClassifier`, `ModelConfig`, `MultiDiseaseLoss`, `build_model`, `compute_pos_weights` |
| `src/text/training/train.py` | `TextTrainConfig`, `MentalHealthDataset`, `_load_processed_artifacts`, `train_text` |
| `src/text/preprocessing/mental_health_preprocessing.ipynb` | Full preprocessing pipeline (EDA, cleaning, split, tokenization, save) |
| `scripts/train_text.py` | CLI entry point: `--config`, `--checkpoints-dir` |
| `configs/text_config.yaml` | `processed_dir`, `raw_dir`, `seed`, `batch_size`, `num_epochs`, `learning_rate` |

### 9.2 Folders

| Folder | Contents |
|--------|----------|
| `data/raw/text/` | `mental_health_training_dataset.json` |
| `data/processed/text/` | Intended by config; actual output may be `src/text/preprocessing/data/processed/text/` |
| `src/text/preprocessing/data/processed/text/` | `train.pkl`, `val.pkl`, `test.pkl`, `*_encodings.pt`, `label_mappings.json`, `dataset_stats.json`, `tokenizer_config/` |
| `checkpoints/text/` | `text_model.pt` (saved after training) |

### 9.3 Naming Conventions

- `sample_id`: `{DISEASE_PREFIX}_{INDEX}` — e.g. `DEP_001`, `ANX_013`, `OCD_012`, `ADHD_022`
- Disease names: lowercase — `depression`, `anxiety`, `ocd`, `adhd`
- Text versions: `text_v1`, `text_v2`, `text_v3`
- Encodings: `train_encodings.pt`, `val_encodings.pt`, `test_encodings.pt`

---

## 10. CONFIGURATION & CONSTANTS

### 10.1 Config File (`configs/text_config.yaml`)

```yaml
modality: text
data:
  raw_dir: data/raw/text
  processed_dir: data/processed/text
training:
  seed: 42
  batch_size: 16
  num_epochs: 3
  learning_rate: 0.00002
model:
  name: baseline_text_transformer
  pretrained_model: distilbert-base-uncased
  max_length: 256
  num_classes: 2
```

**Note:** The text modality uses `bert-base-uncased` and `ModelConfig` defaults, not the YAML `model` section. `processed_dir` must point to where preprocessing actually wrote files (e.g. `src/text/preprocessing/data/processed/text`).

### 10.2 Hardcoded Constants

| Location | Constant | Value |
|----------|----------|-------|
| Notebook | `SEED` | 42 |
| Notebook | `SKLEARN_RANDOM_STATE` | 42 |
| Notebook | `DISEASES` | `["depression", "anxiety", "ocd", "adhd"]` |
| Notebook | `BEST_TEXT_COL` | `"text_v2"` |
| Notebook | `SEP_TOKEN` | `" [SEP] "` |
| Notebook | `ABBREVIATIONS` | `{"dont": "don't", "cant": "can't", "wont": "won't"}` |
| Notebook | `max_repeats` | 2 |
| Notebook | `TOKENIZER_NAME` | `"bert-base-uncased"` |
| Notebook | `RECOMMENDED_MAX_LENGTH` | 256 / 384 / 512 (from QA stats) |
| model.py | `bert_model_name` | `"bert-base-uncased"` |
| model.py | `hidden_size` | 768 |
| model.py | `num_attention_heads` | 8 |
| model.py | `dropout_rate` | 0.3 |
| model.py | `freeze_bert_layers` | 8 |
| model.py | `threshold` | 0.5 |
| train.py | `ModelConfig()` | Uses defaults (not YAML model section) |

---

## 11. KNOWN LIMITATIONS & ASSUMPTIONS

### 11.1 Assumptions

- Each sample has exactly 5 Q&A responses.
- Labels are binary (0 or 1) per disease.
- Disease names are exactly `depression`, `anxiety`, `ocd`, `adhd`.
- Text is in Indian English; cultural references are expected.
- `sample_id` is unique across the dataset.
- Preprocessed artifacts exist in `processed_dir` before training.

### 11.2 Known Limitations

- **No standalone inference script** — inference must be implemented manually.
- **Path mismatch:** Config `processed_dir` may not match notebook output location.
- **Single-disease labels:** Each sample is labeled for one disease; training treats other diseases as 0 (not "unknown").
- **Small dataset:** 116 samples total; risk of overfitting.
- **No explicit accuracy/F1/confusion matrix** in training loop.

### 11.3 Missing or Corrupted Data

| Scenario | Behavior |
|----------|----------|
| `train.pkl` missing | `FileNotFoundError` in `_load_processed_artifacts` |
| Null in answer column | Validation fails; `fillna("")` used elsewhere |
| Empty answer after cleaning | Replaced with `[NO RESPONSE]` |
| Unknown disease in row | `disease_to_id.get(disease, -1)` → target zeros for that sample |
| Corrupted JSON | `json.load()` raises; no recovery |

---

## APPENDIX: Quick Reference Commands

```bash
# Set project root for Python imports
set PYTHONPATH=%cd%

# Run preprocessing (execute notebook cells manually or via Jupyter)

# Train text model
python scripts/train_text.py --config configs/text_config.yaml --checkpoints-dir checkpoints/text
```

---

*End of Text Modality Technical Report*
