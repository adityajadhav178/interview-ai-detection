## Interview AI Detection (Multimodal)

Research-oriented Python project for detecting mental health conditions from **four modalities**:
- **MCQ** questionnaire data
- **Audio** features extracted from interview responses
- **Text** transcripts generated from speech
- **Video** facial behaviour analysis

The repository is structured so **four developers can work independently** on one modality under `src/{audio,video,text,mcq}/` with consistent `preprocessing/`, `feature_extraction/` (or `feature_engineering/`), `training/`, and `inference/` subpackages. Fusion lives in `src/fusion/`.

## Repository layout

- **`data/`**: raw + processed datasets by modality; `annotations/` for labels, segments, alignments, etc.
- **`src/`**: modality pipelines + fusion + evaluation.
- **`models/`**: saved model artifacts (by modality).
- **`checkpoints/`**: training checkpoints (by modality).
- **`configs/`**: YAML configs per modality.
- **`scripts/`**: entrypoint scripts for training.
- **`evaluation/`**: evaluation utilities and results.
- **`notebooks/`**: exploratory experiments (one per modality).

## Quickstart

Create an environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run a modality training script:

```bash
python scripts/train_audio.py --config configs/audio_config.yaml
```

Run the orchestrated pipeline (skeleton):

```bash
python main_pipeline.py --config-dir configs
```

## Conventions (recommended)

- **Data immutability**: treat `data/raw/` as read-only; write derived artifacts to `data/processed/`.
- **Reproducibility**: keep experiment parameters in YAML configs; log runs + random seeds.
- **Separation of concerns**:
  - `preprocessing/`: cleaning, resampling, segmentation, normalization
  - `feature_extraction/` / `feature_engineering/`: transform into model-ready tensors/features
  - `training/`: training loops, model definitions, losses, optimizers
  - `inference/`: loading artifacts and producing predictions
- **Fusion**:
  - **Early fusion**: concatenate/merge representations before a unified model
  - **Late fusion**: combine modality-specific predictions (e.g., stacking, weighted averaging)

## Next steps

- Connect each modality pipeline to real datasets under `data/raw/*`.
- Implement modality-specific feature extraction and training.
- Add experiment tracking (e.g., MLflow/W&B) and tests for data contracts.

