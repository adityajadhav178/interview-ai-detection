"""
Audio Processing Pipeline Steps
Path: pipeline_steps/__init__.py

Individual steps for audio processing pipeline:
- step1_preprocess_audio: Segment and normalize audio
- step2_extract_features: Extract acoustic features
- step3_train_models: Train and evaluate models
"""

__version__ = "1.0.0"

__all__ = [
    "run_step1_preprocess",
    "run_step2_extract_features",
    "run_step3_train_models",
]

from pathlib import Path

def run_step1_preprocess():
    """Run Step 1: Audio Preprocessing and Segmentation"""
    from .step1_preprocess_audio import main as step1_main
    return step1_main()

def run_step2_extract_features():
    """Run Step 2: Feature Extraction"""
    from .step2_extract_features import main as step2_main
    return step2_main()

def run_step3_train_models():
    """Run Step 3: Model Training"""
    from .step3_train_models import main as step3_main
    return step3_main()
