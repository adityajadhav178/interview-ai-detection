#!/usr/bin/env python3
"""
Step 3: Train Audio Models
Path: pipeline_steps/step3_train_models.py

Trains machine learning models and evaluates performance.
- Applies SMOTE for class balancing
- Trains multiple models (RF, XGBoost, SVM, LightGBM)
- 5-Fold Stratified Cross-Validation
- Selects best model by F1-weighted metric
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.training import training_pipeline

def main():
    """Run model training step."""
    # Get project root directory (parent of pipeline_steps)
    project_root = Path(__file__).parent.parent
    feature_csv = project_root / "data/final_features/features.csv"
    model_dir = project_root / "models/audio_global"
    eval_dir = project_root / "evaluation/audio_global"
    
    print("\n" + "="*80)
    print("STEP 3: MODEL TRAINING")
    print("="*80)
    print(f"\nInput features: {feature_csv}")
    print(f"Model output:   {model_dir}")
    print(f"Evaluation:     {eval_dir}")
    print("\nTraining process:")
    print("  1. Load feature dataset")
    print("  2. Apply SMOTE for class balancing")
    print("  3. Split into train/val/test (70/15/15)")
    print("  4. Train 4 models with 5-Fold CV")
    print("  5. Evaluate on test set")
    print("  6. Select best model")
    
    if not Path(feature_csv).exists():
        print(f"\n❌ Feature CSV not found: {feature_csv}")
        print(f"   Run step2_extract_features.py first")
        return False
    
    try:
        training_pipeline(
            feature_csv=str(feature_csv),
            model_dir=str(model_dir),
            eval_dir=str(eval_dir),
            balance=True
        )
        
        print("\n✓ Model training complete!")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
