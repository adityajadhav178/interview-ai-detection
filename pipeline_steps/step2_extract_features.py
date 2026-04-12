#!/usr/bin/env python3
"""
Step 2: Extract Audio Features
Path: pipeline_steps/step2_extract_features.py

Extracts acoustic features from segmented audio.
- 20+ acoustic features per audio file
- Aggregates to 100 features per patient
- Creates CSV dataset ready for training
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.feature_extraction import build_feature_dataset

def main():
    """Run feature extraction step."""
    # Get project root directory (parent of pipeline_steps)
    project_root = Path(__file__).parent.parent
    segmented_audio_dir = project_root / "data/processed/audio/segmented_audio"
    output_csv = project_root / "data/final_features/features.csv"
    
    print("\n" + "="*80)
    print("STEP 2: FEATURE EXTRACTION")
    print("="*80)
    print(f"\nInput:  {segmented_audio_dir}")
    print(f"Output: {output_csv}")
    print("\nExtracting features:")
    print("  • Prosody features (pitch, contours)")
    print("  • Voice quality features (jitter, shimmer, HNR)")
    print("  • Temporal features (speaking rate, voice activity)")
    print("  • Spectral features (MFCC, spectral centroids)")
    print("  • Energy features (intensity, loudness)")
    
    if not Path(segmented_audio_dir).exists():
        print(f"\n❌ Input directory not found: {segmented_audio_dir}")
        print(f"   Run step1_preprocess_audio.py first")
        return False
    
    try:
        stats = build_feature_dataset(
            input_dir=str(segmented_audio_dir),
            output_csv=str(output_csv)
        )
        
        print("\n✓ Feature extraction complete!")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
