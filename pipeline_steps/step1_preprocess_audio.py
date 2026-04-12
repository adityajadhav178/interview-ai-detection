#!/usr/bin/env python3
"""
Step 1: Preprocess Audio - Segment and Normalize
Path: pipeline_steps/step1_preprocess_audio.py

Segments variable-length audio files to normalized length (10-25 seconds).
- Pads short files with silence
- Keeps medium files as-is
- Segments long files with overlap
"""

from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.preprocessing import segment_audio_files

def main():
    """Run audio preprocessing step."""
    # Get project root directory (parent of pipeline_steps)
    project_root = Path(__file__).parent.parent
    raw_audio_dir = project_root / "data/raw/audio"
    segmented_audio_dir = project_root / "data/processed/audio/segmented_audio"
    
    print("\n" + "="*80)
    print("STEP 1: AUDIO PREPROCESSING - SEGMENTATION & NORMALIZATION")
    print("="*80)
    print(f"\nInput:  {raw_audio_dir}")
    print(f"Output: {segmented_audio_dir}")
    print("\nProcessing:")
    print("  • Pad short audio (< 10s) with silence")
    print("  • Keep medium audio (10-25s) as-is")
    print("  • Segment long audio (> 25s) with 50% overlap")
    
    if not Path(raw_audio_dir).exists():
        print(f"\n❌ Input directory not found: {raw_audio_dir}")
        return False
    
    try:
        stats = segment_audio_files(
            input_dir=raw_audio_dir,
            output_dir=segmented_audio_dir,
            target_length=25,
            pad_to_length=10
        )
        
        print("\n✓ Audio preprocessing complete!")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
