import os
import argparse
from pathlib import Path
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def extract_audio_from_video(input_dir: str, output_dir: str):
    """
    Extracts audio from video files (.mp4), converts to mono channel, 
    16kHz sample rate, and saves as .wav files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .mp4 files (can add other extensions if needed)
    video_files = list(input_path.rglob("*.mp4"))
    
    if not video_files:
        print(f"No .mp4 files found in {input_dir}")
        return
        
    print(f"Found {len(video_files)} video files. Extracting audio...")
    
    for video_file in tqdm(video_files, desc="Extracting Audio"):
        try:
            # Recreate directory structure if using recursive glob
            rel_path = video_file.relative_to(input_path)
            out_file = output_path / rel_path.with_suffix('.wav')
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load video
            video = VideoFileClip(str(video_file))
            
            # Extract basic audio first
            if video.audio is None:
                print(f"\nWarning: No audio track found in {video_file.name}")
                continue
                
            # Write audio: 16kHz, 1 channel (mono), 16-bit PCM
            video.audio.write_audiofile(
                str(out_file), 
                fps=16000, 
                nbytes=2, 
                codec='pcm_s16le',
                ffmpeg_params=["-ac", "1"], # Force mono
                verbose=False, 
                logger=None
            )
            
            # Close the clip to free memory
            video.close()
            
        except Exception as e:
            print(f"\nError processing {video_file.name}: {e}")
            
    print(f"\nAudio extraction complete! Saved files to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 16kHz mono audio from video files")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input video files (e.g., data/raw/videos)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted .wav files (e.g., data/raw/audio)")
    
    args = parser.parse_args()
    
    extract_audio_from_video(args.input_dir, args.output_dir)
