import numpy as np
import librosa
import noisereduce as nr
import webrtcvad
import scipy.signal

def run_vad_on_audio(audio_array, sample_rate, aggressiveness=3):
    """
    Applies Voice Activity Detection (VAD) to remove silence.
    Uses webrtcvad which requires 16-bit PCM (int16).
    """
    # webrtcvad needs 10, 20, or 30 ms frames
    frame_duration_ms = 30
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
    
    vad = webrtcvad.Vad(aggressiveness)
    
    # Convert audio (float32, range [-1.0, 1.0]) to int16
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    # Pad audio so it's a multiple of frame_size
    padding_size = frame_size - (len(audio_int16) % frame_size)
    if padding_size < frame_size:
        audio_int16 = np.pad(audio_int16, (0, padding_size), mode='constant')
        
    voiced_frames = []
    
    # Iterate over frames
    for i in range(0, len(audio_int16), frame_size):
        frame = audio_int16[i:i + frame_size]
        is_speech = vad.is_speech(frame.tobytes(), sample_rate)
        
        if is_speech:
            voiced_frames.append(frame)
            
    if not voiced_frames:
        print("Warning: No speech detected in audio file. Returning original audio.")
        return audio_array
        
    # Concatenate voiced frames and convert back to float32
    voiced_audio_int16 = np.concatenate(voiced_frames)
    voiced_audio = voiced_audio_int16.astype(np.float32) / 32767.0
    
    return voiced_audio

def normalize_rms(audio_array, target_rms=0.1):
    """
    Applies RMS normalization to bring audio amplitude to a consistent level.
    """
    rms = np.sqrt(np.mean(audio_array**2))
    if rms > 0:
        return audio_array * (target_rms / rms)
    return audio_array

def apply_preemphasis(audio_array, pre_emphasis=0.97):
    """
    Applies a pre-emphasis filter to amplify high frequencies.
    y(t) = x(t) - alpha * x(t-1)
    """
    return np.append(audio_array[0], audio_array[1:] - pre_emphasis * audio_array[:-1])

def preprocess_audio(file_path):
    """
    Full preprocessing pipeline for mental health audio detection:
    1. Load using librosa (16kHz, mono)
    2. Reduce background noise
    3. Voice Activity Detection (remove silence)
    4. RMS Amplitude Normalization
    5. Pre-emphasis filtering
    """
    # 1. Load audio (resamples to 16kHz, converts to mono automatically)
    sample_rate = 16000
    try:
        y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None
        
    if len(y) == 0:
        return np.array([]), sample_rate
        
    # 2. Noise reduction
    y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
    
    # 3. Voice Activity Detection (remove silence)
    y_vad = run_vad_on_audio(y_denoised, sr)
    
    # 4. RMS Amplitude Normalization
    y_norm = normalize_rms(y_vad)
    
    # 5. Pre-emphasis filter
    y_clean = apply_preemphasis(y_norm)
    
    return y_clean, sr

# Example usage:
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Processing {test_file}...")
        clean_audio, sr = preprocess_audio(test_file)
        if clean_audio is not None:
            print(f"Success! Cleaned audio shape: {clean_audio.shape}, Sample Rate: {sr}")
            # Optional: save to check outputs
            # import soundfile as sf
            # sf.write("cleaned_output.wav", clean_audio, sr)
    else:
        print("Usage: python audio_cleaner.py <path_to_wav_file>")
