import librosa
import numpy as np
from transformers import pipeline
import moviepy.editor as mp
from typing import List, Dict, Tuple

class AudioAnalyzer:
    """Analyzes audio for speech emotion, speaker changes, and audio cues"""
    
    def __init__(self, config: dict):
        self.config = config
        # Load speech emotion recognition model
        self.speech_emotion = pipeline(
            "audio-classification",
            model=config['models']['emotion_speech']
        )
        
    def extract_audio_features(self, video_path: str) -> Dict:
        """Extract audio from video and analyze features"""
        # Extract audio using moviepy
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        
        # Save temporary audio file
        temp_audio_path = "temp_audio.wav"
        audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        
        # Load with librosa for analysis
        y, sr = librosa.load(temp_audio_path, sr=self.config['audio']['sample_rate'])
        
        # Extract various audio features
        features = {
            'audio_path': temp_audio_path,
            'duration': len(y) / sr,
            'sample_rate': sr,
            'rms_energy': librosa.feature.rms(y=y)[0],
            'spectral_centroids': librosa.feature.spectral_centroid(y=y, sr=sr)[0],
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(y)[0],
            'tempo': librosa.beat.tempo(y=y, sr=sr)[0]
        }
        
        # Clean up
        video.close()
        audio.close()
        
        return features
    
    def analyze_speech_emotion(self, audio_path: str, chunk_duration: float = 5.0) -> List[Dict]:
        """Analyze emotional content of speech in chunks"""
        y, sr = librosa.load(audio_path, sr=self.config['audio']['sample_rate'])
        
        chunk_samples = int(chunk_duration * sr)
        emotion_timeline = []
        
        for i in range(0, len(y), chunk_samples):
            chunk = y[i:i + chunk_samples]
            
            if len(chunk) < chunk_samples // 2:  # Skip very short chunks
                continue
                
            # Save chunk temporarily for emotion analysis
            chunk_path = f"temp_chunk_{i}.wav"
            librosa.output.write_wav(chunk_path, chunk, sr)
            
            try:
                # Analyze emotion
                emotions = self.speech_emotion(chunk_path)
                
                emotion_timeline.append({
                    'start_time': i / sr,
                    'end_time': (i + len(chunk)) / sr,
                    'emotion': emotions[0]['label'],
                    'confidence': emotions[0]['score'],
                    'all_emotions': emotions
                })
                
            except Exception as e:
                print(f"Error analyzing chunk {i}: {e}")
                continue
                
        return emotion_timeline
    
    def detect_speaker_changes(self, audio_path: str) -> List[float]:
        """Detect potential speaker changes using audio features"""
        y, sr = librosa.load(audio_path)
        
        # Extract MFCC features for speaker identification
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Simple speaker change detection using MFCC variance
        speaker_changes = [0.0]
        window_size = sr * 2  # 2-second windows
        
        for i in range(window_size, len(y) - window_size, window_size // 2):
            window1 = mfccs[:, max(0, i-window_size//sr):i//sr]
            window2 = mfccs[:, i//sr:(i+window_size)//sr]
            
            if window1.size > 0 and window2.size > 0:
                # Calculate similarity between windows
                similarity = np.corrcoef(
                    window1.flatten(), 
                    window2.flatten()
                )[0, 1]
                
                if similarity < 0.7:  # Threshold for speaker change
                    speaker_changes.append(i / sr)
                    
        return speaker_changes
