import re
import spacy
from transformers import pipeline
from typing import List, Dict, Tuple
import pandas as pd

class ScriptParser:
    """Parses and analyzes script content for dialogue, emotions, and structure"""
    
    def __init__(self, config: dict):
        self.config = config
        # Load pre-trained models
        self.emotion_analyzer = pipeline(
            "text-classification",
            model=config['models']['emotion_text'],
            return_all_scores=True
        )
        self.nlp = spacy.load("en_core_web_sm")
        
    def parse_script_file(self, script_path: str) -> List[Dict]:
        """Parse script file and extract dialogue with metadata"""
        with open(script_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Basic script parsing (can be enhanced for different formats)
        dialogue_pattern = r'([A-Z\s]+):\s*(.+?)(?=\n[A-Z\s]+:|$)'
        matches = re.findall(dialogue_pattern, content, re.DOTALL)
        
        dialogue_data = []
        for i, (speaker, text) in enumerate(matches):
            dialogue_data.append({
                'line_number': i + 1,
                'speaker': speaker.strip(),
                'text': text.strip(),
                'word_count': len(text.split()),
                'estimated_duration': self._estimate_speech_duration(text.strip())
            })
            
        return dialogue_data
    
    def analyze_emotions(self, dialogue_data: List[Dict]) -> List[Dict]:
        """Analyze emotional content of dialogue"""
        for item in dialogue_data:
            text = item['text']
            
            # Get emotion predictions
            emotions = self.emotion_analyzer(text)[0]
            
            # Add top emotion and confidence
            top_emotion = max(emotions, key=lambda x: x['score'])
            item['emotion'] = top_emotion['label']
            item['emotion_confidence'] = top_emotion['score']
            item['all_emotions'] = {e['label']: e['score'] for e in emotions}
            
        return dialogue_data
    
    def detect_emotional_beats(self, dialogue_data: List[Dict]) -> List[float]:
        """Identify significant emotional changes in the script"""
        emotional_beats = []
        prev_emotion_scores = None
        
        for item in dialogue_data:
            current_emotions = item['all_emotions']
            
            if prev_emotion_scores is not None:
                # Calculate emotional distance
                emotion_change = self._calculate_emotion_change(
                    prev_emotion_scores, current_emotions
                )
                
                if emotion_change > self.config['suggestions']['emotion_change_threshold']:
                    emotional_beats.append(item['estimated_start_time'])
                    
            prev_emotion_scores = current_emotions
            
        return emotional_beats
    
    def _estimate_speech_duration(self, text: str) -> float:
        """Estimate speech duration based on word count"""
        # Average speaking rate: ~150 words per minute
        words = len(text.split())
        return (words / 150) * 60  # Convert to seconds
    
    def _calculate_emotion_change(self, emotions1: Dict, emotions2: Dict) -> float:
        """Calculate the magnitude of emotional change between two states"""
        change = 0
        for emotion in emotions1:
            if emotion in emotions2:
                change += abs(emotions1[emotion] - emotions2[emotion])
        return change / len(emotions1)
