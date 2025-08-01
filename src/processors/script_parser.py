import re
import spacy
from transformers import pipeline
from typing import List, Dict, Tuple, Optional
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ScriptParser:
    """
    Parses and analyzes script content for dialogue, emotions, structure, and timing.
    Supports multiple script formats and provides comprehensive text analysis.
    """
    
    def __init__(self, config: dict):
        """
        Initialize ScriptParser with configuration settings.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.script_config = config.get('script', {})
        
        # Load emotion analysis model
        emotion_model = config['models']['emotion_text']
        logger.info(f"Loading text emotion model: {emotion_model}")
        
        self.emotion_analyzer = pipeline(
            "text-classification",
            model=emotion_model,
            return_all_scores=True
        )
        
        # Load spaCy model for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            logger.error("spaCy English model not found. Please install with: python -m spacy download en_core_web_sm")
            raise
        
        # Script parsing patterns
        self.patterns = {
            'character_dialogue': r'^([A-Z\s]+):\s*(.+?)(?=\n[A-Z\s]+:|$)',
            'action_line': r'^\((.+?)\)$',
            'scene_header': r'^(FADE IN:|FADE OUT:|CUT TO:|INT\.|EXT\.)',
            'parenthetical': r'\(([^)]+)\)',
            'speaker_name': r'^([A-Z][A-Z\s]*[A-Z])$'
        }
        
    def parse_script_file(self, script_path: str) -> List[Dict]:
        """
        Parse script file and extract dialogue with metadata.
        Supports multiple script formats (.txt, .srt, .vtt).
        
        Args:
            script_path: Path to script file
            
        Returns:
            List of dialogue entries with metadata
        """
        logger.info(f"Parsing script file: {script_path}")
        
        try:
            # Determine file format
            file_extension = Path(script_path).suffix.lower()
            
            if file_extension == '.srt':
                return self._parse_srt_file(script_path)
            elif file_extension == '.vtt':
                return self._parse_vtt_file(script_path)
            else:
                return self._parse_text_script(script_path)
                
        except Exception as e:
            logger.error(f"Error parsing script file: {e}")
            raise
    
    def _parse_text_script(self, script_path: str) -> List[Dict]:
        """Parse standard text script format."""
        with open(script_path, 'r', encoding=self.script_config.get('encoding', 'utf-8')) as file:
            content = file.read()
        
        # Enhanced dialogue parsing
        dialogue_pattern = r'([A-Z\s]{2,}):\s*(.+?)(?=\n[A-Z\s]{2,}:|$)'
        matches = re.findall(dialogue_pattern, content, re.DOTALL | re.MULTILINE)
        
        dialogue_data = []
        cumulative_time = 0.0
        
        for i, (speaker, text) in enumerate(matches):
            speaker = speaker.strip()
            text = self._clean_dialogue_text(text.strip())
            
            if not text:  # Skip empty dialogue
                continue
            
            duration = self._estimate_speech_duration(text)
            
            dialogue_entry = {
                'line_number': i + 1,
                'speaker': speaker,
                'text': text,
                'word_count': len(text.split()),
                'character_count': len(text),
                'estimated_duration': duration,
                'estimated_start_time': cumulative_time,
                'estimated_end_time': cumulative_time + duration,
                'has_parenthetical': '(' in text and ')' in text,
                'is_action': speaker.upper() in ['ACTION', 'DESCRIPTION', 'SCENE'],
                'scene_type': self._determine_scene_type(text)
            }
            
            dialogue_data.append(dialogue_entry)
            cumulative_time += duration + 0.5  # Add small pause between lines
        
        logger.info(f"Parsed {len(dialogue_data)} dialogue entries")
        return dialogue_data
    
    def _parse_srt_file(self, script_path: str) -> List[Dict]:
        """Parse SRT subtitle format."""
        with open(script_path, 'r', encoding=self.script_config.get('encoding', 'utf-8')) as file:
            content = file.read()
        
        # SRT format parsing
        srt_pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\n\d+\n|$)'
        matches = re.findall(srt_pattern, content, re.DOTALL)
        
        dialogue_data = []
        
        for i, (seq_num, timestamp, text) in enumerate(matches):
            # Parse timestamp
            start_str, end_str = timestamp.split(' --> ')
            start_time = self._srt_time_to_seconds(start_str)
            end_time = self._srt_time_to_seconds(end_str)
            
            text = self._clean_dialogue_text(text.strip())
            
            dialogue_entry = {
                'line_number': int(seq_num),
                'speaker': 'UNKNOWN',  # SRT doesn't typically have speaker info
                'text': text,
                'word_count': len(text.split()),
                'character_count': len(text),
                'estimated_duration': end_time - start_time,
                'estimated_start_time': start_time,
                'estimated_end_time': end_time,
                'has_parenthetical': False,
                'is_action': False,
                'scene_type': 'dialogue'
            }
            
            dialogue_data.append(dialogue_entry)
        
        return dialogue_data
    
    def _parse_vtt_file(self, script_path: str) -> List[Dict]:
        """Parse WebVTT subtitle format."""
        with open(script_path, 'r', encoding=self.script_config.get('encoding', 'utf-8')) as file:
            content = file.read()
        
        # Skip WEBVTT header
        content = re.sub(r'^WEBVTT.*?\n\n', '', content, flags=re.MULTILINE)
        
        # VTT format parsing
        vtt_pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3})\n(.+?)(?=\n\n|\n\d{2}:|$)'
        matches = re.findall(vtt_pattern, content, re.DOTALL)
        
        dialogue_data = []
        
        for i, (timestamp, text) in enumerate(matches):
            # Parse timestamp
            start_str, end_str = timestamp.split(' --> ')
            start_time = self._vtt_time_to_seconds(start_str)
            end_time = self._vtt_time_to_seconds(end_str)
            
            text = self._clean_dialogue_text(text.strip())
            
            dialogue_entry = {
                'line_number': i + 1,
                'speaker': 'UNKNOWN',
                'text': text,
                'word_count': len(text.split()),
                'character_count': len(text),
                'estimated_duration': end_time - start_time,
                'estimated_start_time': start_time,
                'estimated_end_time': end_time,
                'has_parenthetical': False,
                'is_action': False,
                'scene_type': 'dialogue'
            }
            
            dialogue_data.append(dialogue_entry)
        
        return dialogue_data
    
    def analyze_emotions(self, dialogue_data: List[Dict]) -> List[Dict]:
        """
        Analyze emotional content of dialogue using transformer model.
        
        Args:
            dialogue_data: List of dialogue entries
            
        Returns:
            Dialogue data enhanced with emotion analysis
        """
        logger.info("Analyzing emotions in dialogue")
        
        for item in dialogue_data:
            text = item['text']
            
            if not text or item.get('is_action', False):
                # Skip empty text or action lines
                item.update({
                    'emotion': 'neutral',
                    'emotion_confidence': 0.5,
                    'all_emotions': {'neutral': 0.5}
                })
                continue
            
            try:
                # Get emotion predictions
                emotions = self.emotion_analyzer(text)
                
                if emotions and len(emotions) > 0:
                    if isinstance(emotions[0], list):
                        emotions = emotions[0]  # Handle nested structure
                    
                    # Find dominant emotion
                    top_emotion = max(emotions, key=lambda x: x['score'])
                    
                    item.update({
                        'emotion': top_emotion['label'],
                        'emotion_confidence': top_emotion['score'],
                        'all_emotions': {e['label']: e['score'] for e in emotions}
                    })
                else:
                    # Fallback
                    item.update({
                        'emotion': 'neutral',
                        'emotion_confidence': 0.5,
                        'all_emotions': {'neutral': 0.5}
                    })
                    
            except Exception as e:
                logger.warning(f"Error analyzing emotion for line {item['line_number']}: {e}")
                item.update({
                    'emotion': 'neutral',
                    'emotion_confidence': 0.5,
                    'all_emotions': {'neutral': 0.5}
                })
        
        logger.info("Completed emotion analysis")
        return dialogue_data
    
    def detect_emotional_beats(self, dialogue_data: List[Dict]) -> List[Dict]:
        """
        Identify significant emotional changes in the script.
        
        Args:
            dialogue_data: List of dialogue entries with emotion analysis
            
        Returns:
            List of emotional beat points with metadata
        """
        logger.info("Detecting emotional beats")
        
        emotional_beats = []
        prev_emotions = None
        threshold = self.config['suggestions']['emotion_change_threshold']
        
        for i, item in enumerate(dialogue_data):
            current_emotions = item.get('all_emotions', {})
            
            if prev_emotions is not None and current_emotions:
                # Calculate emotional distance
                emotion_change = self._calculate_emotion_change(prev_emotions, current_emotions)
                
                if emotion_change > threshold:
                    beat = {
                        'timestamp': item.get('estimated_start_time', 0.0),
                        'line_number': item['line_number'],
                        'speaker': item['speaker'],
                        'emotion_from': max(prev_emotions, key=prev_emotions.get),
                        'emotion_to': item['emotion'],
                        'change_magnitude': emotion_change,
                        'context': item['text'][:100] + '...' if len(item['text']) > 100 else item['text']
                    }
                    emotional_beats.append(beat)
                    
            prev_emotions = current_emotions
        
        logger.info(f"Detected {len(emotional_beats)} emotional beats")
        return emotional_beats
    
    def extract_character_analysis(self, dialogue_data: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze character-specific patterns and emotions.
        
        Args:
            dialogue_data: List of dialogue entries
            
        Returns:
            Dictionary mapping character names to their analysis
        """
        character_analysis = {}
        
        for item in dialogue_data:
            speaker = item['speaker']
            
            if speaker not in character_analysis:
                character_analysis[speaker] = {
                    'total_lines': 0,
                    'total_words': 0,
                    'emotions': {},
                    'average_line_length': 0,
                    'dominant_emotion': 'neutral',
                    'emotional_range': 0
                }
            
            char_data = character_analysis[speaker]
            char_data['total_lines'] += 1
            char_data['total_words'] += item['word_count']
            
            # Track emotions
            emotion = item.get('emotion', 'neutral')
            char_data['emotions'][emotion] = char_data['emotions'].get(emotion, 0) + 1
        
        # Calculate derived metrics
        for speaker, data in character_analysis.items():
            if data['total_lines'] > 0:
                data['average_line_length'] = data['total_words'] / data['total_lines']
                data['dominant_emotion'] = max(data['emotions'], key=data['emotions'].get)
                data['emotional_range'] = len(data['emotions'])
        
        return character_analysis
    
    def _clean_dialogue_text(self, text: str) -> str:
        """Clean and normalize dialogue text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove stage directions in parentheses (optional)
        # text = re.sub(r'\([^)]*\)', '', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _estimate_speech_duration(self, text: str) -> float:
        """
        Estimate speech duration based on word count and complexity.
        
        Args:
            text: Text to analyze
            
        Returns:
            Estimated duration in seconds
        """
        words = len(text.split())
        
        # Base speaking rate: ~150 words per minute
        base_rate = 150
        
        # Adjust for punctuation (pauses)
        pause_chars = text.count('.') + text.count(',') + text.count('!') + text.count('?')
        pause_time = pause_chars * 0.3
        
        # Adjust for complexity (longer words = slower speech)
        avg_word_length = sum(len(word) for word in text.split()) / max(words, 1)
        complexity_factor = 1.0 + (avg_word_length - 5) * 0.02
        
        duration = (words / base_rate) * 60 * complexity_factor + pause_time
        return max(duration, 0.5)  # Minimum 0.5 seconds
    
    def _srt_time_to_seconds(self, time_str: str) -> float:
        """Convert SRT timestamp to seconds."""
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    
    def _vtt_time_to_seconds(self, time_str: str) -> float:
        """Convert VTT timestamp to seconds."""
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    
    def _calculate_emotion_change(self, emotions1: Dict, emotions2: Dict) -> float:
        """Calculate magnitude of emotional change between two states."""
        if not emotions1 or not emotions2:
            return 0.0
        
        change = 0.0
        all_emotions = set(emotions1.keys()) | set(emotions2.keys())
        
        for emotion in all_emotions:
            val1 = emotions1.get(emotion, 0.0)
            val2 = emotions2.get(emotion, 0.0)
            change += abs(val1 - val2)
        
        return change / len(all_emotions) if all_emotions else 0.0
    
    def _determine_scene_type(self, text: str) -> str:
        """Determine the type of scene based on text content."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['action', 'fight', 'chase', 'run', 'explosion']):
            return 'action'
        elif any(word in text_lower for word in ['says', 'tells', 'asks', 'replies', 'whispers']):
            return 'dialogue'
        elif any(word in text_lower for word in ['fade', 'cut', 'scene', 'int.', 'ext.']):
            return 'transition'
        else:
            return 'dialogue'  # Default
