import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class CutSuggestion:
    timestamp: float
    confidence: float
    reason: str
    suggestion_type: str  # 'scene_change', 'emotion_beat', 'speaker_change'
    
class CutSuggester:
    """Generates intelligent cut suggestions based on multimodal analysis"""
    
    def __init__(self, config: dict):
        self.config = config
        
    def generate_suggestions(self, 
                           video_analysis: Dict,
                           script_analysis: List[Dict],
                           audio_analysis: Dict) -> List[CutSuggestion]:
        """Combine all analyses to generate cut suggestions"""
        
        suggestions = []
        
        # Add scene change suggestions
        scene_changes = video_analysis.get('scene_changes', [])
        for timestamp in scene_changes:
            suggestions.append(CutSuggestion(
                timestamp=timestamp,
                confidence=0.8,
                reason="Visual scene change detected",
                suggestion_type="scene_change"
            ))
            
        # Add emotional beat suggestions
        emotional_beats = script_analysis.get('emotional_beats', [])
        for timestamp in emotional_beats:
            suggestions.append(CutSuggestion(
                timestamp=timestamp,
                confidence=0.7,
                reason="Emotional transition in dialogue",
                suggestion_type="emotion_beat"
            ))
            
        # Add speaker change suggestions
        speaker_changes = audio_analysis.get('speaker_changes', [])
        for timestamp in speaker_changes:
            suggestions.append(CutSuggestion(
                timestamp=timestamp,
                confidence=0.6,
                reason="Potential speaker change detected",
                suggestion_type="speaker_change"
            ))
            
        # Sort by timestamp and filter close suggestions
        suggestions = self._filter_close_suggestions(suggestions)
        suggestions.sort(key=lambda x: x.timestamp)
        
        return suggestions
    
    def _filter_close_suggestions(self, suggestions: List[CutSuggestion]) -> List[CutSuggestion]:
        """Remove suggestions that are too close to each other"""
        min_interval = self.config['suggestions']['minimum_cut_interval']
        filtered = []
        
        for suggestion in sorted(suggestions, key=lambda x: x.timestamp):
            if not filtered or suggestion.timestamp - filtered[-1].timestamp >= min_interval:
                filtered.append(suggestion)
            elif suggestion.confidence > filtered[-1].confidence:
                # Replace with higher confidence suggestion
                filtered[-1] = suggestion
                
        return filtered
