from enum import Enum
from typing import List, Dict
from dataclasses import dataclass

class TransitionType(Enum):
    CUT = "cut"
    FADE = "fade"
    DISSOLVE = "dissolve"
    WIPE = "wipe"
    ZOOM = "zoom"

@dataclass
class TransitionSuggestion:
    start_time: float
    end_time: float
    transition_type: TransitionType
    confidence: float
    reason: str

class TransitionRecommender:
    """Recommends appropriate transitions based on content analysis"""
    
    def __init__(self, config: dict):
        self.config = config
        
    def suggest_transitions(self, 
                          cut_suggestions: List,
                          emotion_analysis: List[Dict],
                          visual_analysis: Dict) -> List[TransitionSuggestion]:
        """Suggest appropriate transitions for each cut point"""
        
        transitions = []
        
        for i, cut in enumerate(cut_suggestions):
            # Analyze context around cut point
            context = self._analyze_cut_context(
                cut.timestamp, 
                emotion_analysis, 
                visual_analysis
            )
            
            # Determine best transition type
            transition_type = self._determine_transition_type(context, cut.reason)
            
            # Calculate next cut point for transition duration
            next_timestamp = cut_suggestions[i + 1].timestamp if i + 1 < len(cut_suggestions) else cut.timestamp + 2.0
            
            transitions.append(TransitionSuggestion(
                start_time=cut.timestamp,
                end_time=min(cut.timestamp + 1.0, next_timestamp),
                transition_type=transition_type,
                confidence=self._calculate_transition_confidence(context),
                reason=self._get_transition_reason(transition_type, context)
            ))
            
        return transitions
    
    def _analyze_cut_context(self, timestamp: float, emotion_analysis: List[Dict], visual_analysis: Dict) -> Dict:
        """Analyze the context around a cut point"""
        context = {
            'emotion_before': None,
            'emotion_after': None,
            'visual_change_intensity': 0.5,
            'is_dialogue_scene': False
        }
        
        # Find emotions before and after the cut
        for emotion_data in emotion_analysis:
            if emotion_data['start_time'] <= timestamp <= emotion_data['end_time']:
                context['emotion_before'] = emotion_data['emotion']
            elif emotion_data['start_time'] > timestamp:
                context['emotion_after'] = emotion_data['emotion']
                break
                
        return context
    
    def _determine_transition_type(self, context: Dict, cut_reason: str) -> TransitionType:
        """Determine the most appropriate transition type"""
        
        # Default to cut for most scenarios
        if cut_reason == "scene_change":
            if context['visual_change_intensity'] > 0.7:
                return TransitionType.CUT
            else:
                return TransitionType.DISSOLVE
                
        elif cut_reason == "emotion_beat":
            emotion_before = context.get('emotion_before', '')
            emotion_after = context.get('emotion_after', '')
            
            # Fade for dramatic emotional changes
            if any(emotion in [emotion_before, emotion_after] for emotion in ['sadness', 'grief']):
                return TransitionType.FADE
            else:
                return TransitionType.CUT
                
        elif cut_reason == "speaker_change":
            return TransitionType.CUT
            
        return TransitionType.CUT
    
    def _calculate_transition_confidence(self, context: Dict) -> float:
        """Calculate confidence score for transition suggestion"""
        base_confidence = 0.7
        
        # Adjust based on context clarity
        if context['emotion_before'] and context['emotion_after']:
            base_confidence += 0.1
            
        if context['visual_change_intensity'] > 0.6:
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)
    
    def _get_transition_reason(self, transition_type: TransitionType, context: Dict) -> str:
        """Generate human-readable reason for transition choice"""
        reasons = {
            TransitionType.CUT: "Sharp visual/emotional change",
            TransitionType.FADE: "Dramatic emotional transition",
            TransitionType.DISSOLVE: "Smooth scene transition",
            TransitionType.WIPE: "Dynamic action sequence",
            TransitionType.ZOOM: "Focus change emphasis"
        }
        
        return reasons.get(transition_type, "Standard transition")
