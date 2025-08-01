"""
AI Film Editor - Smart Cut and Transition Suggestions

A comprehensive video editing assistant that uses AI to analyze video content,
scripts, and audio to provide intelligent cut points and transition suggestions.
"""

__version__ = "1.0.0"
__author__ = "AI Film Editor Team"
__email__ = "contact@aifilmeditor.com"

# Package imports for easy access
from .processors import VideoAnalyzer, ScriptParser, AudioAnalyzer, SceneDetector
from .ai_models import EmotionDetector, SentimentAnalyzer, VisualAnalyzer
from .suggestions import CutSuggester, TransitionRecommender
from .ui import TimelineViewer, SuggestionPanel
from .utils import FileHandler, TimelineSync

__all__ = [
    'VideoAnalyzer',
    'ScriptParser', 
    'AudioAnalyzer',
    'SceneDetector',
    'EmotionDetector',
    'SentimentAnalyzer',
    'VisualAnalyzer',
    'CutSuggester',
    'TransitionRecommender',
    'TimelineViewer',
    'SuggestionPanel',
    'FileHandler',
    'TimelineSync'
]
