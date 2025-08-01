"""
Video, Audio, and Script Processing Modules

This package contains all the processors for analyzing different types of content:
- VideoAnalyzer: Processes video frames for visual features
- AudioAnalyzer: Extracts and analyzes audio features  
- ScriptParser: Parses and analyzes script content
- SceneDetector: Detects scene changes in video content
"""

from .video_analyzer import VideoAnalyzer
from .audio_analyzer import AudioAnalyzer
from .script_parser import ScriptParser
from .scene_detector import SceneDetector

__all__ = ['VideoAnalyzer', 'AudioAnalyzer', 'ScriptParser', 'SceneDetector']
