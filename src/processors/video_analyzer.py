import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import moviepy.editor as mp
from transformers import CLIPProcessor, CLIPModel
import logging

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    """
    Analyzes video content for visual features, shot types, and content understanding.
    Uses CLIP model for semantic understanding of video frames.
    """
    
    def __init__(self, config: dict):
        """
        Initialize VideoAnalyzer with configuration settings.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.video_config = config.get('video', {})
        
        # Load CLIP model for visual understanding
        model_name = config['models']['visual_features']
        logger.info(f"Loading CLIP model: {model_name}")
        
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        
        # Define visual concepts for film analysis
        self.visual_concepts = [
            "close-up shot", "medium shot", "wide shot", "extreme close-up",
            "over-the-shoulder shot", "establishing shot", "tracking shot",
            "dark scene", "bright scene", "night scene", "day scene",
            "outdoor scene", "indoor scene", "urban scene", "nature scene",
            "action scene", "dialogue scene", "emotional scene", "dramatic scene",
            "happy scene", "sad scene", "tense scene", "peaceful scene"
        ]
        
    def extract_frames(self, video_path: str, interval: Optional[float] = None) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to video file
            interval: Time interval between frames (seconds)
            
        Returns:
            List of tuples containing (timestamp, frame_array)
        """
        if interval is None:
            interval = self.video_config.get('frame_extraction_interval', 1.0)
            
        logger.info(f"Extracting frames from {video_path} at {interval}s intervals")
        
        try:
            video = mp.VideoFileClip(video_path)
            frames = []
            
            for t in np.arange(0, video.duration, interval):
                try:
                    frame = video.get_frame(t)
                    frames.append((t, frame))
                except Exception as e:
                    logger.warning(f"Failed to extract frame at {t}s: {e}")
                    continue
                    
            video.close()
            logger.info(f"Extracted {len(frames)} frames")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise
    
    def analyze_visual_content(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Analyze frame for visual features using CLIP.
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            Dictionary mapping visual concepts to confidence scores
        """
        try:
            # Convert frame to PIL Image
            if frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            image = Image.fromarray(frame)
            
            # Process image and text concepts
            inputs = self.clip_processor(
                text=self.visual_concepts,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
            # Return concept probabilities
            concept_scores = {
                concept: prob.item() 
                for concept, prob in zip(self.visual_concepts, probs[0])
            }
            
            return concept_scores
            
        except Exception as e:
            logger.error(f"Error analyzing visual content: {e}")
            return {concept: 0.0 for concept in self.visual_concepts}
    
    def analyze_video_timeline(self, video_path: str) -> Dict[str, List]:
        """
        Analyze entire video timeline for visual features.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing timeline analysis data
        """
        logger.info(f"Analyzing video timeline: {video_path}")
        
        frames_with_timestamps = self.extract_frames(video_path)
        
        timeline_analysis = {
            'timestamps': [],
            'visual_features': [],
            'shot_types': [],
            'scene_moods': [],
            'lighting_conditions': []
        }
        
        for timestamp, frame in frames_with_timestamps:
            # Analyze visual content
            visual_analysis = self.analyze_visual_content(frame)
            
            # Extract specific feature categories
            shot_types = {k: v for k, v in visual_analysis.items() 
                         if 'shot' in k.lower()}
            scene_moods = {k: v for k, v in visual_analysis.items() 
                          if any(mood in k.lower() for mood in ['happy', 'sad', 'tense', 'peaceful', 'dramatic', 'emotional'])}
            lighting = {k: v for k, v in visual_analysis.items() 
                       if any(light in k.lower() for light in ['dark', 'bright', 'night', 'day'])}
            
            # Store results
            timeline_analysis['timestamps'].append(timestamp)
            timeline_analysis['visual_features'].append(visual_analysis)
            timeline_analysis['shot_types'].append(max(shot_types, key=shot_types.get) if shot_types else 'medium shot')
            timeline_analysis['scene_moods'].append(max(scene_moods, key=scene_moods.get) if scene_moods else 'neutral')
            timeline_analysis['lighting_conditions'].append(max(lighting, key=lighting.get) if lighting else 'normal')
        
        logger.info(f"Completed timeline analysis with {len(timeline_analysis['timestamps'])} data points")
        return timeline_analysis
    
    def calculate_visual_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate visual similarity between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Get visual features for both frames
            features1 = self.analyze_visual_content(frame1)
            features2 = self.analyze_visual_content(frame2)
            
            # Calculate cosine similarity
            v1 = np.array(list(features1.values()))
            v2 = np.array(list(features2.values()))
            
            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating visual similarity: {e}")
            return 0.5  # Default similarity
    
    def get_video_metadata(self, video_path: str) -> Dict:
        """
        Extract basic video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video metadata
        """
        try:
            video = mp.VideoFileClip(video_path)
            metadata = {
                'duration': video.duration,
                'fps': video.fps,
                'size': video.size,
                'width': video.w,
                'height': video.h,
                'aspect_ratio': video.w / video.h if video.h > 0 else 1.0
            }
            video.close()
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting video metadata: {e}")
            return {}
