import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import moviepy.editor as mp

class VideoAnalyzer:
    """Analyzes video content for visual features and scene changes"""
    
    def __init__(self, config: dict):
        self.config = config
        # Load CLIP model for visual understanding
        self.clip_model = CLIPModel.from_pretrained(config['models']['visual_features'])
        self.clip_processor = CLIPProcessor.from_pretrained(config['models']['visual_features'])
        
    def extract_frames(self, video_path: str, interval: float = 1.0) -> List[np.ndarray]:
        """Extract frames at specified intervals"""
        video = mp.VideoFileClip(video_path)
        frames = []
        
        for t in np.arange(0, video.duration, interval):
            frame = video.get_frame(t)
            frames.append(frame)
            
        video.close()
        return frames
    
    def analyze_visual_content(self, frame: np.ndarray) -> Dict:
        """Analyze frame for visual features using CLIP"""
        # Convert frame to PIL Image
        image = Image.fromarray(frame)
        
        # Predefined visual concepts for film editing
        concepts = [
            "close-up shot", "wide shot", "medium shot", 
            "dark scene", "bright scene", "outdoor scene", "indoor scene",
            "action scene", "dialogue scene", "emotional scene"
        ]
        
        # Process image and text
        inputs = self.clip_processor(
            text=concepts, 
            images=image, 
            return_tensors="pt", 
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
        # Return concept probabilities
        return {concept: prob.item() for concept, prob in zip(concepts, probs[0])}
    
    def detect_scene_changes(self, video_path: str) -> List[float]:
        """Detect scene boundaries using visual similarity"""
        frames = self.extract_frames(video_path, interval=0.5)
        scene_changes = [0.0]  # Always include start
        
        prev_features = None
        for i, frame in enumerate(frames):
            features = self.analyze_visual_content(frame)
            
            if prev_features is not None:
                # Calculate similarity between consecutive frames
                similarity = self._calculate_similarity(prev_features, features)
                
                if similarity < self.config['suggestions']['scene_change_threshold']:
                    timestamp = i * 0.5
                    scene_changes.append(timestamp)
                    
            prev_features = features
            
        return scene_changes
    
    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate cosine similarity between feature vectors"""
        v1 = np.array(list(features1.values()))
        v2 = np.array(list(features2.values()))
        
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
