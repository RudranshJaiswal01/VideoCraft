import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class TimelineViewer:
    """
    Interactive timeline visualization component for displaying AI suggestions
    and analysis results in a user-friendly interface.
    """
    
    def __init__(self, config: dict):
        """
        Initialize TimelineViewer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.timeline_config = config.get('timeline', {})
        self.ui_config = config.get('ui', {})
        
        # Color schemes for different elements
        self.colors = {
            'scene_change': '#FF6B6B',
            'emotion_beat': '#4ECDC4', 
            'speaker_change': '#45B7D1',
            'dialogue_pause': '#96CEB4',
            'audio_silence': '#FECA57',
            'manual': '#6C5CE7',
            'cut': '#FF6B6B',
            'dissolve': '#A8E6CF',
            'fade': '#FFB347',
            'wipe': '#AED6F1'
        }
        
    def create_timeline_visualization(self, 
                                    video_duration: float,
                                    cut_suggestions: List,
                                    transition_suggestions: List,
                                    emotion_timeline: List[Dict],
                                    audio_timeline: Optional[List[Dict]] = None) -> go.Figure:
        """
        Create comprehensive timeline visualization with all analysis data.
        
        Args:
            video_duration: Total video duration in seconds
            cut_suggestions: List of cut suggestions
            transition_suggestions: List of transition suggestions  
            emotion_timeline: Timeline of emotion analysis
            audio_timeline: Optional audio analysis timeline
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating timeline visualization")
        
        fig = go.Figure()
        
        # Add emotion background
        self._add_emotion_background(fig, emotion_timeline, video_duration)
        
        # Add audio energy visualization
        if audio_timeline:
            self._add_audio_visualization(fig, audio_timeline)
        
        # Add cut suggestions
        self._add_cut_suggestions(fig, cut_suggestions)
        
        # Add transition suggestions
        self._add_transition_suggestions(fig, transition_suggestions)
        
        # Customize layout
        self._customize_layout(fig, video_duration)
        
        return fig
    
    def _add_emotion_background(self, fig: go.Figure, emotion_timeline: List[Dict], duration: float):
        """Add emotion analysis as background colored regions."""
        if not emotion_timeline:
            return
        
        # Define emotion colors
        emotion_colors = {
            'happy': 'rgba(255, 235, 59, 0.3)',      # Yellow
            'sad': 'rgba(63, 81, 181, 0.3)',         # Blue  
            'angry': 'rgba(244, 67, 54, 0.3)',       # Red
            'fearful': 'rgba(156, 39, 176, 0.3)',    # Purple
            'surprised': 'rgba(255, 152, 0, 0.3)',   # Orange
            'disgusted': 'rgba(76, 175, 80, 0.3)',   # Green
            'neutral': 'rgba(158, 158, 158, 0.2)'    # Gray
        }
        
        for emotion_data in emotion_timeline:
            start_time = emotion_data.get('start_time', 0)
            end_time = emotion_data.get('end_time', start_time + 5)
            emotion = emotion_data.get('emotion', 'neutral')
            
            color = emotion_colors.get(emotion, 'rgba(158, 158, 158, 0.2)')
            
            # Add background rectangle
            fig.add_shape(
                type="rect",
                x0=start_time,
                x1=end_time,
                y0=-0.5,
                y1=4.5,
                fillcolor=color,
                line=dict(width=0),
                layer="below"
            )
            
            # Add emotion label
            if end_time - start_time > 5:  # Only label longer segments
                fig.add_annotation(
                    x=(start_time + end_time) / 2,
                    y=4,
                    text=emotion.capitalize(),
                    showarrow=False,
                    font=dict(size=10, color="rgba(0,0,0,0.6)"),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1
                )
    
    def _add_audio_visualization(self, fig: go.Figure, audio_timeline: List[Dict]):
        """Add audio energy visualization."""
        if not audio_timeline:
            return
        
        # Extract time and energy data
        times = [item.get('start_time', 0) for item in audio_timeline]
        energies = [item.get('rms_energy', 0) for item in audio_timeline]
        
        # Normalize energies
        if energies:
            max_energy = max(energies)
            if max_energy > 0:
                energies = [e / max_energy for e in energies]
        
        # Add audio waveform representation
        fig.add_trace(go.Scatter(
            x=times,
            y=[3.5 + e * 0.4 for e in energies],  # Scale and offset
            mode='lines',
            name='Audio Energy',
            line=dict(color='rgba(0,100,0,0.6)', width=1),
            fill='tonexty',
            fillcolor='rgba(0,100,0,0.1)'
        ))
    
    def _add_cut_suggestions(self, fig: go.Figure, cut_suggestions: List):
        """Add cut suggestions as vertical lines."""
        suggestion_types = {}
        
        # Group suggestions by type
        for cut in cut_suggestions:
            suggestion_type = getattr(cut, 'suggestion_type', 'unknown')
            if suggestion_type not in suggestion_types:
                suggestion_types[suggestion_type] = []
            suggestion_types[suggestion_type].append(cut)
        
        # Add each type with different styling
        for suggestion_type, cuts in suggestion_types.items():
            color = self.colors.get(suggestion_type, '#888888')
            
            for cut in cuts:
                timestamp = getattr(cut, 'timestamp', 0)
                confidence = getattr(cut, 'confidence', 0.5)
                reason = getattr(cut, 'reason', 'No reason')
                
                # Line opacity based on confidence
                opacity = 0.5 + (confidence * 0.5)
                
                fig.add_vline(
                    x=timestamp,
                    line=dict(
                        color=color,
                        width=2,
                        dash="solid" if confidence > 0.7 else "dash"
                    ),
                    opacity=opacity,
                    annotation_text=f"{suggestion_type.replace('_', ' ').title()}<br>Conf: {confidence:.2f}",
                    annotation_position="top",
                    annotation=dict(
                        font=dict(size=9),
                        bgcolor=color,
                        bordercolor="white",
                        borderwidth=1
                    )
                )
    
    def _add_transition_suggestions(self, fig: go.Figure, transition_suggestions: List):
        """Add transition suggestions as spans."""
        for trans in transition_suggestions:
            start_time = getattr(trans, 'start_time', 0)
            end_time = getattr(trans, 'end_time', start_time + 1)
            transition_type = getattr(trans, 'transition_type', None)
            confidence = getattr(trans, 'confidence', 0.5)
            
            if transition_type:
                transition_name = transition_type.value if hasattr(transition_type, 'value') else str(transition_type)
                color = self.colors.get(transition_name, '#CCCCCC')
                
                # Add transition span
                fig.add_shape(
                    type="rect",
                    x0=start_time,
                    x1=end_time,
                    y0=2.3,
                    y1=2.7,
                    fillcolor=color,
                    line=dict(color=color, width=1),
                    opacity=0.5 + (confidence * 0.5)
                )
                
                # Add transition label
                fig.add_annotation(
                    x=(start_time + end_time) / 2,
                    y=2.5,
                    text=transition_name.replace('_', ' ').title(),
                    showarrow=False,
                    font=dict(size=8, color="white"),
                    bgcolor=color,
                    bordercolor="white",
                    borderwidth=1
                )
    
    def _customize_layout(self, fig: go.Figure, video_duration: float):
        """Customize the layout of the timeline visualization."""
        timeline_height = self.ui_config.get('timeline_height', 400)
        
        fig.update_layout(
            title={
                'text': "AI Film Editing Timeline",
                'x': 0.5,
                'font': {'size': 18, 'color': '#2C3E50'}
            },
            xaxis={
                'title': "Time (seconds)",
                'range': [0, video_duration],
                'showgrid': True,
                'gridcolor': 'rgba(128,128,128,0.2)',
                'tickformat': '.1f'
            },
            yaxis={
                'title': "",
                'range': [-0.5, 4.5],
                'showticklabels': False,
                'showgrid': False
            },
            height=timeline_height,
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            margin=dict(l=50, r=200, t=80, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add timeline sections labels
        fig.add_annotation(x=-video_duration*0.08, y=4, text="Emotions", showarrow=False, font=dict(size=12, color='#7F8C8D'))
        fig.add_annotation(x=-video_duration*0.08, y=3.5, text="Audio", showarrow=False, font=dict(size=12, color='#7F8C8D'))
        fig.add_annotation(x=-video_duration*0.08, y=2.5, text="Transitions", showarrow=False, font=dict(size=12, color='#7F8C8D'))
        fig.add_annotation(x=-video_duration*0.08, y=1, text="Cuts", showarrow=False, font=dict(size=12, color='#7F8C8D'))
    
    def create_suggestion_summary_chart(self, cut_suggestions: List) -> go.Figure:
        """Create a summary chart of suggestion types and confidences."""
        if not cut_suggestions:
            return go.Figure()
        
        # Analyze suggestion types
        type_counts = {}
        type_confidences = {}
        
        for cut in cut_suggestions:
            suggestion_type = getattr(cut, 'suggestion_type', 'unknown')
            confidence = getattr(cut, 'confidence', 0.5)
            
            type_counts[suggestion_type] = type_counts.get(suggestion_type, 0) + 1
            if suggestion_type not in type_confidences:
                type_confidences[suggestion_type] = []
            type_confidences[suggestion_type].append(confidence)
        
        # Calculate average confidences
        avg_confidences = {
            t: np.mean(confidences) 
            for t, confidences in type_confidences.items()
        }
        
        # Create summary chart
        fig = go.Figure()
        
        # Bar chart of suggestion counts
        fig.add_trace(go.Bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            name='Count',
            marker_color=[self.colors.get(t, '#888888') for t in type_counts.keys()],
            text=[f"{v} suggestions" for v in type_counts.values()],
            textposition='auto'
        ))
        
        # Add average confidence as secondary y-axis
        fig.add_trace(go.Scatter(
            x=list(avg_confidences.keys()),
            y=list(avg_confidences.values()),
            mode='markers+lines',
            name='Avg Confidence',
            yaxis='y2',
            line=dict(color='red', width=2),
            marker=dict(size=8, color='red')
        ))
        
        fig.update_layout(
            title='Suggestion Summary',
            xaxis=dict(title='Suggestion Type'),
            yaxis=dict(title='Number of Suggestions'),
            yaxis2=dict(
                title='Average Confidence',
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            height=300
        )
        
        return fig
    
    def format_timestamp(self, seconds: float) -> str:
        """Format timestamp for display."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
