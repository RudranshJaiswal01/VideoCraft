import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict
import pandas as pd

class TimelineViewer:
    """Interactive timeline visualization for AI suggestions"""
    
    def __init__(self, config: dict):
        self.config = config
        
    def create_timeline_visualization(self, 
                                    video_duration: float,
                                    cut_suggestions: List,
                                    transition_suggestions: List,
                                    emotion_timeline: List[Dict]) -> go.Figure:
        """Create interactive timeline with all suggestions"""
        
        fig = go.Figure()
        
        # Add emotion timeline as background
        self._add_emotion_background(fig, emotion_timeline, video_duration)
        
        # Add cut suggestions
        self._add_cut_suggestions(fig, cut_suggestions)
        
        # Add transition suggestions
        self._add_transition_suggestions(fig, transition_suggestions)
        
        # Customize layout
        fig.update_layout(
            title="AI Film Editing Timeline",
            xaxis_title="Time (seconds)",
            yaxis_title="Suggestion Type",
            height=600,
            showlegend=True,
            xaxis=dict(range=[0, video_duration])
        )
        
        return fig
    
    def _add_emotion_background(self, fig: go.Figure, emotion_timeline: List[Dict], duration: float):
        """Add emotion analysis as background color"""
        if not emotion_timeline:
            return
            
        emotions_df = pd.DataFrame(emotion_timeline)
        
        # Create emotion color mapping
        emotion_colors = {
            'joy': 'rgba(255, 255, 0, 0.3)',
            'sadness': 'rgba(0, 0, 255, 0.3)',
            'anger': 'rgba(255, 0, 0, 0.3)',
            'fear': 'rgba(128, 0, 128, 0.3)',
            'surprise': 'rgba(255, 165, 0, 0.3)',
            'neutral': 'rgba(128, 128, 128, 0.3)'
        }
        
        for _, emotion_data in emotions_df.iterrows():
            color = emotion_colors.get(emotion_data['emotion'], 'rgba(128, 128, 128, 0.3)')
            
            fig.add_shape(
                type="rect",
                x0=emotion_data['start_time'],
                x1=emotion_data['end_time'],
                y0=-0.5,
                y1=3.5,
                fillcolor=color,
                line=dict(width=0),
                layer="below"
            )
    
    def _add_cut_suggestions(self, fig: go.Figure, cut_suggestions: List):
        """Add cut suggestions as vertical lines"""
        for cut in cut_suggestions:
            color = self._get_suggestion_color(cut.suggestion_type)
            
            fig.add_vline(
                x=cut.timestamp,
                line=dict(color=color, width=2, dash="dash"),
                annotation_text=f"Cut: {cut.reason[:20]}...",
                annotation_position="top"
            )
    
    def _add_transition_suggestions(self, fig: go.Figure, transition_suggestions: List):
        """Add transition suggestions as spans"""
        for trans in transition_suggestions:
            fig.add_shape(
                type="rect",
                x0=trans.start_time,
                x1=trans.end_time,
                y0=2.5,
                y1=3,
                fillcolor="rgba(0, 255, 0, 0.5)",
                line=dict(color="green", width=1)
            )
            
            # Add annotation
            fig.add_annotation(
                x=(trans.start_time + trans.end_time) / 2,
                y=2.75,
                text=trans.transition_type.value,
                showarrow=False,
                font=dict(size=10)
            )
    
    def _get_suggestion_color(self, suggestion_type: str) -> str:
        """Get color for different suggestion types"""
        colors = {
            'scene_change': 'red',
            'emotion_beat': 'blue',
            'speaker_change': 'orange'
        }
        return colors.get(suggestion_type, 'gray')
    
    def render_suggestion_panel(self, suggestions: List):
        """Render interactive suggestion panel"""
        st.subheader("AI Editing Suggestions")
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5)
            
        with col2:
            suggestion_types = st.multiselect(
                "Suggestion Types",
                ['scene_change', 'emotion_beat', 'speaker_change'],
                default=['scene_change', 'emotion_beat', 'speaker_change']
            )
            
        with col3:
            sort_by = st.selectbox("Sort By", ['timestamp', 'confidence'])
        
        # Filter suggestions
        filtered_suggestions = [
            s for s in suggestions 
            if s.confidence >= min_confidence and s.suggestion_type in suggestion_types
        ]
        
        # Sort suggestions
        if sort_by == 'confidence':
            filtered_suggestions.sort(key=lambda x: x.confidence, reverse=True)
        else:
            filtered_suggestions.sort(key=lambda x: x.timestamp)
        
        # Display suggestions
        for i, suggestion in enumerate(filtered_suggestions):
            with st.expander(f"Cut {i+1}: {suggestion.timestamp:.1f}s - {suggestion.reason}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Confidence", f"{suggestion.confidence:.2f}")
                    
                with col2:
                    st.metric("Type", suggestion.suggestion_type)
                    
                with col3:
                    if st.button(f"Apply Cut {i+1}", key=f"apply_{i}"):
                        st.success("Cut applied to timeline!")
