import streamlit as st
from typing import List, Dict, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class SuggestionPanel:
    """
    Interactive panel for displaying, filtering, and managing AI suggestions.
    Provides user controls for reviewing and applying editing suggestions.
    """
    
    def __init__(self, config: dict):
        """
        Initialize SuggestionPanel with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ui_config = config.get('ui', {})
        
    def render_suggestion_controls(self) -> Dict:
        """
        Render control panel for filtering and sorting suggestions.
        
        Returns:
            Dictionary with user-selected filter options
        """
        st.subheader("🎛️ Suggestion Controls")
        
        with st.expander("Filter & Sort Options", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_confidence = st.slider(
                    "Minimum Confidence", 
                    0.0, 1.0, 0.5, 0.1,
                    help="Filter suggestions below this confidence level"
                )
                
            with col2:
                suggestion_types = st.multiselect(
                    "Suggestion Types",
                    ['scene_change', 'emotion_beat', 'speaker_change', 'dialogue_pause', 'audio_silence'],
                    default=['scene_change', 'emotion_beat', 'speaker_change'],
                    help="Select which types of suggestions to show"
                )
                
            with col3:
                sort_by = st.selectbox(
                    "Sort By", 
                    ['timestamp', 'confidence', 'type'],
                    help="Sort suggestions by selected criteria"
                )
                
                sort_order = st.radio(
                    "Order",
                    ['Ascending', 'Descending'],
                    horizontal=True
                )
        
        return {
            'min_confidence': min_confidence,
            'suggestion_types': suggestion_types,
            'sort_by': sort_by,
            'sort_order': sort_order.lower()
        }
    
    def render_suggestion_list(self, 
                              suggestions: List,
                              filters: Dict,
                              transition_suggestions: Optional[List] = None) -> List:
        """
        Render filtered and sorted list of suggestions.
        
        Args:
            suggestions: List of cut suggestions
            filters: Filter options from controls
            transition_suggestions: Optional list of transition suggestions
            
        Returns:
            List of user-selected suggestions
        """
        if not suggestions:
            st.warning("No suggestions to display")
            return []
        
        # Apply filters
        filtered_suggestions = self._apply_filters(suggestions, filters)
        
        if not filtered_suggestions:
            st.warning("No suggestions match the current filters")
            return []
        
        # Sort suggestions
        sorted_suggestions = self._sort_suggestions(filtered_suggestions, filters)
        
        st.subheader(f"📋 AI Suggestions ({len(sorted_suggestions)} of {len(suggestions)})")
        
        selected_suggestions = []
        
        # Render each suggestion
        for i, suggestion in enumerate(sorted_suggestions):
            with st.container():
                # Create suggestion card
                suggestion_data = self._render_suggestion_card(
                    suggestion, i, transition_suggestions
                )
                
                if suggestion_data['selected']:
                    selected_suggestions.append(suggestion)
        
        return selected_suggestions
    
    def _render_suggestion_card(self, 
                               suggestion,
                               index: int,
                               transition_suggestions: Optional[List] = None) -> Dict:
        """Render individual suggestion card."""
        
        # Extract suggestion attributes
        timestamp = getattr(suggestion, 'timestamp', 0)
        confidence = getattr(suggestion, 'confidence', 0.5)
        reason = getattr(suggestion, 'reason', 'No reason provided')
        suggestion_type = getattr(suggestion, 'suggestion_type', 'unknown')
        metadata = getattr(suggestion, 'metadata', {})
        
        # Format timestamp
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        # Create expandable card
        with st.expander(
            f"🎬 Cut #{index + 1}: {time_str} - {suggestion_type.replace('_', ' ').title()} (Conf: {confidence:.2f})",
            expanded=False
        ):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Reason:** {reason}")
                
                # Show metadata if available
                if metadata:
                    with st.expander("Additional Details", expanded=False):
                        for key, value in metadata.items():
                            if key != 'source':
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
                st.metric("Time", time_str)
            
            with col3:
                selected = st.checkbox(
                    "Apply Cut",
                    key=f"cut_{index}_{timestamp}",
                    help="Select this cut to apply to timeline"
                )
                
                if st.button(
                    "Preview",
                    key=f"preview_{index}_{timestamp}",
                    help="Preview this cut point"
                ):
                    self._show_preview_info(suggestion)
                
                # Find associated transition
                transition = self._find_transition_for_cut(
                    timestamp, transition_suggestions
                )
                
                if transition:
                    transition_type = getattr(transition, 'transition_type', None)
                    if transition_type:
                        transition_name = transition_type.value if hasattr(transition_type, 'value') else str(transition_type)
                        st.info(f"🎭 **Transition:** {transition_name.replace('_', ' ').title()}")
        
        return {'selected': selected, 'suggestion': suggestion}
    
    def render_batch_actions(self, selected_suggestions: List) -> Dict:
        """Render batch action controls for selected suggestions."""
        if not selected_suggestions:
            return {}
        
        st.subheader(f"🎯 Batch Actions ({len(selected_suggestions)} selected)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Apply All Cuts", type="primary"):
                return {'action': 'apply_all', 'suggestions': selected_suggestions}
        
        with col2:
            if st.button("Export Cut List"):
                return {'action': 'export', 'suggestions': selected_suggestions}
        
        with col3:
            if st.button("Clear Selection"):
                return {'action': 'clear', 'suggestions': []}
        
        return {}
    
    def render_suggestion_analytics(self, suggestions: List):
        """Render analytics panel for suggestions."""
        if not suggestions:
            return
        
        st.subheader("📊 Suggestion Analytics")
        
        # Create analytics data
        analytics_data = self._create_analytics_data(suggestions)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Suggestion type distribution
            type_counts = analytics_data['type_counts']
            if type_counts:
                st.write("**Suggestion Types:**")
                for suggestion_type, count in type_counts.items():
                    percentage = (count / len(suggestions)) * 100
                    st.write(f"• {suggestion_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        with col2:
            # Confidence statistics
            confidences = analytics_data['confidences']
            if confidences:
                st.write("**Confidence Statistics:**")
                st.write(f"• Average: {analytics_data['avg_confidence']:.2%}")
                st.write(f"• High confidence (>70%): {analytics_data['high_confidence_count']} ({analytics_data['high_confidence_percent']:.1f}%)")
                st.write(f"• Total suggestions: {len(suggestions)}")
    
    def _apply_filters(self, suggestions: List, filters: Dict) -> List:
        """Apply user-selected filters to suggestions."""
        filtered = []
        
        for suggestion in suggestions:
            # Check confidence threshold
            confidence = getattr(suggestion, 'confidence', 0.5)
            if confidence < filters['min_confidence']:
                continue
            
            # Check suggestion type
            suggestion_type = getattr(suggestion, 'suggestion_type', 'unknown')
            if suggestion_type not in filters['suggestion_types']:
                continue
            
            filtered.append(suggestion)
        
        return filtered
    
    def _sort_suggestions(self, suggestions: List, filters: Dict) -> List:
        """Sort suggestions based on user selection."""
        sort_by = filters['sort_by']
        reverse = filters['sort_order'] == 'descending'
        
        if sort_by == 'timestamp':
            return sorted(suggestions, key=lambda x: getattr(x, 'timestamp', 0), reverse=reverse)
        elif sort_by == 'confidence':
            return sorted(suggestions, key=lambda x: getattr(x, 'confidence', 0.5), reverse=reverse)
        elif sort_by == 'type':
            return sorted(suggestions, key=lambda x: getattr(x, 'suggestion_type', 'unknown'), reverse=reverse)
        
        return suggestions
    
    def _show_preview_info(self, suggestion):
        """Show preview information for a suggestion."""
        st.info(f"Preview functionality for cut at {getattr(suggestion, 'timestamp', 0):.1f}s would be implemented here")
    
    def _find_transition_for_cut(self, timestamp: float, transition_suggestions: Optional[List]):
        """Find transition suggestion associated with cut timestamp."""
        if not transition_suggestions:
            return None
        
        for transition in transition_suggestions:
            start_time = getattr(transition, 'start_time', 0)
            if abs(start_time - timestamp) < 0.5:  # Within 0.5 seconds
                return transition
        
        return None
    
    def _create_analytics_data(self, suggestions: List) -> Dict:
        """Create analytics data from suggestions."""
        type_counts = {}
        confidences = []
        
        for suggestion in suggestions:
            # Count suggestion types
            suggestion_type = getattr(suggestion, 'suggestion_type', 'unknown')
            type_counts[suggestion_type] = type_counts.get(suggestion_type, 0) + 1
            
            # Collect confidences
            confidence = getattr(suggestion, 'confidence', 0.5)
            confidences.append(confidence)
        
        # Calculate statistics
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        high_confidence_count = sum(1 for c in confidences if c > 0.7)
        high_confidence_percent = (high_confidence_count / len(confidences) * 100) if confidences else 0
        
        return {
            'type_counts': type_counts,
            'confidences': confidences,
            'avg_confidence': avg_confidence,
            'high_confidence_count': high_confidence_count,
            'high_confidence_percent': high_confidence_percent
        }
