import streamlit as st
import yaml
from pathlib import Path
import tempfile
import os

from src.processors.video_analyzer import VideoAnalyzer
from src.processors.script_parser import ScriptParser
from src.processors.audio_analyzer import AudioAnalyzer
from src.suggestions.cut_suggester import CutSuggester
from src.suggestions.transition_recommender import TransitionRecommender
from src.ui.timeline_viewer import TimelineViewer

def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="AI Film Editor",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎬 AI Film Editor - Smart Cut & Transition Suggestions")
    st.markdown("Upload your video and script to get intelligent editing suggestions powered by AI")
    
    # Load configuration
    config = load_config()
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 Upload Files")
        
        # Video upload
        video_file = st.file_uploader(
            "Upload Video File",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload your raw video footage"
        )
        
        # Script upload
        script_file = st.file_uploader(
            "Upload Script File",
            type=['txt', 'srt'],
            help="Upload the corresponding script"
        )
        
        # Processing options
        st.header("⚙️ Processing Options")
        
        analyze_video = st.checkbox("Analyze Video Content", value=True)
        analyze_audio = st.checkbox("Analyze Audio", value=True)
        analyze_script = st.checkbox("Analyze Script", value=True)
        
        # Suggestion sensitivity
        st.subheader("Suggestion Sensitivity")
        emotion_threshold = st.slider("Emotion Change Sensitivity", 0.1, 1.0, 0.3)
        scene_threshold = st.slider("Scene Change Sensitivity", 0.1, 1.0, 0.4)
        
    # Main processing area
    if video_file is not None:
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_file.read())
            video_path = tmp_video.name
            
        script_path = None
        if script_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w') as tmp_script:
                tmp_script.write(script_file.read().decode('utf-8'))
                script_path = tmp_script.name
        
        # Update config with user preferences
        config['suggestions']['emotion_change_threshold'] = emotion_threshold
        config['suggestions']['scene_change_threshold'] = scene_threshold
        
        # Initialize processors
        video_analyzer = VideoAnalyzer(config) if analyze_video else None
        script_parser = ScriptParser(config) if analyze_script and script_path else None
        audio_analyzer = AudioAnalyzer(config) if analyze_audio else None
        cut_suggester = CutSuggester(config)
        transition_recommender = TransitionRecommender(config)
        timeline_viewer = TimelineViewer(config)
        
        # Processing button
        if st.button("🚀 Process Video", type="primary"):
            with st.spinner("Analyzing video content... This may take a few minutes."):
                
                # Initialize results
                video_analysis = {}
                script_analysis = {}
                audio_analysis = {}
                
                # Video analysis
                if video_analyzer:
                    with st.status("Analyzing video content..."):
                        scene_changes = video_analyzer.detect_scene_changes(video_path)
                        video_analysis['scene_changes'] = scene_changes
                        st.write(f"Detected {len(scene_changes)} scene changes")
                
                # Script analysis
                if script_parser:
                    with st.status("Analyzing script..."):
                        dialogue_data = script_parser.parse_script_file(script_path)
                        dialogue_with_emotions = script_parser.analyze_emotions(dialogue_data)
                        emotional_beats = script_parser.detect_emotional_beats(dialogue_with_emotions)
                        
                        script_analysis['dialogue_data'] = dialogue_with_emotions
                        script_analysis['emotional_beats'] = emotional_beats
                        st.write(f"Analyzed {len(dialogue_data)} dialogue lines")
                        st.write(f"Detected {len(emotional_beats)} emotional beats")
                
                # Audio analysis
                if audio_analyzer:
                    with st.status("Analyzing audio..."):
                        audio_features = audio_analyzer.extract_audio_features(video_path)
                        speech_emotions = audio_analyzer.analyze_speech_emotion(audio_features['audio_path'])
                        speaker_changes = audio_analyzer.detect_speaker_changes(audio_features['audio_path'])
                        
                        audio_analysis['features'] = audio_features
                        audio_analysis['speech_emotions'] = speech_emotions
                        audio_analysis['speaker_changes'] = speaker_changes
                        st.write(f"Detected {len(speaker_changes)} potential speaker changes")
                
                # Generate suggestions
                with st.status("Generating AI suggestions..."):
                    cut_suggestions = cut_suggester.generate_suggestions(
                        video_analysis, script_analysis, audio_analysis
                    )
                    
                    transition_suggestions = transition_recommender.suggest_transitions(
                        cut_suggestions,
                        audio_analysis.get('speech_emotions', []),
                        video_analysis
                    )
                    
                    st.success(f"Generated {len(cut_suggestions)} cut suggestions and {len(transition_suggestions)} transition recommendations!")
                
                # Display results
                st.header("📊 Analysis Results")
                
                # Timeline visualization
                if cut_suggestions:
                    fig = timeline_viewer.create_timeline_visualization(
                        audio_analysis.get('features', {}).get('duration', 60),
                        cut_suggestions,
                        transition_suggestions,
                        audio_analysis.get('speech_emotions', [])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Suggestion panel
                    timeline_viewer.render_suggestion_panel(cut_suggestions)
                    
                    # Export options
                    st.header("📤 Export Options")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Export Timeline"):
                            st.info("Timeline export functionality to be implemented")
                            
                    with col2:
                        if st.button("Export Cut List"):
                            st.info("Cut list export functionality to be implemented")
                
                else:
                    st.warning("No cut suggestions generated. Try adjusting sensitivity settings.")
        
        # Cleanup temporary files
        try:
            os.unlink(video_path)
            if script_path:
                os.unlink(script_path)
        except:
            pass
            
    else:
        # Welcome screen
        st.markdown("""
        ## 🎯 How it works:
        
        1. **Upload** your video file and script
        2. **Configure** analysis settings in the sidebar
        3. **Process** the content with AI models
        4. **Review** intelligent cut and transition suggestions
        5. **Export** your editing timeline
        
        ### 🤖 AI Models Used:
        - **Text Emotion**: DistilRoBERTa for sentiment analysis
        - **Speech Emotion**: Wav2Vec2 for vocal emotion recognition
        - **Visual Analysis**: CLIP for scene understanding
        - **Scene Detection**: Computer vision algorithms
        
        ### ✨ Features:
        - Scene change detection
        - Emotional beat identification
        - Speaker change recognition
        - Smart transition recommendations
        - Interactive timeline visualization
        """)

if __name__ == "__main__":
    main()
