import streamlit as st
import yaml
import os
import tempfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from src.processors.video_analyzer import VideoAnalyzer
from src.processors.script_parser import ScriptParser
from src.processors.audio_analyzer import AudioAnalyzer
from src.processors.scene_detector import SceneDetector
from src.ai_models.emotion_detector import EmotionDetector
from src.suggestions.cut_suggester import CutSuggester
from src.suggestions.transition_recommender import TransitionRecommender
from src.ui.timeline_viewer import TimelineViewer
from src.ui.suggestion_panel import SuggestionPanel
from src.utils.file_handler import FileHandler
from src.utils.timeline_sync import TimelineSync

def load_config():
    """Load configuration from YAML file."""
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error("config.yaml not found. Please ensure the configuration file exists.")
        st.error("Configuration file not found. Please check your setup.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error loading config.yaml: {e}")
        st.error("Error in configuration file format.")
        return None

def initialize_components(config):
    """Initialize all processing components."""
    try:
        components = {
            'video_analyzer': VideoAnalyzer(config),
            'script_parser': ScriptParser(config),
            'audio_analyzer': AudioAnalyzer(config),
            'scene_detector': SceneDetector(config),
            'emotion_detector': EmotionDetector(config),
            'cut_suggester': CutSuggester(config),
            'transition_recommender': TransitionRecommender(config),
            'timeline_viewer': TimelineViewer(config),
            'suggestion_panel': SuggestionPanel(config),
            'file_handler': FileHandler(config),
            'timeline_sync': TimelineSync(config)
        }
        return components
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        st.error(f"Error initializing AI components: {e}")
        return None

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="AI Film Editor",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 AI Film Editor</h1>
        <p>Smart Cut & Transition Suggestions Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load configuration
    config = load_config()
    if config is None:
        return
    
    # Initialize components
    with st.spinner("Initializing AI components..."):
        components = initialize_components(config)
    
    if components is None:
        return
    
    # Sidebar for file uploads and settings
    with st.sidebar:
        st.header("📁 File Upload")
        
        # Video file upload
        video_file = st.file_uploader(
            "Upload Video File",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload your raw video footage (max 500MB)"
        )
        
        # Script file upload
        script_file = st.file_uploader(
            "Upload Script File",
            type=['txt', 'srt', 'vtt'],
            help="Upload the corresponding script or subtitles"
        )
        
        st.divider()
        
        # Processing options
        st.header("⚙️ Analysis Options")
        
        analyze_video = st.checkbox("Analyze Video Content", value=True)
        analyze_audio = st.checkbox("Analyze Audio", value=True)
        analyze_script = st.checkbox("Analyze Script", value=bool(script_file))
        
        st.divider()
        
        # Sensitivity settings
        st.header("🎛️ Sensitivity Settings")
        
        emotion_threshold = st.slider(
            "Emotion Change Sensitivity", 
            0.1, 1.0, 0.3, 0.1,
            help="How sensitive to emotional changes in content"
        )
        
        scene_threshold = st.slider(
            "Scene Change Sensitivity", 
            0.1, 1.0, 0.4, 0.1,
            help="How sensitive to visual scene changes"
        )
        
        min_cut_interval = st.slider(
            "Minimum Cut Interval (seconds)", 
            0.5, 10.0, 2.0, 0.5,
            help="Minimum time between suggested cuts"
        )
    
    # Main processing area
    if video_file is not None:
        # Display file information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>📹 Video File</h4>
                <p>{}</p>
                <p>Size: {:.1f} MB</p>
            </div>
            """.format(video_file.name, video_file.size / (1024*1024)), unsafe_allow_html=True)
        
        with col2:
            if script_file:
                st.markdown("""
                <div class="metric-card">
                    <h4>📝 Script File</h4>
                    <p>{}</p>
                    <p>Size: {:.1f} KB</p>
                </div>
                """.format(script_file.name, script_file.size / 1024), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <h4>📝 Script File</h4>
                    <p>No script uploaded</p>
                    <p>Script analysis disabled</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>🎯 Analysis Mode</h4>
                <p>Video: {}</p>
                <p>Audio: {}</p>
                <p>Script: {}</p>
            </div>
            """.format(
                "✅" if analyze_video else "❌",
                "✅" if analyze_audio else "❌", 
                "✅" if analyze_script else "❌"
            ), unsafe_allow_html=True)
        
        st.divider()
        
        # Update config with user preferences
        config['suggestions']['emotion_change_threshold'] = emotion_threshold
        config['suggestions']['scene_change_threshold'] = scene_threshold
        config['suggestions']['minimum_cut_interval'] = min_cut_interval
        
        # Process button
        if st.button("🚀 Analyze Video & Generate Suggestions", type="primary", use_container_width=True):
            
            # Save uploaded files
            with st.status("Preparing files...") as status:
                file_handler = components['file_handler']
                
                video_path = file_handler.save_uploaded_file(video_file, 'video')
                if not video_path:
                    st.error("Failed to save video file")
                    return
                
                script_path = None
                if script_file and analyze_script:
                    script_path = file_handler.save_uploaded_file(script_file, 'script')
                    if not script_path:
                        st.warning("Failed to save script file - continuing without script analysis")
                
                status.update(label="Files prepared successfully!", state="complete")
            
            # Initialize analysis results
            video_analysis = {}
            script_analysis = {}
            audio_analysis = {}
            
            try:
                # Video Analysis
                if analyze_video:
                    with st.status("Analyzing video content...") as status:
                        video_analyzer = components['video_analyzer']
                        scene_detector = components['scene_detector']
                        
                        # Scene detection
                        scene_changes = scene_detector.detect_scenes(video_path, method='combined')
                        video_analysis['scene_changes'] = scene_changes
                        
                        # Video timeline analysis
                        timeline_analysis = video_analyzer.analyze_video_timeline(video_path)
                        video_analysis.update(timeline_analysis)
                        
                        status.update(
                            label=f"Video analysis complete! Found {len(scene_changes)} scene changes",
                            state="complete"
                        )
                
                # Audio Analysis  
                if analyze_audio:
                    with st.status("Analyzing audio content...") as status:
                        audio_analyzer = components['audio_analyzer']
                        
                        # Extract audio from video
                        audio_path = audio_analyzer.extract_audio_from_video(video_path)
                        
                        # Audio feature extraction
                        audio_features = audio_analyzer.extract_audio_features(audio_path)
                        audio_analysis['features'] = audio_features
                        
                        # Speech emotion analysis
                        speech_emotions = audio_analyzer.analyze_speech_emotion(audio_path)
                        audio_analysis['speech_emotions'] = speech_emotions
                        
                        # Speaker change detection
                        speaker_changes = audio_analyzer.detect_speaker_changes(audio_path)
                        audio_analysis['speaker_changes'] = speaker_changes
                        
                        # Audio energy analysis
                        energy_timeline = audio_analyzer.analyze_audio_energy(audio_path)
                        audio_analysis['energy_timeline'] = energy_timeline
                        
                        status.update(
                            label=f"Audio analysis complete! Found {len(speaker_changes)} potential speaker changes",
                            state="complete"
                        )
                
                # Script Analysis
                if analyze_script and script_path:
                    with st.status("Analyzing script content...") as status:
                        script_parser = components['script_parser']
                        timeline_sync = components['timeline_sync']
                        
                        # Parse script
                        dialogue_data = script_parser.parse_script_file(script_path)
                        
                        # Emotion analysis
                        dialogue_with_emotions = script_parser.analyze_emotions(dialogue_data)
                        
                        # Align script to video duration
                        video_duration = audio_analysis.get('features', {}).get('duration', 60)
                        aligned_dialogue = timeline_sync.align_script_to_video(
                            dialogue_with_emotions, video_duration
                        )
                        
                        # Detect emotional beats
                        emotional_beats = script_parser.detect_emotional_beats(aligned_dialogue)
                        
                        script_analysis['dialogue_data'] = aligned_dialogue
                        script_analysis['emotional_beats'] = emotional_beats
                        
                        status.update(
                            label=f"Script analysis complete! Found {len(emotional_beats)} emotional beats",
                            state="complete"
                        )
                
                # Generate AI Suggestions
                with st.status("Generating AI suggestions...") as status:
                    cut_suggester = components['cut_suggester']
                    transition_recommender = components['transition_recommender']
                    emotion_detector = components['emotion_detector']
                    
                    # Generate cut suggestions
                    cut_suggestions = cut_suggester.generate_suggestions(
                        video_analysis, script_analysis, audio_analysis
                    )
                    
                    # Generate transition suggestions
                    transition_suggestions = transition_recommender.suggest_transitions(
                        cut_suggestions,
                        audio_analysis.get('speech_emotions', []),
                        video_analysis,
                        script_analysis
                    )
                    
                    status.update(
                        label=f"AI analysis complete! Generated {len(cut_suggestions)} cut suggestions",
                        state="complete"
                    )
                
                # Display Results
                if cut_suggestions:
                    st.success(f"✅ Analysis complete! Generated {len(cut_suggestions)} cut suggestions and {len(transition_suggestions)} transition recommendations.")
                    
                    st.header("📊 Analysis Results")
                    
                    # Timeline Visualization
                    timeline_viewer = components['timeline_viewer']
                    
                    # Create main timeline
                    timeline_fig = timeline_viewer.create_timeline_visualization(
                        audio_analysis.get('features', {}).get('duration', 60),
                        cut_suggestions,
                        transition_suggestions,
                        audio_analysis.get('speech_emotions', [])
                    )
                    
                    st.plotly_chart(timeline_fig, use_container_width=True)
                    
                    # Suggestion summary chart
                    summary_fig = timeline_viewer.create_suggestion_summary_chart(cut_suggestions)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(summary_fig, use_container_width=True)
                    
                    with col2:
                        # Display statistics
                        st.subheader("📈 Analysis Statistics")
                        
                        total_duration = audio_analysis.get('features', {}).get('duration', 0)
                        avg_confidence = sum(s.confidence for s in cut_suggestions) / len(cut_suggestions)
                        
                        st.metric("Video Duration", f"{total_duration:.1f} seconds")
                        st.metric("Total Suggestions", len(cut_suggestions))
                        st.metric("Average Confidence", f"{avg_confidence:.1%}")
                        st.metric("High Confidence Cuts", len([s for s in cut_suggestions if s.confidence > 0.7]))
                    
                    # Interactive Suggestion Panel
                    suggestion_panel = components['suggestion_panel']
                    
                    # Render suggestion controls
                    filters = suggestion_panel.render_suggestion_controls()
                    
                    # Render suggestion list
                    selected_suggestions = suggestion_panel.render_suggestion_list(
                        cut_suggestions, filters, transition_suggestions
                    )
                    
                    # Batch actions
                    if selected_suggestions:
                        batch_result = suggestion_panel.render_batch_actions(selected_suggestions)
                        
                        if batch_result.get('action') == 'export':
                            # Export functionality
                            export_format = st.selectbox("Export Format", ['json', 'csv', 'txt'])
                            
                            if st.button("Download Export"):
                                export_path = file_handler.export_suggestions(
                                    selected_suggestions, export_format
                                )
                                if export_path:
                                    with open(export_path, 'rb') as f:
                                        st.download_button(
                                            f"Download {export_format.upper()} File",
                                            data=f.read(),
                                            file_name=Path(export_path).name,
                                            mime=f"application/{export_format}"
                                        )
                    
                    # Render analytics
                    suggestion_panel.render_suggestion_analytics(cut_suggestions)
                    
                else:
                    st.warning("No cut suggestions were generated. Try adjusting the sensitivity settings.")
                
            except Exception as e:
                logger.error(f"Error during processing: {e}")
                st.error(f"An error occurred during processing: {e}")
            
            finally:
                # Cleanup temporary files
                try:
                    file_handler.cleanup_file(video_path)
                    if script_path:
                        file_handler.cleanup_file(script_path)
                    if 'audio_path' in locals():
                        file_handler.cleanup_file(audio_path)
                except:
                    pass
    
    else:
        # Welcome screen when no video is uploaded
        st.markdown("""
        ## 🎯 How AI Film Editor Works
        
        Our AI-powered film editing assistant analyzes your video content and scripts to suggest intelligent cut points and transitions that enhance your storytelling.
        
        ### 📋 Getting Started:
        
        1. **📹 Upload Video**: Upload your raw footage (MP4, AVI, MOV, MKV)
        2. **📝 Upload Script**: Optionally upload your script or subtitles
        3. **⚙️ Configure Settings**: Adjust sensitivity settings in the sidebar
        4. **🚀 Process**: Click "Analyze Video" to generate AI suggestions
        5. **✨ Review & Apply**: Review suggestions and export your cut list
        
        ### 🤖 AI Analysis Features:
        
        - **🎬 Scene Detection**: Automatically identify visual scene changes
        - **😊 Emotion Analysis**: Detect emotional beats in dialogue and speech
        - **🗣️ Speaker Changes**: Identify when speakers change in audio
        - **🎵 Audio Cues**: Analyze audio energy and silence for cut opportunities
        - **🔄 Smart Transitions**: Recommend appropriate transition types
        
        ### 🎨 Supported Formats:
        
        **Video**: MP4, AVI, MOV, MKV, WebM  
        **Scripts**: TXT, SRT, VTT  
        **Export**: JSON, CSV, TXT
        
        ---
        
        *Ready to revolutionize your editing workflow? Upload a video file to begin!*
        """)
        
        # Feature showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **🎬 Scene Analysis**
            - Visual scene detection
            - Shot type identification  
            - Mood analysis
            - Composition evaluation
            """)
        
        with col2:
            st.info("""
            **🗣️ Audio Intelligence**
            - Speech emotion recognition
            - Speaker change detection
            - Audio energy analysis
            - Silence detection
            """)
        
        with col3:
            st.info("""
            **📝 Script Understanding**
            - Dialogue emotion analysis
            - Character analysis
            - Emotional beat detection
            - Timeline synchronization
            """)

if __name__ == "__main__":
    main()
