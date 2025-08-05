# 🎬 AI Film Editor - Smart Cut & Transition Suggestions

An intelligent video editing assistant that analyzes video content and scripts to suggest optimal cut points and transitions using state-of-the-art AI models.

## ✨ Features

- **Scene Change Detection**: Automatically identify visual scene boundaries
- **Emotion Analysis**: Detect emotional beats in both script and speech
- **Speaker Recognition**: Identify speaker changes in dialogue
- **Smart Transitions**: Recommend appropriate transition types based on content
- **Interactive Timeline**: Visualize suggestions on an interactive timeline
- **No Training Required**: Uses pre-trained Hugging Face models

## 🚀 Quick Start

1. **Clone and Setup**:
git clone <your-repo-url>
cd ai-film-editor
python setup.py


2. **Run the Application**:
streamlit run main.py

3. **Upload and Process**:
- Upload your video file (.mp4, .avi, .mov, .mkv)
- Upload corresponding script file (.txt, .srt)
- Click "Process Video" and review AI suggestions

## 🤖 AI Models Used

- **cardiffnlp/twitter-roberta-base-emotion**: Text emotion analysis
- **ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition**: Speech emotion
- **openai/clip-vit-base-patch32**: Visual content understanding
- **trpakov/vit-face-expression**: Facial emotion recognition

## 📁 Project Structure
```
ai-film-editor/
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml
├── main.py
├── src/
│   ├── __init__.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── video_analyzer.py
│   │   ├── script_parser.py
│   │   ├── audio_analyzer.py
│   │   └── scene_detector.py
│   ├── ai_models/
│   │   ├── __init__.py
│   │   ├── emotion_detector.py
│   │   ├── sentiment_analyzer.py
│   │   └── visual_analyzer.py
│   ├── suggestions/
│   │   ├── __init__.py
│   │   ├── cut_suggester.py
│   │   └── transition_recommender.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── timeline_viewer.py
│   │   └── suggestion_panel.py
│   └── utils/
│       ├── __init__.py
│       ├── file_handler.py
│       └── timeline_sync.py
├── data/
│   ├── input/
│   ├── output/
│   └── cache/
└── tests/
    └── __init__.py
```

## ⚙️ Configuration

Modify `config.yaml` to adjust:
- Model preferences
- Processing thresholds
- Suggestion sensitivity
- Output formats

## 🔧 Advanced Usage

### Custom Thresholds
Adjust sensitivity in the sidebar:
- **Emotion Change Sensitivity**: How dramatic emotional changes need to be
- **Scene Change Sensitivity**: How different scenes need to be visually

### Suggestion Types
Filter suggestions by:
- Scene changes
- Emotional beats
- Speaker changes

## 📊 Output

The application provides:
- Interactive timeline with suggestions
- Confidence scores for each suggestion
- Reasoning for cut and transition recommendations
- Export options for editing software integration

## 🛠️ Development

To add new features:
1. Add new processors in `src/processors/`
2. Implement AI models in `src/ai_models/`
3. Create suggestion engines in `src/suggestions/`
4. Update UI components in `src/ui/`

## 📝 License

MIT License - Feel free to use and modify for your projects.
