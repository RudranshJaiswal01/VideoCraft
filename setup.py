#!/usr/bin/env python3
"""
Setup script for AI Film Editor
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_spacy_model():
    """Download spaCy English model"""
    print("Downloading spaCy English model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    directories = [
        "data/input",
        "data/output", 
        "data/cache",
        "models/cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create .gitkeep files
        (Path(directory) / ".gitkeep").touch()

def main():
    """Main setup function"""
    print("🎬 Setting up AI Film Editor...")
    
    try:
        install_requirements()
        download_spacy_model()
        create_directories()
        
        print("✅ Setup completed successfully!")
        print("\nTo run the application:")
        print("streamlit run main.py")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
