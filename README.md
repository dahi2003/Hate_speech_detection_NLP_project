# Hate Speech Detection NLP Project

A Flask-based web application that detects hate speech, offensive language, and neutral content from text, audio, and video inputs using machine learning and natural language processing.

## 🎯 Project Overview

This project implements a multi-modal hate speech detection system that can classify content into three categories:
- **Hate Speech**: Content that promotes violence or discrimination
- **Offensive Language**: Vulgar or offensive content without hate speech
- **Neither**: Neutral, non-offensive content

The application supports three input methods:
1. **Text Input**: Direct text/tweet submission
2. **Audio Input**: Speech-to-text transcription and classification
3. **Video Input**: Audio extraction, transcription, and classification

## ✨ Features

- 🎤 **Multi-modal Input Support**: Text, audio, and video file processing
- 🔍 **NLP Text Cleaning**: Removes URLs, mentions, hashtags, punctuation, and numbers
- 🧠 **Machine Learning Classification**: TF-IDF vectorization with pre-trained model
- 🌐 **Web Interface**: User-friendly Flask-based web application
- 📝 **Text Preprocessing**: Comprehensive text normalization pipeline
- 🎙️ **Speech Recognition**: Google Speech Recognition API integration
- 🎬 **Video Processing**: Automatic audio extraction from video files

## 📋 Requirements

- Python 3.7+
- Flask
- scikit-learn
- speech_recognition
- moviepy
- pandas, nltk, numpy
- matplotlib, seaborn

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dahi2003/Hate_speech_detection_NLP_project.git
cd Hate_speech_detection_NLP_project
