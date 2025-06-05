# Chat Simulator

A powerful video processing application that generates context-aware, Twitch-style chat overlays for your videos. Using local LLMs (Ollama) for chat generation and advanced video processing techniques, Chat Simulator creates engaging, realistic chat interactions that match your video content.

## 🌟 Features

- **Context-Aware Chat Generation**: Uses video content and audio transcription to generate relevant chat messages
- **Twitch-Style Overlay**: Professional-looking chat overlay with customizable positioning
- **Local LLM Processing**: Powered by Ollama for privacy and offline processing
- **Customizable Chat Settings**:
  - Message frequency
  - Number of chatters
  - Emote frequency
  - Spam level
- **Real-time Progress Updates**: WebSocket-based progress tracking
- **Optimized Performance**: Efficient video processing with resource management
- **Multiple Output Formats**: Support for various video codecs and resolutions

## 🏗️ Technical Architecture

### Core Components

- **VideoProcessor**: Handles video analysis, frame extraction, and audio transcription
- **ChatGenerator**: Generates context-aware chat messages using Ollama LLM
- **OverlayGenerator**: Creates and renders chat overlays on video frames

### Tech Stack

- **Backend**: FastAPI (Python)
- **Video Processing**: OpenCV, MoviePy
- **LLM Integration**: Ollama (Mistral/LLaVA models)
- **Audio Processing**: Whisper
- **Frontend**: HTML/CSS/JavaScript

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Ollama installed and running
- FFmpeg installed

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chat-simulator.git
cd chat-simulator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start Ollama and pull required models:
```bash
ollama serve
ollama pull mistral
ollama pull llava
```

5. Run the application:
```bash
python -m llama_chat.src.main
```

The application will be available at `http://localhost:8000`

## 💻 Usage

1. **Upload Video**: Select a video file through the web interface
2. **Configure Settings**: Adjust chat parameters to your preference
3. **Generate Preview**: Test chat generation with a small preview
4. **Process Video**: Generate the final video with chat overlay
5. **Download**: Get your processed video with the chat overlay

### Example Configuration

```python
chat_config = {
    'messages_per_minute': 30,
    'num_chatters': 100,
    'emote_frequency': 0.5,
    'spam_level': 0.2
}
```

## 📊 Performance Considerations

- **Video Length**: Optimized for videos up to 10 minutes
- **Resolution**: Recommended input resolution: 720p-1080p
- **Processing Time**: Approximately 2-3x video length
- **Memory Usage**: ~2GB RAM for standard processing
- **Storage**: Temporary files are automatically cleaned up

## 🖼️ Screenshots & Demos

[Placeholder for application screenshots]

[Placeholder for demo video]

## 🔧 Configuration

### Environment Variables

- `OUTPUTS_DIR`: Directory for processed videos
- `CACHE_DIR`: Directory for chat message cache
- `LOG_LEVEL`: Application logging level

### Processing Limits

```python
SAFE_PROCESSING_LIMITS = {
    'messages_per_minute': {'min': 10, 'max': 60, 'default': 30},
    'num_chatters': {'min': 20, 'max': 200, 'default': 100},
    'emote_frequency': {'min': 0.1, 'max': 0.8, 'default': 0.5},
    'spam_level': {'min': 0.1, 'max': 0.5, 'default': 0.2}
}
```

## 🔮 Future Improvements

- [ ] Support for multiple chat styles (YouTube, Discord, etc.)
- [ ] Custom emote/emotion detection
- [ ] Real-time chat generation during video processing
- [ ] GPU acceleration for faster processing
- [ ] Web-based video editor integration
- [ ] Chat message customization templates
- [ ] Batch processing for multiple videos

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM capabilities
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [OpenCV](https://opencv.org/) for video processing
- [Whisper](https://github.com/openai/whisper) for audio transcription 