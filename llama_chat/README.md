# StreamChatter

A powerful video processing application that generates context-aware, Twitch-style chat overlays for your videos. Using local LLMs (Ollama) for chat generation and advanced video processing techniques, StreamChatter creates engaging, realistic chat interactions that match your video content.

## Prerequisites

- Python 3.10+
- Ollama installed on your system ([Install Ollama](https://ollama.ai/download))
- Virtual environment (recommended)

## Setup Development Environment

1. Clone the repository and navigate to the project directory:
```bash
cd llama_chat
```

2. Create and activate a virtual environment:
```bash
python -m venv llama_env
source llama_env/bin/activate  # On Windows: llama_env\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Start the Ollama service in a separate terminal:
```bash
ollama serve
```

## Running the Application

1. Make sure Ollama is running (you should see the Ollama icon in your menu bar)

2. Start the FastAPI server:
```bash
uvicorn src.main:app --reload
```

3. Open your browser and navigate to:
```
http://127.0.0.1:8000
```

## Project Structure

```
llama_chat/
├── README.md
├── setup.py
└── src/
    ├── __init__.py
    ├── main.py
    ├── static/
    │   └── index.html
    └── core/
        ├── __init__.py
        ├── chat_generator.py
        ├── video_processor.py
        └── overlay_generator.py
```

## Development Notes

- The `--reload` flag enables auto-reload when code changes
- Static files are served from `src/static/`
- Temporary files and uploads are stored in `src/temp/`
- Logs are written to `app.log`

## Common Issues

1. If you see "Connection refused" errors:
   - Make sure Ollama is running (`ollama serve`)
   - Wait a few seconds for Ollama to initialize

2. If static files aren't found:
   - Verify that `index.html` exists in `src/static/`
   - Check file permissions

3. If imports fail:
   - Make sure you've installed the package in dev mode (`pip install -e .`)
   - Verify you're running from the correct directory

## License

[Your License Here]