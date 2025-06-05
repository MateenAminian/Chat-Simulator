from setuptools import setup, find_packages

setup(
    name="llama_chat",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart",
        "ollama",
        "openai-whisper",
        "transformers",
        "torch",
        "opencv-python",
        "moviepy",
        "numpy",
        "Pillow",
        # other dependencies...
    ],
) 