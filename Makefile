# Makefile for Chat Simulator

.PHONY: install run dev clean lint

install:
	python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install -r llama_chat/requirements.txt

run:
	source venv/bin/activate && uvicorn llama_chat.src.main:app --host 0.0.0.0 --port 8000

dev:
	source venv/bin/activate && uvicorn llama_chat.src.main:app --reload --host 0.0.0.0 --port 8000

lint:
	flake8 llama_chat/src

clean:
	rm -rf venv llama_env outputs temp cache logs llama_chat.egg-info 