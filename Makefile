.PHONY: help install clean dataset evaluate test lint format all

# Default target
help:
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  dataset    - Build the complete dataset"
	@echo "  evaluate   - Run model evaluation on benchmarks"
	@echo "  test       - Run tests"
	@echo "  lint       - Run linting checks"
	@echo "  format     - Format code"
	@echo "  clean      - Clean generated files"
	@echo "  all        - Run install, dataset, and evaluate"

# Install dependencies
install:
	pip install -r requirements.txt

# Dataset construction pipeline
dataset: dataset/merged_personas.json

dataset/prepared/: dataset/raw/
	python dataset/dataset_construction/prepare.py --raw-dataset-path dataset/raw

dataset/transformed/: dataset/prepared/
	python dataset/dataset_construction/transform.py

dataset/merged_personas.json: dataset/transformed/
	python dataset/dataset_construction/merge.py

# Testing
test:
	python -m pytest tests/ -v

# Code quality
lint:
	flake8 dataset/ evaluation/ --max-line-length=100
	mypy dataset/ evaluation/ --ignore-missing-imports

format:
	black dataset/ evaluation/
	isort dataset/ evaluation/

# Clean up generated files
clean:
	rm -rf dataset/prepared/
	rm -rf dataset/transformed/
	rm -f dataset/merged_personas.json
	rm -f benchmark_results.json
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Complete pipeline
all: install dataset evaluate

# Quick development setup
dev-setup: install
	@echo "Development environment ready!"
	@echo "Set your API keys:"
	@echo "  export OPENAI_API_KEY=..."
	@echo "  export GEMINI_API_KEY=..."
	@echo "  export ANTHROPIC_API_KEY=..."
	@echo "  export HF_API_KEY=..."
