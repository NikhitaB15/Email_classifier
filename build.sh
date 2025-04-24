#!/bin/bash
set -e

# Upgrade pip
pip install --upgrade pip

# Install requirements without hash checking
pip install --no-cache-dir fastapi uvicorn spacy transformers torch scikit-learn pandas

# Download spaCy model directly
python -m spacy download en_core_web_sm

# Print success message
echo "Build completed successfully!"
