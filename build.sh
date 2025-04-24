#!/bin/bash
set -e

# Upgrade pip
pip install --upgrade pip

# Install requirements without hash checking
pip install --no-cache-dir fastapi uvicorn spacy transformers torch scikit-learn pandas

# Download spaCy model directly
python -m spacy download en_core_web_sm

# Create directories if they don't exist
mkdir -p final_files

# Copy necessary files to ensure imports work
if [ -f "model_utils.py" ]; then
    cp model_utils.py final_files/
fi

if [ -f "pii_utils.py" ]; then
    cp pii_utils.py final_files/
fi

# Print success message
echo "Build completed successfully!"
