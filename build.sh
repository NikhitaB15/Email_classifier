#!/bin/bash
set -e

# Upgrade pip without using hashes
pip install --upgrade pip

# Install requirements without hash checking
pip install --no-cache-dir -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
