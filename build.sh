#!/usr/bin/env bash
# Ensure Python 3.10 is active (Render respects runtime.txt, but this adds redundancy)
python --version
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir --ignore-installed
python -m spacy download en_core_web_sm
