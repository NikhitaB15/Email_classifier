#!/usr/bin/env bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir --ignore-installed
python -m spacy download en_core_web_sm

