#!/bin/bash
# Run AlfWorld AI annotation with required API keys
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY before running}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY before running}"

exec conda run -n Fraud python ai_annotate.py "$@"
