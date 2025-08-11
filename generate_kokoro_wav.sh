#!/bin/bash

# Exit on error
set -e

# Check for dependencies
if ! command -v jq &> /dev/null; then
  echo "❌ 'jq' is not installed. Please install it: sudo apt install jq"
  exit 1
fi

# Input arguments
INPUT_FILE="$1"
VOICE="$2"

# Validation
if [ ! -f "$INPUT_FILE" ]; then
  echo "❌ Input file not found: $INPUT_FILE"
  exit 1
fi

if [ -z "$VOICE" ]; then
  echo "❌ Voice not specified"
  exit 1
fi

# Extract raw text and escape it for JSON
TEXT=$(<"$INPUT_FILE")
ESCAPED_TEXT=$(jq -Rn --arg text "$TEXT" '$text')

# Output path
OUTPUT_FILE="${INPUT_FILE%.txt}.wav"

# Send POST request to Kokoro TTS API
echo "🎤 Generating narration:"
echo "  File: $INPUT_FILE"
echo "  Voice: $VOICE"
echo "  Output: $OUTPUT_FILE"

RESPONSE=$(curl -s -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"input\": $ESCAPED_TEXT, \"voice\": \"$VOICE\", \"model\": \"kokoro\", \"response_format\": \"wav\", \"stream\": false}" \
  --output "$OUTPUT_FILE")

# Validate result
FILESIZE=$(stat -c%s "$OUTPUT_FILE")
if [ "$FILESIZE" -lt 1000 ]; then
  echo "❌ Output file is too small. Likely an error occurred."
  echo "🔁 Response (might contain error info):"
  echo "$RESPONSE"
else
  echo "✅ Narration saved to $OUTPUT_FILE"
fi
