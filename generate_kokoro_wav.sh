#!/bin/bash

# Exit on error
set -euo pipefail

# Check for dependencies
if ! command -v jq &> /dev/null; then
  echo "❌ 'jq' is not installed. Please install it: sudo apt install jq"
  exit 1
fi

# ---- Inputs ----
INPUT_FILE="${1:-}"
VOICE="${2:-}"
SPEED_RAW="${3:-}"         # optional (numeric, e.g., 0.95)
OUTPUT_RAW="${4:-}"        # optional (filename/path)

# ---- Validation ----
if [[ -z "$INPUT_FILE" || ! -f "$INPUT_FILE" ]]; then
  echo "❌ Input file not found: $INPUT_FILE"
  echo "   Usage: $0 <input.txt> <voice_name> [speed] [output.wav]"
  exit 1
fi

if [[ -z "$VOICE" ]]; then
  echo "❌ Voice not specified"
  echo "   Usage: $0 <input.txt> <voice_name> [speed] [output.wav]"
  exit 1
fi

# ---- Speed handling ----
SPEED_FLAG=""
if [[ -n "$SPEED_RAW" ]]; then
  if [[ "$SPEED_RAW" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    SPEED_FLAG="$SPEED_RAW"
  else
    echo "⚠️  Ignoring invalid speed '$SPEED_RAW' (must be numeric like 0.95 or 1.0)."
  fi
fi

# ---- Output path ----
if [[ -n "$OUTPUT_RAW" ]]; then
  OUTPUT_FILE="$OUTPUT_RAW"
else
  base="${INPUT_FILE%.*}"
  [[ -z "$base" ]] && base="$INPUT_FILE"
  OUTPUT_FILE="${base}.wav"
fi

# Ensure .wav extension (if user passed a name without it)
if [[ "${OUTPUT_FILE##*.}" != "wav" ]]; then
  OUTPUT_FILE="${OUTPUT_FILE}.wav"
fi

# Create output directory if needed
outdir="$(dirname "$OUTPUT_FILE")"
mkdir -p "$outdir"

# ---- Prepare payload ----
TEXT="$(<"$INPUT_FILE")"

# Build valid JSON with jq (handles quotes in SSML safely)
if [[ -n "$SPEED_FLAG" ]]; then
  PAYLOAD=$(jq -n \
    --arg input "$TEXT" \
    --arg voice "$VOICE" \
    --arg model "kokoro" \
    --arg fmt "wav" \
    --argjson stream false \
    --argjson speed "$SPEED_FLAG" \
    '{input:$input, voice:$voice, model:$model, response_format:$fmt, stream:$stream, speed:$speed}')
else
  PAYLOAD=$(jq -n \
    --arg input "$TEXT" \
    --arg voice "$VOICE" \
    --arg model "kokoro" \
    --arg fmt "wav" \
    --argjson stream false \
    '{input:$input, voice:$voice, model:$model, response_format:$fmt, stream:$stream}')
fi

# ---- Info ----
echo "🎤 Generating narration:"
echo "  File:   $INPUT_FILE"
echo "  Voice:  $VOICE"
echo "  Speed:  ${SPEED_FLAG:-default (1.0)}"
echo "  Output: $OUTPUT_FILE"

# ---- POST to Kokoro ----
HTTP_CODE=$(curl -sS -o "$OUTPUT_FILE" -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -X POST "http://localhost:8880/v1/audio/speech" \
  -d "$PAYLOAD")

# ---- Validate result ----
if [[ "$HTTP_CODE" != "200" ]]; then
  echo "❌ HTTP $HTTP_CODE"
  echo "🔁 Server response:"
  cat "$OUTPUT_FILE" || true
  exit 1
fi

FILESIZE=$(stat -c%s "$OUTPUT_FILE" 2>/dev/null || echo 0)
if [[ "$FILESIZE" -lt 1000 ]]; then
  echo "❌ Output file is too small ($FILESIZE bytes). Likely an error occurred."
  echo "🔁 Server response (if any):"
  cat "$OUTPUT_FILE" || true
  exit 1
fi

echo "✅ Narration saved to $OUTPUT_FILE"
