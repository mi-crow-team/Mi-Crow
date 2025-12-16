#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESENTATION_FILE="${1:-presentation_1 copy.md}"
OUTPUT_FILE="${2:-presentation_1.pdf}"

if [ ! -f "$SCRIPT_DIR/$PRESENTATION_FILE" ]; then
    echo "Error: Presentation file '$PRESENTATION_FILE' not found in $SCRIPT_DIR"
    exit 1
fi

if ! command -v marp &> /dev/null; then
    echo "Error: marp-cli is not installed."
    echo ""
    echo "Install it with one of the following methods:"
    echo "  npm install -g @marp-team/marp-cli"
    echo "  or"
    echo "  brew install marp-cli"
    echo ""
    exit 1
fi

echo "Compiling '$PRESENTATION_FILE' to PDF..."
echo "Output: $OUTPUT_FILE"
echo ""

cd "$SCRIPT_DIR"

marp "$PRESENTATION_FILE" \
    --pdf \
    --output "$OUTPUT_FILE" \
    --allow-local-files \
    --theme-set ./ \
    --pdf-outline

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Successfully compiled to: $SCRIPT_DIR/$OUTPUT_FILE"
else
    echo ""
    echo "✗ Compilation failed"
    exit 1
fi
