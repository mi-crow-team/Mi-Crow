# Presentation Compilation

This directory contains Marp presentation files and scripts to compile them to PDF.

## Prerequisites

Install Marp CLI:

```bash
# Using npm (recommended)
npm install -g @marp-team/marp-cli

# Or using Homebrew (macOS)
brew install marp-cli

# Or use npx without installation
npx @marp-team/marp-cli
```

## Usage

### Bash Script

```bash
# Compile with default files
./compile_to_pdf.sh

# Compile with custom input/output
./compile_to_pdf.sh "presentation_1 copy.md" "output.pdf"
```

### Python Script

```bash
# Compile with default files
python compile_to_pdf.py

# Compile with custom input/output
python compile_to_pdf.py "presentation_1 copy.md" "output.pdf"
```

## Files

- `presentation_1 copy.md` - Main presentation file (Marp format)
- `compile_to_pdf.sh` - Bash script for compilation
- `compile_to_pdf.py` - Python script for compilation

## Output

The compiled PDF will be saved in the same directory as the input file.

## Notes

- The scripts automatically handle file paths relative to the script directory
- PDF outline is enabled for better navigation
- Local files are allowed (for images, etc.)
