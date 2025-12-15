#!/usr/bin/env python3
"""
Script to compile Marp presentation markdown to PDF.

Usage:
    python compile_to_pdf.py [input_file] [output_file]

Examples:
    python compile_to_pdf.py "presentation_1 copy.md" "presentation_1.pdf"
    python compile_to_pdf.py  # Uses defaults: "presentation_1 copy.md" -> "presentation_1.pdf"
"""

import subprocess
import sys
from pathlib import Path


def check_marp_installed():
    """Check if marp-cli is installed."""
    try:
        subprocess.run(
            ["marp", "--version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_instructions():
    """Print installation instructions."""
    print("Error: marp-cli is not installed.")
    print()
    print("Install it with one of the following methods:")
    print("  npm install -g @marp-team/marp-cli")
    print("  or")
    print("  brew install marp-cli")
    print("  or")
    print("  npx @marp-team/marp-cli (runs without installation)")
    print()


def compile_to_pdf(input_file: str, output_file: str):
    """Compile Marp markdown to PDF."""
    script_dir = Path(__file__).parent
    input_path = script_dir / input_file
    output_path = script_dir / output_file

    if not input_path.exists():
        print(f"Error: Presentation file '{input_file}' not found in {script_dir}")
        sys.exit(1)

    print(f"Compiling '{input_file}' to PDF...")
    print(f"Output: {output_file}")
    print()

    try:
        subprocess.run(
            [
                "marp",
                str(input_path),
                "--pdf",
                "--output",
                str(output_path),
                "--allow-local-files",
                "--pdf-outline",
            ],
            check=True,
            cwd=script_dir
        )
        print()
        print(f"✓ Successfully compiled to: {output_path}")
    except subprocess.CalledProcessError as e:
        print()
        print(f"✗ Compilation failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        install_instructions()
        sys.exit(1)


def main():
    """Main function."""
    input_file = sys.argv[1] if len(sys.argv) > 1 else "presentation_1 copy.md"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "presentation_1.pdf"

    if not check_marp_installed():
        install_instructions()
        sys.exit(1)

    compile_to_pdf(input_file, output_file)


if __name__ == "__main__":
    main()
