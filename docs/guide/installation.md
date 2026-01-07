# Installation & Setup

This guide will help you install mi-crow and configure your environment for mechanistic interpretability research.

## System Requirements

- **Python**: 3.10, 3.11, or 3.12
- **PyTorch**: 2.8.0 or later
- **CUDA** (optional): For GPU acceleration
- **MPS** (optional): For Apple Silicon GPU support

## Installation Methods

### Using pip

```bash
pip install mi-crow
```

### Using uv (Recommended)

```bash
uv pip install mi-crow
```

### From Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/AdamKaniasty/Inzynierka.git
cd Inzynierka
pip install -e .
```

## Optional Dependencies

### Server Dependencies

For running the FastAPI server:

```bash
uv sync --group server
```

Or with pip:

```bash
pip install mi-crow[server]
```

### Documentation Dependencies

For building documentation locally:

```bash
uv sync --group docs
```

Or with pip:

```bash
pip install mi-crow[docs]
```

## Environment Setup

### Virtual Environment

We recommend using a virtual environment:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using uv
uv venv
source .venv/bin/activate
```

### Device Configuration

mi-crow automatically detects available devices. You can also explicitly set the device:

```python
import torch

# Check available device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Or for Apple Silicon:
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

### HuggingFace Setup

For downloading models from HuggingFace, you may need to authenticate:

```bash
huggingface-cli login
```

Set your cache directory (optional):

```bash
export HF_HOME=/path/to/huggingface/cache
```

## Verification

Test your installation:

```python
from mi_crow import ping

print(ping())  # Should print "pong"
```

Test with a simple model load:

```python
from mi_crow.language_model import LanguageModel
from mi_crow.store import LocalStore
import torch

store = LocalStore(base_path="./store")

# Use GPU when available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

lm = LanguageModel.from_huggingface(
    "sshleifer/tiny-gpt2",
    store=store,
    device=device,
)
print("Installation successful!")
```

## Troubleshooting

### PyTorch Installation Issues

If you encounter PyTorch installation problems:

1. Visit [pytorch.org](https://pytorch.org) for platform-specific instructions
2. For CUDA support, ensure your CUDA version matches PyTorch's requirements
3. For Apple Silicon, PyTorch should automatically support MPS

### Import Errors

If you get import errors:

1. Ensure you're in the correct virtual environment
2. Verify installation: `pip list | grep mi-crow`
3. Try reinstalling: `pip install --force-reinstall mi-crow`

### GPU Not Detected

If CUDA is not detected:

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())   # Number of GPUs
```

If False, check:
- CUDA drivers are installed
- PyTorch was installed with CUDA support
- GPU is compatible with your CUDA version

## Next Steps

Once installation is complete, proceed to:

- **[Quick Start](quickstart.md)** - Run your first example
- **[Core Concepts](core-concepts.md)** - Understand the fundamentals

