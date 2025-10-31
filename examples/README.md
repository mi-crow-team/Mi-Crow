# SAE Examples

This directory contains three Jupyter notebooks that demonstrate a complete workflow for training Sparse Autoencoders (SAEs) and working with neuron-text associations.

## Overview

The examples build upon each other, with each notebook using outputs from the previous one:

1. **01_train_sae_model.ipynb** - Train a SAE model on language model activations
2. **02_attach_sae_and_save_texts.ipynb** - Attach SAE to model and collect top texts for neurons
3. **03_load_and_manipulate_concepts.ipynb** - Load and manipulate concept dictionaries

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install torch transformers datasets
```

## Usage

### 1. Training SAE Model

Run the first notebook to train a SAE:

```bash
jupyter notebook examples/01_train_sae_model.ipynb
```

This will:
- Load a small language model (tiny-gpt2)
- Load a dataset (TinyStories)
- Save activations from a specific layer
- Train a SAE on those activations
- Save the trained SAE model

**Outputs:**
- `outputs/sae_model.pt` - Trained SAE model
- `outputs/training_metadata.json` - Training configuration and metadata
- `outputs/store/` - Saved activations
- `outputs/cache/` - Cached dataset

### 2. Attach SAE and Collect Texts

Run the second notebook to attach the SAE and collect top texts:

```bash
jupyter notebook examples/02_attach_sae_and_save_texts.ipynb
```

This will:
- Load the trained SAE from the previous step
- Attach it to the language model
- Enable text tracking for neurons
- Run inference on new text data
- Collect top activating texts for each neuron

**Outputs:**
- `outputs/top_texts.json` - Top texts for each neuron
- `outputs/concept_dictionary/` - Concept dictionary
- `outputs/attachment_metadata.json` - Attachment metadata

### 3. Load and Manipulate Concepts

Run the third notebook to work with the collected concepts:

```bash
jupyter notebook examples/03_load_and_manipulate_concepts.ipynb
```

This will:
- Load the top texts and concept dictionary
- Demonstrate concept manipulation
- Export concepts to different formats
- Analyze concept patterns and relationships
- Create comprehensive analysis reports

**Outputs:**
- `outputs/concepts_export.json` - Concept dictionary in JSON format
- `outputs/concepts_export.csv` - Concept dictionary in CSV format
- `outputs/concept_analysis_report.json` - Comprehensive analysis report

## Configuration

Each notebook has configuration sections at the top where you can modify:

- **Model**: Change the language model (default: `sshleifer/tiny-gpt2`)
- **Dataset**: Change the dataset (default: `roneneldan/TinyStories`)
- **Layer**: Choose which layer to analyze (default: attention projection layer)
- **Data limits**: Adjust the amount of data to process
- **SAE parameters**: Modify SAE architecture and training parameters
- **Text tracking**: Configure how many top texts to collect per neuron

## Key Features Demonstrated

### SAE Training
- Loading language models and datasets
- Saving activations from specific layers
- Training SAEs with various configurations
- Saving trained models with metadata

### Text Collection
- Attaching SAEs to language models
- Enabling text tracking for neurons
- Collecting top activating texts
- Managing concept dictionaries

### Concept Manipulation
- Creating and modifying concept dictionaries
- Adding, removing, and sorting concepts
- Exporting to different formats (JSON, CSV)
- Loading concepts from various sources
- Analyzing concept patterns and relationships

## File Structure

```
examples/
├── 01_train_sae_model.ipynb
├── 02_attach_sae_and_save_texts.ipynb
├── 03_load_and_manipulate_concepts.ipynb
├── README.md
└── outputs/
    ├── sae_model.pt
    ├── training_metadata.json
    ├── attachment_metadata.json
    ├── top_texts.json
    ├── concepts_export.json
    ├── concepts_export.csv
    ├── concept_analysis_report.json
    ├── concept_dictionary/
    ├── store/
    └── cache/
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `DATA_LIMIT` or `BATCH_SIZE_*` parameters
2. **Model not found**: Ensure you have internet connection for downloading models
3. **Previous outputs missing**: Run notebooks in order (01 → 02 → 03)
4. **Slow performance**: Use smaller models or reduce data limits

### Performance Tips

- Use GPU if available (automatically detected)
- Start with small data limits for testing
- Use smaller models for quick experimentation
- Adjust batch sizes based on available memory

## Next Steps

After running these examples, you can:

- Experiment with different language models and layers
- Modify SAE architectures and training parameters
- Analyze different types of text data
- Build custom concept analysis tools
- Integrate SAEs into your own projects

## API Reference

The examples demonstrate the following key APIs:

- `LanguageModel.from_huggingface()` - Load language models
- `TextSnippetDataset.from_huggingface()` - Load text datasets
- `model.activations.infer_and_save()` - Save activations
- `SAETrainer` - Train SAE models
- `AutoencoderConcepts` - Manage neuron-text associations
- `ConceptDictionary` - Store and manipulate concepts
