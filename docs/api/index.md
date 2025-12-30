# API Reference

mi_crow's public Python API is documented automatically from docstrings.

The top-level `mi_crow` package is intentionally minimal (it only exports things
like `ping`). The real functionality lives in subpackages, which are documented
in the sections below.

## Table of Contents

- [Language Model](language_model.md) - Core language model API for loading models, running inference, and managing activations
- [Mechanistic Interpretability (SAE)](sae.md) - Sparse Autoencoders, training, concepts, and related modules
- [Datasets](datasets.md) - Dataset loading and management utilities
- [Store](store.md) - Persistence layer for activations, models, and runs
- [Hooks](hooks.md) - Hook system for intercepting model activations

## Top-level Package

::: mi_crow

