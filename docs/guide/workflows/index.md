# Workflows

This section provides step-by-step guides for common tasks in mi-crow. Each workflow is designed to be self-contained and complete.

## Available Workflows

### [Saving Activations](saving-activations.md)
Learn how to collect and save activations from model layers for analysis and SAE training.

**When to use**:
- Preparing data for SAE training
- Analyzing activation patterns
- Debugging model behavior
- Collecting datasets for research

### [Training SAE Models](training-sae.md)
Complete guide to training sparse autoencoders on saved activations.

**When to use**:
- Training your first SAE
- Understanding hyperparameter selection
- Monitoring training progress
- Saving and loading trained models

### [Concept Discovery](concept-discovery.md)
Discover interpretable concepts by analyzing SAE neuron activations.

**When to use**:
- Finding what each SAE neuron represents
- Collecting examples for manual curation
- Validating concept quality
- Understanding model features

### [Concept Manipulation](concept-manipulation.md)
Control model behavior by manipulating discovered concepts.

**When to use**:
- Steering model outputs
- Running intervention experiments
- A/B testing concept effects
- Real-time model control

### [Activation Control](activation-control.md)
Directly manipulate activations using hooks for fine-grained control.

**When to use**:
- Custom intervention experiments
- Fine-grained activation modification
- Multi-layer interventions
- Advanced control patterns

## Workflow Dependencies

Most workflows build on each other:

```
Saving Activations
    ↓
Training SAE Models
    ↓
Concept Discovery
    ↓
Concept Manipulation
```

**Activation Control** can be used independently or in combination with SAE-based workflows.

## Quick Reference

| Workflow | Input | Output | Time |
|----------|-------|--------|------|
| Saving Activations | Model + Dataset | Saved activations | Minutes |
| Training SAE | Saved activations | Trained SAE | Hours |
| Concept Discovery | Trained SAE + Dataset | Top texts per neuron | Minutes |
| Concept Manipulation | Trained SAE + Concepts | Modified outputs | Seconds |
| Activation Control | Model + Hooks | Modified outputs | Seconds |

## Getting Started

If you're new to mi-crow, we recommend following workflows in order:

1. Start with **[Saving Activations](saving-activations.md)** to understand data collection
2. Move to **[Training SAE Models](training-sae.md)** to learn feature discovery
3. Try **[Concept Discovery](concept-discovery.md)** to find interpretable features
4. Use **[Concept Manipulation](concept-manipulation.md)** to control model behavior

For advanced users, **[Activation Control](activation-control.md)** provides direct hook-based control.

## Integration with Examples

Each workflow references relevant example notebooks:

- Examples are in the `examples/` directory
- Workflows explain the concepts
- Examples provide runnable code
- See [Examples Guide](../examples.md) for the full list

## Next Steps

Choose a workflow that matches your goal:

- **New to mi-crow?** → Start with [Saving Activations](saving-activations.md)
- **Want to train SAEs?** → See [Training SAE Models](training-sae.md)
- **Ready to discover concepts?** → Try [Concept Discovery](concept-discovery.md)
- **Need model control?** → Use [Concept Manipulation](concept-manipulation.md) or [Activation Control](activation-control.md)

