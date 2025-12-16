---
marp: true
theme: default
paginate: true
style: |
  section {
    background-color: #ffffff;
    background-image: linear-gradient(to bottom, #4A90E2 0%, #4A90E2 80px, #ffffff 80px);
    padding-top: 100px;
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 24px;

  }
  section::after {
    content: attr(data-marpit-pagination) ' / ' attr(data-marpit-pagination-total);
    position: absolute;
    bottom: 20px;
    right: 30px;
    color: #666;
    font-size: 18px;
  }
  h1 {
    color: #2C3E50;
    border-bottom: 3px solid #4A90E2;
    padding-bottom: 10px;
  }
  h2 {
    color: #4A90E2;
    margin-top: 20px;
  }
  h3 {
    color: #5A6C7D;
  }
  code {
    background-color: #f4f4f4;
    border-radius: 4px;
  }
  section.title {
    background-image: linear-gradient(135deg, #81b8f7ff 0%, #235093ff 100%);
    color: white;
    text-align: center;
    padding-top: 200px;
  }
  section.title h1 {
    color: white;
    border: none;
    font-size: 60px;
  }
  section.title h2 {
    color: #E8F4FD;
    margin-top: 20px;
  }
  section.title::after {
    color: #E8F4FD;
  }
  img {
    display: block;
    margin: 20px auto;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  }
  .caption {
    text-align: center;
    font-size: 14px;
    color: #666;
    font-style: italic;
    margin-top: -10px;
    margin-bottom: 20px;
  }
  .reference {
    font-size: 16px;
    line-height: 1.6;
    margin-bottom: 15px;
  }
  sup {
    color: #4A90E2;
    font-weight: bold;
  }
  ul {
    line-height: 1.8;
  }
  table {
    font-size: 14px;
    margin: 20px auto;
  }
---
<!-- _class: title -->
# *Concept analysis in multimodal models using sparse autoencoders*

**Authors: Hubert Kowalski, Adam Kaniasty**
**Thesis Supervisor: Mgr. Inż. Vladimir Zaigrajew**
**Additional Supervisor: Prof. dr hab. Inż. Przemysław Biecek**

---

## 1. Motivation: Why Mechanistic Interpretability?

**LLM Safety Challenge**
- Growing AI deployment in high-stakes domains
- Lack of transparency in decisions
- Need for controllable, interpretable models

**LLM Moderation Problem**
- Prevent harmful content generation
- Detect high-stakes interactions
- Ensure safe production responses<sup>[1]</sup>

---

## 2. Motivation: The Gap in XAI Tools

**Current State**
- Growing XAI use in industry & research
- Need to understand & control operation

**Key Gaps**
- No comprehensive open-source XAI framework
- Limited workflow integration
- Lack of user-friendly interfaces
- Minimal Polish language model research

---

## 3. Collaboration with Bielik Team

**Bielik: Polish Open-Source LLM**
- Available on HuggingFace
- Comparable to international models (Polish)
- Bielik-Guard (Sójka): Existing moderation solution

**Our Goals**
- Compare MI methods with Bielik-Guard
- Faster, less costly, less hardware-intensive moderation
- Access to evaluation datasets

---

## 4. Thesis Deliverables

1. **Amber Python Package**
   - Comprehensive MI research library
   - Non-invasive hook system
   - SAE with overcomplete integration
   - Hierarchical storage for large-scale experiments

2. **UI for LLM Control** *(In Progress)*
   - Concept-based steering (SAE)
   - Interactive visualization

3. **Experiments & Validation**
   - LPM for content moderation
   - Linear probing for classification
   - Comparison with Bielik-Guard
   - Concept discovery on Bielik-1.5B

---

## 5. System Design and Architecture

**Three-Tier Architecture**

- **Amber Python Library**: Core MI package
  - Modular design, independent testing
  - Models, hooks, storage, training separation

- **FastAPI Backend** *(Planned)*: REST API for inference & SAE ops

- **User Interface** *(Planned)*: Web UI for concept manipulation

**Design**: Modularity → independent testing, easier maintenance, extensibility

---

## 5.1. Amber Library Structure

**Five Core Modules**

```
amber/
├── language_model/     # LLM wrapper, inference, tokenization
├── datasets/           # TextDataset, ClassificationDataset, loading strategies
├── hooks/             # Detector, Controller implementations
├── mechanistic/       # SAE, concepts, training
└── store/             # Persistence (LocalStore)
```

**Key Features**
- Extensibility: Custom hook implementations
- Data flexibility: Multiple loading strategies
- Python 3.10, 3.11, 3.12 support
- GPU acceleration: CUDA, MPS, CPU

---

## 5.2. Planned Extensions

**FastAPI Backend** *(In Progress)*
- REST API for inference

**User Interface** *(In Progress)*
- Interactive concept visualization
- Model steering interface

---

## 6. Language Model Integration Design

**Fence Pattern: Non-Invasive Wrapper**

- Zero model changes: Weights & architecture untouched
- Unified interface: Any HuggingFace model
- Hook attachment: No retraining needed
- Flexibility: Hub, local files, checkpoints

**Research Benefits**
- Analyze any pre-trained model immediately
- Compare models with same interface
- Reproducible across architectures

**Components**
- `LanguageModelLayers`: Unified layer access
- `LanguageModelTokenizer`: Tokenizer handling
- `LanguageModelActivations`: Collection & storage
- `InferenceEngine`: Forward pass with controllers
- `LanguageModelContext`: Shared statea

---

## 6.1. Unified Layer Access and Hook Integration

**Layer Access**
- Automatic name flattening
- Name-based & index-based selection
- Architecture-agnostic hook registration

**Hook Integration**
- Multiple hooks per layer
- Dynamic enable/disable
- Automatic cleanup

**Research Capabilities**
- Forward pass with controllers
- Text generation with concept manipulation
- Large-scale activation saving

---

## 6.2. Dataset Abstractions: Flexible Data Loading

**Dataset Types**
- `BaseDataset`: Abstract foundation + Store integration
- `TextDataset`: Text processing
- `ClassificationDataset`: Labeled data (single/multi-category)

**Loading Strategies**

**MEMORY**: Full dataset in RAM
- Fastest random access
- Small datasets

**DYNAMIC_LOAD**: Memory-mapped Arrow files
- Lower memory usage
- Large datasets, random access

**ITERABLE_ONLY**: Streaming mode
- Lowest memory
- Very large datasets, sequential only

**Benefits**: Flexibility, Store integration, HuggingFace compatible, memory efficient

---

## 7. Hook Abstractions: Design and Purpose

**What Are Hooks?**

Interception mechanisms for non-invasive observation & modification of activations during inference

**Design Rationale**
- Research on any pre-trained model (no retraining)
- Zero model modifications
- PyTorch-compatible with enhanced features
- Activation analysis, concept discovery, interventions

**Key Innovation**
- Enable/disable functionality
- Unique IDs for tracking
- Automatic lifecycle management
- Unified layer access API

**Hook Types**
- `FORWARD`: Post-activation (observe outputs)
- `PRE_FORWARD`: Pre-activation (modify inputs)

---

## 7.1. Hook Lifecycle and Memory Management

**Lifecycle**
1. Registration: Attach to layer
2. Execution: Triggered during forward pass
3. Processing: Detection or modification
4. Storage: Save to Store (detectors)
5. Cleanup: Automatic removal

**Memory Management**
- Lightweight objects
- Disk-based storage (safetensors)
- Batch processing
- Automatic cleanup
- CPU offloading option
- Streaming support

---

## 7.2. Detector vs Controller Design

**Detector** (`Detector` class)
- Captures activations (no modification)
- `process_activations()` for custom logic
- Examples: `LayerActivationDetector`, SAE
- Auto-saves to Store
- Applications: Analysis, data collection, visualization

**Controller** (`Controller` class)
- Modifies activations during forward pass
- `modify_activations()` transforms tensors
- Examples: `FunctionController`, SAE
- Can modify inputs (pre_forward) or outputs (forward)
- Applications: Steering, concept manipulation, causal experiments

**SAE Dual Role**
- Implements both `Detector` + `Controller`
- Detects: Saves SAE activations & metadata
- Controls: Modifies outputs for concept manipulation

---

## 7.3. Hook Flow and Activation Storage

**Inference Flow with Hooks**

<!-- ![w:800 center](img/hook_flow_diagram.png) -->
<!-- <div class="caption">Figure: Hook execution flow during model inference</div> -->

**Flow**: Text Input → Tokenizer → Model Forward → Layer Computation → Hook? → Detector/Controller → Store/Modified Tensor → Output

**Key Design**
- Non-invasive: No model modifications
- Automatic: Seamless execution during forward pass
- Dual modes: Detectors (observe) + Controllers (modify)
- Hierarchical storage: `runs/run_id/batch/layer/`

---

## 8. Mechanistic Interpretability Methods

**Sparse Autoencoders (SAE)**<sup>[2]</sup>
$$
\mathcal{L}(\theta) = \underbrace{\|x - \hat{x}\|^2}_{\text{Reconstruction}} + \lambda \underbrace{\|z\|_1}_{\text{Sparsity}}
$$

**Implementation**
- Foundation: `overcomplete.Sae` engine
- Training: `overcomplete.sae.train` via `SaeTrainer`
- Our contributions: Concept tracking, manipulation, LanguageModel integration

**Other Methods**
- **LPM**: Classification via learned prototypes (latent space)
- **Linear Probing**: Classifier on frozen activations

---

## 9. SAE Training Implementation

**Integration with Overcomplete**
- `overcomplete.Sae` engine
- `overcomplete.sae.train` via `SaeTrainer`
- Battle-tested, optimized, actively maintained
- Our contribution: Concept tracking, manipulation, LanguageModel integration

**Training Pipeline** (`SaeTrainer`)
- Load activations from Store
- Device placement & memory optimization
- `StoreDataloader` for data loading
- Wandb experiment tracking
- Save outputs & metadata

**Key Parameters**
- `n_latents`: Overcomplete factor (4x-32x)
- `n_inputs`: Layer hidden dimension
- `k`: Top-K sparsity (active neurons per sample)
- `device`: Auto-detect (CPU, CUDA, MPS)

---

## 9.1. Training Process and Experiment Tracking

**Workflow**
1. Activation Collection: Save to Store
2. Training: `sae.trainer.train()` with auto data loading
3. Persistence: Save model + metadata

**Outputs**
- Model weights: Encoder/decoder (safetensors)
- Training history: Loss, reconstruction error, sparsity
- Metadata: Config, layer signature, model ID
- Wandb: Real-time monitoring

**Wandb Features**
- Real-time: Loss, reconstruction error, L1/L0 sparsity
- Training curves: Convergence analysis
- Slow metrics: Dead features, L0 per epoch
- Auto hyperparameter logging
- Modes: Online, offline, disabled

<!-- ![w:900 center](img/wandb_training_dashboard.png) -->
<!-- <div class="caption">Figure: Wandb dashboard showing real-time SAE training metrics</div> -->


---

## 10. Concept Discovery: Experimental Findings

**Bielik-1.5B Experiments**

*Layer 16 (post-attention) + TinyStories dataset*

**Discovered Concepts**
- **Neuron 42**: Story beginnings ("Once upon a time...", "Long ago...")
- **Neuron 128**: Character actions ("jumped onto...", "ran quickly...")
- **Neuron 256**: Emotional states ("very happy...", "felt sad...")
- **Neuron 512**: Question patterns ("What would you do...", "How did...")

**Setup**
- Model: `speakleash/Bielik-1.5B-v3.0-Instruct`
- Layer: `llamaforcausallm_model_layers_16_post_attention_layernorm`
- SAE: 6144 latents (4x overcomplete), TopK=8, 1000 samples
- Coverage: 6144/6144 neurons (100%)
- Text tracking: Auto-collection via hooks, Top-K per neuron

---

## 10.1. Concept Dictionary and Manipulation

**Concept Dictionary**
- Loading: CSV & JSON formats
- Curation: Manual labeling from top texts
- Discovery: Automated naming (planned)

**Manipulation Capabilities**
- Amplification/Suppression: Control via multipliers
- Per-neuron control: Fine-grained manipulation
- Real-time: Immediate effect during inference
- Applications: Test causal relationships

**Parameters**
- `multiplication`: Per-neuron multiplier (default: 1.0)
- `bias`: Per-neuron bias (default: 0.0)
- Applied via SAE forward pass (hook system)

<!-- ![w:700 center](img/concept_dictionary_example.png) -->
<!-- <div class="caption">Figure: Example concept dictionary showing neuron-to-concept mappings</div> -->

---

## 10.2. Activation Control for Intervention Experiments

**Custom Controller**
- `FunctionController`: Arbitrary transformations
- Logic: Amplify, suppress, filter, custom
- Layer targeting: Unified hook API

**Dynamic Control**
- Enable/disable: `hook.enable()`, `hook.disable()`
- Conditional: `with_controllers` for A/B testing
- Real-time steering: Modify during generation
- Intervention studies: Compare with/without

<!-- ![w:750 center](img/activation_control_diagram.png) -->
<!-- <div class="caption">Figure: Custom controller modifying activations during inference</div> -->

---

## 11. Experimental Results and Findings

**SAE Training Achievements**
- Convergence: Reconstruction error decreases over epochs
- Sparsity: Target TopK=8 achieved
- Efficiency: Overcomplete optimizations
- Scalability: Batch-based processing

**Metrics Tracked**
- Reconstruction loss: MSE (input vs reconstructed)
- L1 sparsity: Sum of absolute activations
- L0 sparsity: Active neurons per sample (target: TopK=8)
- Dead features: Never-activating neurons

<!-- ![w:900 center](img/training_metrics.png) -->
<!-- <div class="caption">Figure: SAE training convergence showing loss reduction and sparsity metrics</div> -->

---

## 11.1. Concept Discovery Results

**Interpretable Concepts**
- Coverage: 6144/6144 neurons (100%)
- Quality: High semantic coherence
- Examples: Story beginnings, character actions, emotional states, questions
- Layer 16 (post-attention): Interpretable feature separation

**Concept Manipulation**
- Amplification: Increased concept strength
- Suppression: Reduced concept influence
- Real-time steering: Observable behavior changes
- Validation: Metrics confirm impact

---

## 11.2. Classification Performance

**LPM Experiments on WildGuardMix** *(In Progress)*

**Comparison with Bielik-Guard (Sójka)**
- Metrics: Accuracy, precision, recall
- Efficiency: Latency, resource requirements
- Goal: Faster, less costly moderation

**Expected Advantages**
- Faster inference: Concept-based vs full model
- Lower hardware: Reduced computation
- Interpretability: Understandable decisions
- Scalability: Production-ready

---

## 12. Quality Assurance and Reproducibility

**Research Reproducibility**
- Automated testing: GitHub Actions on every PR
- Multi-version: Python 3.10, 3.11, 3.12
- Coverage: 85% minimum threshold
- Parallel execution: pytest-xdist

**CI/CD**
- Tests Workflow: Code quality & functionality
- Docs Workflow: Auto-deploy documentation
- Pre-commit: Ruff linting & formatting
- Branch protection: CI approval required

**Benefits**: Reproducibility, reliability, up-to-date docs

---

## 12.1. Documentation and Knowledge Sharing

**Documentation System**
- MkDocs: Material theme, search, navigation
- Auto-generated API: mkdocstrings integration
- Example notebooks: Step-by-step guides
- Structure: API reference, guides, workflows

**Features**
- Search functionality
- Syntax highlighting
- Responsive design
- Auto-deployment

**Access**: https://adamkaniasty.github.io/Inzynierka/

---

## 13. Testing Strategy for Research Reliability

**Three-Tier Architecture**

**Unit Tests** (`tests/unit/`)
- Isolated component testing
- Fast execution
- No external dependencies

**Integration Tests** (`tests/integration/`)
- Cross-module interactions
- External: HuggingFace, overcomplete
- Internal: LanguageModel + Datasets/Hooks

**E2E Tests** (`tests/e2e/`)
- Complete workflows
- Real-world scenarios

**Execution**: `pytest --unit`, `--integration`, `--e2e`, or all with coverage

---

## 13.1. Test Coverage and Research Validation

**Coverage Enforcement**
- 85% minimum (CI enforced)
- Formats: Terminal, XML, HTML
- Branch coverage: All code paths
- CI artifacts: Reports available

**E2E Validation**
1. Train SAE (Store activations)
2. Attach SAE to layer (hooks)
3. Track concepts (inference)
4. Export & verify

**Coverage**: LanguageModel, Hooks, SAE, Store, Datasets

---

## 14. Division of Work

| **Task** | **Hubert Kowalski** | **Adam Kaniasty** |
|----------|---------------------|-------------------|
| **Completed** |
| MI Research & Literature Review | ✓ | |
| Datasets Module | ✓ | |
| Ruff & Pre-commit Setup | ✓ | |
| Language Model Module | | ✓ |
| Hooks System | | ✓ |
| SAE Implementation | | ✓ |
| Store & Persistence | | ✓ |
| CI/CD Configuration | | ✓ |
| **In Progress / Planned** |
| LPM Experiments | ✓ | |
| Linear Probing Experiments | ✓ | |
| UI Development | | ✓ |
| Concept Visualization | | ✓ |

---

## 15. Current Status & Next Steps

**Completed ✓**
- Core Amber package (85%+ coverage)
- SAE training & concept tracking
- CI/CD pipeline (multi-version testing)

**In Progress**
- LPM & probing experiments (WildGuardMix)
- Benchmarking vs. Bielik-Guard

**Future Work**
- Extended experiments (diverse datasets)
- UI for interactive concept manipulation


---

## 16. References

<div class="reference">
[1] Templeton, A., et al. (2024). "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." <i>Anthropic</i>. https://transformer-circuits.pub/2024/scaling-monosemanticity/
</div>

<div class="reference">
[2] Ng, A. (2011). "Sparse Autoencoder." <i>CS294A Lecture Notes</i>, Stanford University.
</div>

<div class="reference">
[3] Sharkey, L., et al. (2022). "Taking features out of superposition with sparse autoencoders." <i>arXiv:2209.10652</i>.
</div>

<div class="reference">
[4] Olah, C., et al. (2020). "Zoom In: An Introduction to Circuits." <i>Distill</i>, 5(3). https://distill.pub/2020/circuits/zoom-in/
</div>

---

## 17. Q&A

**Thank you for your attention!**

Questions?

**Project Resources:**
- GitHub: https://github.com/AdamKaniasty/Inzynierka
- Documentation: https://adamkaniasty.github.io/Inzynierka/
- Contact: adam.kaniasty@gmail.com, hubert.kowalski.prv@gmail.com
