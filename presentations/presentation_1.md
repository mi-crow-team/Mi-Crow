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

**Authors: Hubert Kowalski, Adam Kanisty**
**Thesis Supervisor: Mgr. Inż. Vladimir Zaigrajew**
**Additional Supervisor: Prof. dr hab. Inż. Przemysław Biecek**

---

## 1. Motivation: Why Mechanistic Interpretability?

**The Challenge of LLM Safety**
- Growing deployment of AI systems in high-stakes domains
- Lack of transparency in decision-making processes
- Need for controllable and interpretable models

**Specific Problem: LLM Moderation**
- Preventing harmful content generation
- Detecting high-stakes interactions
- Ensuring safe responses in production environments<sup>[1]</sup>

---

## 2. Motivation: The Gap in XAI Tools

**Current State**
- Growing use of explainable artificial intelligence systems in industry and research
- Need to thoroughly understand and control their operation.


**Key Gaps**
- No comprehensive, open-source XAI framework
- Limited integration with existing workflows
- Lack of user-friendly interfaces for non-technical teams
- Minimal research on Polish language models

---

## 3. Collaboration with Bielik Team

**Bielik: Polish Open-Source LLM**
- Available on HuggingFace
- Comparable to international models for Polish language
- Bielik-Guard (Sójka): Existing moderation solution

**Our Goals**
- Compare mechanistic interpretability methods with Bielik-Guard
- Make moderation **faster, less costly, and less hardware-intensive**
- Access to evaluation datasets from Bielik team

---

## 4. Thesis Deliverables

1. **mi_crow Python Package** 
   - Facilitates LLM interactions, dataset handling, activations collection
   - Ready-to-use SAE implementation

2. **UI for LLM Control** *(In Progress)*
   - Concept-based steering using SAE
   - Interactive visualization

3. **Experiments & Validation**
   - Latent Prototype Moderator (LPM)
   - Linear probing for prompt classification
   - Comparison with existing methods

---

## 5. mi_crow Package: Architecture

**Modular Design Philosophy**

```
mi_crow/
├── language_model/     # LLM wrapper & inference
├── datasets/           # Data loading strategies
├── hooks/             # Activation inspection system
├── mechanistic/       # MI techniques (SAE, concepts)
└── store/             # Persistence layer
```

**Key Features**
- Multi-version Python support (3.10, 3.11, 3.12)
- GPU acceleration
- Extensible hook architecture

---

## 6. mi_crow: Key Modules (1/2)

**LanguageModel Module**
```python
from mi_crow.language_model import LanguageModel

# Load from HuggingFace
lm = LanguageModel.from_huggingface(
    "speakleash/Bielik-7B-Instruct-v0.1",
    store=store
)
```

**Datasets Module**
- `TextDataset`: General text processing
- `ClassificationDataset`: Labeled data for training
- Support for HuggingFace datasets and custom loaders

---

## 7. mi_crow: Key Modules (2/2)

**Hooks System**
- **Detector**: Capture activations without modification
- **Controller**: Modify activations during forward pass
- **SAE Integration**: Both detector and controller capabilities

**Store Module**
- Hierarchical organization: `run_id → batch → layer → key`
- Local filesystem implementation
- Supports tensors and metadata

---

## 8. Mechanistic Interpretability Methods

**Sparse Autoencoders (SAE)**<sup>[2]</sup>
$$
\mathcal{L}(\theta) = \underbrace{\|x - \hat{x}\|^2}_{\text{Reconstruction}} + \lambda \underbrace{\|z\|_1}_{\text{Sparsity}}
$$

**Latent Prototype Moderator (LPM)**
- Classification using learned prototypes in latent space
- Efficient for harmful content detection

**Linear Probing**
- Train classifier on frozen model activations
- Fast baseline for interpretability

<!-- Optional: Include classification results when ready -->

---

## 9. SAE Training Example

**TopK Sparse Autoencoder Implementation**

```python
from mi_crow.mechanistic.sae import TopKSae, SaeTrainingConfig

# Initialize SAE
sae = TopKSae(
    n_latents=4096,
    n_inputs=768,
    k=32,  # Top-K sparsity
    device="mps"
)

config = SaeTrainingConfig(
    epochs=100,
    batch_size=256,
    lr=1e-3,
    l1_lambda=1e-4,
    use_wandb=True
)

history = sae.trainer.train(store, run_id, layer_signature, config)
```

---

## 10. Concept Manipulation with SAE

**Identifying Interpretable Features**

```python
# Enable text tracking
sae.concepts.enable_text_tracking()

# Process texts and track activations
lm.layers.register_hook(layer_signature, sae)
lm.forwards(texts=dataset_texts)

# Get top activating texts for neuron
top_texts = sae.concepts.get_top_texts_for_neuron(
    neuron_idx=42,
    top_m=10
)

# Manipulate concept (amplify or suppress)
sae.concepts.manipulate_concept(
    neuron_idx=42,
    multiplier=2.0  # Amplify by 2x
)
```

---

## 11. mi_crow in Action: Results

**Training Metrics**
- if ready

<!-- Add visualization when available -->
<!-- ![w:800 center](img/training_metrics.png) -->
<!-- <div class="caption">Figure 1: SAE training convergence over 100 epochs.</div> -->

**Classification Performance** *(Optional - if ready)*
<!-- Results from LPM experiments on WildGuardMix dataset -->

---

## 12. CI/CD & Quality Assurance

**GitHub Actions Workflows**
- Tests run on every PR to `main`/`stage`
- Multi-version testing: Python 3.10, 3.11, 3.12
- **Minimum coverage requirement: 85%**

**Pre-commit Hooks**
- Ruff linting and formatting (PEP8 standard)
- Prevents non-compliant code from being committed

**Branch Protection**
- `main` and `stage` branches: **no direct pushes**
- All CI checks must pass before merge

---

## 13. Testing Strategy

**Test Organization**
```bash
pytest --unit    # Fast unit tests
pytest --e2e     # End-to-end workflows
pytest           # All tests with coverage
```

**Coverage Enforcement**
- Unit tests: `tests/unit/` - isolated component testing
- Integration tests: `tests/integration/` - cross-module interactions
- E2E tests: `tests/e2e/` - complete workflows
- Minimum 85% coverage enforced in CI

**Example E2E Test**: Train SAE → Attach to LM → Track concepts → Export

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
- Core mi_crow package with 85%+ test coverage
- SAE training and concept tracking
- CI/CD pipeline with multi-version testing

**In Progress**
- LPM and probing experiments on WildGuardMix and other datasets
- Comprehensive benchmarking vs. Bielik-Guard

**Future Work** 
- Extended experiments on diverse datasets
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
