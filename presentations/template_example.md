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
  section.title h3 {
    color: #cad5f0ff
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
    font-size: 18px;
    color: #666;
    font-style: italic;
    margin-top: -10px;
    margin-bottom: 20px;
  }
  .reference {
    font-size: 18px;
    line-height: 1.6;
    margin-bottom: 15px;
  }
  sup {
    color: #4A90E2;
    font-weight: bold;
  }
---
<!-- _class: title -->
# Concept analysis in multimodal models using sparse autoencoders


### Authors: Hubert Kowalski, Adam Kanisty
### Supervisor: Mgr. Inż. Vladimir Zaigrajew, 
### Prof. dr hab. Inż. Przemysław Biecek

---

## 1. Introduction
- Overview of the project
- Main goals and motivation
- Application of mechanistic interpretability techniques<sup>[1]</sup>

---

## 2. Project Structure
- Source code in `src/`
- Experiments in `experiments/`
- Documentation in `docs/`
- Tests in `tests/`

---

## 3. Key Features
- Modular design
- Dataset handling
- Language model integration
- Mechanistic interpretability

---

## 4. Example Workflow
1. Train SAE model
2. Attach SAE and save texts
3. Load concepts
4. Save inputs and outputs

---

## 5. Results & Visualizations

![w:1000 center](img/IMG_3111.PNG)
<div class="caption">Figure 1: System architecture showing the interaction between language model components and SAE layers.</div>

---

## 6. Code Examples

### Training Pipeline
```python
from amber.language_model import LanguageModel
from amber.datasets import TextDataset

def train_sae_model(model: LanguageModel, data: TextDataset):
    """Train Sparse Autoencoder on language model activations."""
    activations = model.get_activations(data)
    sae = SparseAutoencoder(input_dim=activations.shape[-1])
    
    for epoch in range(num_epochs):
        loss = sae.train_step(activations)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    return sae
```

---

## 7. Mathematical Formulations

### Loss Function
The training objective combines reconstruction and sparsity<sup>[2]</sup>:

$$
\mathcal{L}(\theta) = \underbrace{\sum_{i=1}^N \|x_i - \hat{x}_i\|^2}_{\text{Reconstruction Loss}} + \lambda \underbrace{\sum_{j=1}^d |z_j|}_{\text{L1 Sparsity}}
$$

### Activation Formula
$$
h = \text{ReLU}(W_{\text{enc}} \cdot x + b_{\text{enc}})
$$

---

## 8. Image Sizing Examples

### Small Image (300px width)
![w:300](img/IMG_3111.PNG)
<div class="caption">Figure 2: Detailed view of activation patterns (300px width).</div>

### Medium Image (600px width)
![w:600 center](img/IMG_3111.PNG)
<div class="caption">Figure 3: Comparison of model outputs across different configurations (600px width).</div>

---

## 9. Future Work
- Planned improvements
- Open questions
- Extended model architectures

---

## 10. References

<div class="reference">
[1] Olah, C., et al. (2020). "Zoom In: An Introduction to Circuits." <i>Distill</i>, 5(3). https://distill.pub/2020/circuits/zoom-in/
</div>

<div class="reference">
[2] Ng, A. (2011). "Sparse Autoencoder." <i>CS294A Lecture Notes</i>, Stanford University.
</div>

<div class="reference">
[3] Vaswani, A., et al. (2017). "Attention Is All You Need." <i>Advances in Neural Information Processing Systems</i>, 30.
</div>

---

## 11. Q&A
- Thank you for your attention!
- Questions?
