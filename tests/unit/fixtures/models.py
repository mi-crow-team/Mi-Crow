"""Mock PyTorch models for testing."""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class SimpleLM(nn.Module):
    """Simple language model for testing."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Forward pass."""
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = torch.relu(layer(x))
        logits = self.lm_head(x)
        return logits


class SequentialModel(nn.Module):
    """Sequential model with named children for layer testing."""

    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class ModelWithHiddenState(nn.Module):
    """Model that returns a tuple with hidden state (like some HuggingFace models)."""

    def __init__(self, input_size: int = 10, hidden_size: int = 20):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (output, hidden_state)."""
        output = self.linear(x)
        hidden_state = torch.randn(x.shape[0], self.hidden_size)
        return output, hidden_state


class ModelWithLastHiddenState(nn.Module):
    """Model that returns object with last_hidden_state attribute (like HuggingFace models)."""

    def __init__(self, input_size: int = 10, hidden_size: int = 20):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor):
        """Forward pass returning object with last_hidden_state."""
        hidden = self.linear(x)
        
        class Output:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state
        
        return Output(hidden)


def create_mock_model(
    model_type: str = "simple",
    vocab_size: int = 1000,
    hidden_size: int = 128,
    num_layers: int = 2,
) -> nn.Module:
    """
    Create a mock model for testing.
    
    Args:
        model_type: Type of model ("simple", "sequential", "hidden_state", "last_hidden_state")
        vocab_size: Vocabulary size for language models
        hidden_size: Hidden layer size
        num_layers: Number of layers
        
    Returns:
        Mock PyTorch model
    """
    if model_type == "simple":
        return SimpleLM(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers)
    elif model_type == "sequential":
        return SequentialModel(input_size=vocab_size, hidden_size=hidden_size, output_size=vocab_size)
    elif model_type == "hidden_state":
        return ModelWithHiddenState(input_size=vocab_size, hidden_size=hidden_size)
    elif model_type == "last_hidden_state":
        return ModelWithLastHiddenState(input_size=vocab_size, hidden_size=hidden_size)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

