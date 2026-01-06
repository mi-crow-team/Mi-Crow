import pytest
import torch

from experiments.utils.activation_aggregation import aggregate_activations_batch, aggregate_activations_single


def test_aggregate_activations_single_last():
    # Regular case: last non-special token
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    attention_mask = torch.tensor([False, True, True])
    out = aggregate_activations_single(activations, attention_mask, agg="last")
    assert torch.allclose(out, torch.tensor([5.0, 6.0]))


def test_aggregate_activations_single_mean():
    # Regular case: mean over non-special tokens
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    attention_mask = torch.tensor([1, 0, 1])
    out = aggregate_activations_single(activations, attention_mask, agg="mean")
    expected = torch.tensor([[1.0, 2.0], [5.0, 6.0]]).mean(dim=0)
    assert torch.allclose(out, expected)


def test_aggregate_activations_single_all_special_last():
    # All tokens are special: fallback to last token
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    attention_mask = torch.tensor([0, 0, 0])
    out = aggregate_activations_single(activations, attention_mask, agg="last")
    assert torch.allclose(out, torch.tensor([5.0, 6.0]))


def test_aggregate_activations_single_all_special_mean():
    # All tokens are special: fallback to mean over all
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    attention_mask = torch.tensor([0, 0, 0])
    out = aggregate_activations_single(activations, attention_mask, agg="mean")
    expected = activations.mean(dim=0)
    assert torch.allclose(out, expected)


def test_aggregate_activations_batch():
    activations = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    attention_mask = torch.tensor([[1, 0, 1], [0, 0, 0]])
    # First batch: last non-special, second batch: fallback to last
    out_last = aggregate_activations_batch(activations, attention_mask, agg="last")
    assert torch.allclose(out_last[0], torch.tensor([5.0, 6.0]))
    assert torch.allclose(out_last[1], torch.tensor([11.0, 12.0]))
    # First batch: mean over non-special, second batch: fallback to mean over all
    out_mean = aggregate_activations_batch(activations, attention_mask, agg="mean")
    expected0 = torch.tensor([[1.0, 2.0], [5.0, 6.0]]).mean(dim=0)
    expected1 = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]).mean(dim=0)
    assert torch.allclose(out_mean[0], expected0)
    assert torch.allclose(out_mean[1], expected1)


def test_shape_errors():
    # Wrong shapes should raise
    with pytest.raises(ValueError):
        aggregate_activations_single(torch.randn(3, 2), torch.randn(3, 2), "last")
    with pytest.raises(ValueError):
        aggregate_activations_batch(torch.randn(2, 3, 4), torch.randn(2, 3, 4), "last")


def test_empty_sequence():
    # Empty sequence should raise
    with pytest.raises(IndexError):
        aggregate_activations_single(torch.empty((0, 2)), torch.empty((0,), dtype=torch.long), "last")
