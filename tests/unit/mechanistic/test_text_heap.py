from __future__ import annotations

import pytest

from mi_crow.mechanistic.sae.concepts.text_heap import TextHeap


def test_text_heap_basic_operations():
    """Test basic heap operations."""
    heap = TextHeap(max_size=3)
    
    heap.update("text1", 0.5, 0)
    heap.update("text2", 0.8, 1)
    heap.update("text3", 0.3, 2)
    
    items = heap.get_items()
    assert len(items) == 3
    
    scores = [item[0] for item in items]
    assert 0.5 in scores
    assert 0.8 in scores
    assert 0.3 in scores


def test_text_heap_max_size():
    """Test that heap respects max_size."""
    heap = TextHeap(max_size=2)
    
    heap.update("text1", 0.1, 0)
    heap.update("text2", 0.2, 1)
    heap.update("text3", 0.9, 2)
    
    items = heap.get_items()
    assert len(items) == 2
    
    scores = [item[0] for item in items]
    assert 0.9 in scores
    assert 0.2 in scores
    assert 0.1 not in scores


def test_text_heap_duplicate_update():
    """Test updating existing text with higher score."""
    heap = TextHeap(max_size=3)
    
    heap.update("same_text", 0.3, 0)
    heap.update("same_text", 0.7, 1)
    
    items = heap.get_items()
    assert len(items) == 1
    
    score, text, token_idx = items[0]
    assert text == "same_text"
    assert score == 0.7
    assert token_idx == 1


def test_text_heap_duplicate_update_lower_score():
    """Test updating existing text with lower score (should not update)."""
    heap = TextHeap(max_size=3)
    
    heap.update("same_text", 0.7, 0)
    heap.update("same_text", 0.3, 1)
    
    items = heap.get_items()
    assert len(items) == 1
    
    score, text, token_idx = items[0]
    assert text == "same_text"
    assert score == 0.7
    assert token_idx == 0


def test_text_heap_adjusted_score():
    """Test heap ordering with adjusted_score."""
    heap = TextHeap(max_size=3)
    
    heap.update("text1", score=0.1, token_idx=0, adjusted_score=0.9)
    heap.update("text2", score=0.2, token_idx=1, adjusted_score=0.8)
    heap.update("text3", score=0.3, token_idx=2, adjusted_score=0.7)
    
    items = heap.get_items()
    assert len(items) == 3
    
    scores = [item[0] for item in items]
    assert 0.1 in scores
    assert 0.2 in scores
    assert 0.3 in scores
    
    adjusted_scores = [heap._heap[i][0] for i in range(len(heap._heap))]
    assert 0.9 in adjusted_scores
    assert 0.8 in adjusted_scores
    assert 0.7 in adjusted_scores


def test_text_heap_replace_minimum():
    """Test replacing minimum element when heap is full."""
    heap = TextHeap(max_size=2)
    
    heap.update("text1", 0.1, 0)
    heap.update("text2", 0.2, 1)
    heap.update("text3", 0.9, 2)
    
    items = heap.get_items()
    assert len(items) == 2
    
    scores = [item[0] for item in items]
    assert 0.9 in scores
    assert 0.2 in scores
    assert 0.1 not in scores


def test_text_heap_clear():
    """Test clearing the heap."""
    heap = TextHeap(max_size=3)
    
    heap.update("text1", 0.5, 0)
    heap.update("text2", 0.8, 1)
    
    assert len(heap) == 2
    
    heap.clear()
    
    assert len(heap) == 0
    assert heap.get_items() == []


def test_text_heap_multiple_duplicates():
    """Test handling multiple duplicate updates."""
    heap = TextHeap(max_size=3)
    
    heap.update("text1", 0.1, 0)
    heap.update("text2", 0.2, 1)
    heap.update("text1", 0.5, 2)
    heap.update("text2", 0.8, 3)
    heap.update("text3", 0.3, 4)
    
    items = heap.get_items()
    assert len(items) == 3
    
    texts = [item[1] for item in items]
    assert "text1" in texts
    assert "text2" in texts
    assert "text3" in texts
    
    text_to_score = {item[1]: item[0] for item in items}
    assert text_to_score["text1"] == 0.5
    assert text_to_score["text2"] == 0.8
    assert text_to_score["text3"] == 0.3


def test_text_heap_siftup_index_consistency():
    """Test that indices remain consistent after siftup operations."""
    heap = TextHeap(max_size=5)
    
    heap.update("text1", 0.1, 0)
    heap.update("text2", 0.2, 1)
    heap.update("text3", 0.3, 2)
    heap.update("text4", 0.4, 3)
    
    text1_idx_before = heap._text_to_index.get("text1")
    assert text1_idx_before is not None
    
    heap.update("text1", 0.9, 4)
    
    text1_idx_after = heap._text_to_index.get("text1")
    assert text1_idx_after is not None
    
    score, text, token_idx = heap._heap[text1_idx_after][1]
    assert text == "text1"
    assert score == 0.9
    assert token_idx == 4
