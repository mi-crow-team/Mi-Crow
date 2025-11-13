"""Additional tests for InputTracker edge cases."""
import pytest

from amber.mechanistic.sae.concepts.input_tracker import InputTracker


class _FakeLM:
    def __init__(self):
        self.model_id = "test_model"


def test_input_tracker_enable_disable():
    """Test enable and disable methods."""
    lm = _FakeLM()
    tracker = InputTracker(lm)
    
    assert not tracker.enabled
    
    tracker.enable()
    assert tracker.enabled
    
    tracker.disable()
    assert not tracker.enabled


def test_input_tracker_set_current_texts_when_disabled():
    """Test set_current_texts doesn't save when disabled."""
    lm = _FakeLM()
    tracker = InputTracker(lm)
    
    tracker.disable()
    tracker.set_current_texts(["text1", "text2"])
    
    # Should not save texts when disabled
    assert len(tracker.get_current_texts()) == 0


def test_input_tracker_set_current_texts_when_enabled():
    """Test set_current_texts saves when enabled."""
    lm = _FakeLM()
    tracker = InputTracker(lm)
    
    tracker.enable()
    tracker.set_current_texts(["text1", "text2"])
    
    # Should save texts when enabled
    texts = tracker.get_current_texts()
    assert len(texts) == 2
    assert texts == ["text1", "text2"]


def test_input_tracker_reset():
    """Test reset clears stored texts."""
    lm = _FakeLM()
    tracker = InputTracker(lm)
    
    tracker.enable()
    tracker.set_current_texts(["text1", "text2"])
    
    tracker.reset()
    
    # Should be empty after reset
    assert len(tracker.get_current_texts()) == 0


def test_input_tracker_get_current_texts_returns_copy():
    """Test get_current_texts returns a copy, not reference."""
    lm = _FakeLM()
    tracker = InputTracker(lm)
    
    tracker.enable()
    tracker.set_current_texts(["text1", "text2"])
    
    texts1 = tracker.get_current_texts()
    texts2 = tracker.get_current_texts()
    
    # Should be equal but not the same object
    assert texts1 == texts2
    assert texts1 is not texts2
    
    # Modifying one shouldn't affect the other
    texts1.append("text3")
    assert len(tracker.get_current_texts()) == 2  # Original unchanged


def test_input_tracker_with_empty_texts():
    """Test InputTracker handles empty text list."""
    lm = _FakeLM()
    tracker = InputTracker(lm)
    
    tracker.enable()
    tracker.set_current_texts([])
    
    texts = tracker.get_current_texts()
    assert texts == []


def test_input_tracker_with_single_text():
    """Test InputTracker handles single text."""
    lm = _FakeLM()
    tracker = InputTracker(lm)
    
    tracker.enable()
    tracker.set_current_texts(["single text"])
    
    texts = tracker.get_current_texts()
    assert texts == ["single text"]


def test_input_tracker_multiple_set_calls():
    """Test InputTracker overwrites on multiple set calls."""
    lm = _FakeLM()
    tracker = InputTracker(lm)
    
    tracker.enable()
    tracker.set_current_texts(["text1", "text2"])
    tracker.set_current_texts(["text3", "text4", "text5"])
    
    # Should have latest texts
    texts = tracker.get_current_texts()
    assert texts == ["text3", "text4", "text5"]


def test_input_tracker_with_sequence_types():
    """Test InputTracker handles different sequence types."""
    lm = _FakeLM()
    tracker = InputTracker(lm)
    
    tracker.enable()
    
    # Test with tuple
    tracker.set_current_texts(("text1", "text2"))
    assert tracker.get_current_texts() == ["text1", "text2"]
    
    # Test with list
    tracker.set_current_texts(["text3", "text4"])
    assert tracker.get_current_texts() == ["text3", "text4"]

