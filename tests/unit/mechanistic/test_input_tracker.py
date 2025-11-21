import pytest

from amber.mechanistic.sae.concepts.input_tracker import InputTracker
from tests.unit.fixtures.language_models import create_language_model_from_mock
from tests.unit.fixtures.stores import create_temp_store


def _build_tracker(tmp_path):
    store = create_temp_store(tmp_path)
    lm = create_language_model_from_mock(store)
    return InputTracker(lm)


def test_input_tracker_enable_disable(tmp_path):
    tracker = _build_tracker(tmp_path)

    assert tracker.enabled is False
    tracker.enable()
    tracker.set_current_texts(["a"])
    assert tracker.enabled is True
    assert tracker.get_current_texts() == ["a"]

    tracker.disable()
    tracker.set_current_texts(["b"])
    assert tracker.get_current_texts() == ["a"]


def test_input_tracker_reset(tmp_path):
    tracker = _build_tracker(tmp_path)
    tracker.enable()
    tracker.set_current_texts(["first", "second"])

    tracker.reset()

    assert tracker.get_current_texts() == []


def test_input_tracker_copies_output(tmp_path):
    tracker = _build_tracker(tmp_path)
    tracker.enable()
    tracker.set_current_texts(["hello"])

    result = tracker.get_current_texts()
    result.append("mutated")

    assert tracker.get_current_texts() == ["hello"]

