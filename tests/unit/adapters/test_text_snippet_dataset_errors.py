import pytest
from pathlib import Path

from amber.adapters.text_snippet_dataset import TextSnippetDataset


def test_from_local_raises_on_missing_text_field_in_structured(tmp_path):
    # Write a CSV with column 'msg' not 'text'
    csv = tmp_path / "data.csv"
    csv.write_text("msg\nhello\nworld\n", encoding="utf-8")
    with pytest.raises(ValueError):
        TextSnippetDataset.from_local(csv, dataset_dir=tmp_path/"cache", text_field="text")
    # But succeeds if specifying correct text_field
    ds = TextSnippetDataset.from_local(csv, dataset_dir=tmp_path/"cache2", text_field="msg")
    assert len(ds) == 2
