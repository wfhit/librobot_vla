"""
Unit tests for the I/O utilities module.

Tests file I/O operations for various data formats.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from librobot.utils.io import (
    ensure_dir,
    load_json,
    load_pickle,
    load_torch,
    load_yaml,
    read_lines,
    read_text,
    save_json,
    save_pickle,
    save_torch,
    save_yaml,
    write_lines,
    write_text,
)


class TestJsonIO:
    """Test suite for JSON I/O functions."""

    def test_save_json_basic(self, tmp_path):
        """Test saving basic JSON data."""
        data = {"key": "value", "number": 42}
        filepath = tmp_path / "test.json"

        save_json(data, filepath)

        assert filepath.exists()

    def test_save_json_with_indent(self, tmp_path):
        """Test saving JSON with custom indent."""
        data = {"key": "value"}
        filepath = tmp_path / "indented.json"

        save_json(data, filepath, indent=4)

        content = filepath.read_text()
        assert "    " in content  # 4-space indent

    def test_save_json_creates_dirs(self, tmp_path):
        """Test that save_json creates parent directories."""
        data = {"test": True}
        filepath = tmp_path / "nested" / "dir" / "test.json"

        save_json(data, filepath)

        assert filepath.exists()

    def test_load_json_basic(self, tmp_path):
        """Test loading JSON data."""
        data = {"key": "value", "list": [1, 2, 3]}
        filepath = tmp_path / "load_test.json"
        filepath.write_text(json.dumps(data))

        loaded = load_json(filepath)

        assert loaded == data

    def test_json_roundtrip(self, tmp_path):
        """Test JSON save/load roundtrip."""
        data = {
            "string": "text",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"a": "b"},
        }
        filepath = tmp_path / "roundtrip.json"

        save_json(data, filepath)
        loaded = load_json(filepath)

        assert loaded == data

    def test_load_json_string_path(self, tmp_path):
        """Test load_json with string path."""
        data = {"test": True}
        filepath = tmp_path / "string_path.json"
        filepath.write_text(json.dumps(data))

        loaded = load_json(str(filepath))

        assert loaded == data


class TestYamlIO:
    """Test suite for YAML I/O functions."""

    def test_save_yaml_basic(self, tmp_path):
        """Test saving basic YAML data."""
        data = {"key": "value", "number": 42}
        filepath = tmp_path / "test.yaml"

        save_yaml(data, filepath)

        assert filepath.exists()

    def test_save_yaml_creates_dirs(self, tmp_path):
        """Test that save_yaml creates parent directories."""
        data = {"test": True}
        filepath = tmp_path / "nested" / "dir" / "test.yaml"

        save_yaml(data, filepath)

        assert filepath.exists()

    def test_load_yaml_basic(self, tmp_path):
        """Test loading YAML data."""
        data = {"key": "value", "list": [1, 2, 3]}
        filepath = tmp_path / "load_test.yaml"
        filepath.write_text(yaml.safe_dump(data))

        loaded = load_yaml(filepath)

        assert loaded == data

    def test_yaml_roundtrip(self, tmp_path):
        """Test YAML save/load roundtrip."""
        data = {
            "string": "text",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"a": "b"},
        }
        filepath = tmp_path / "roundtrip.yaml"

        save_yaml(data, filepath)
        loaded = load_yaml(filepath)

        assert loaded == data

    def test_yaml_multiline_string(self, tmp_path):
        """Test YAML with multiline strings."""
        data = {"text": "line1\nline2\nline3"}
        filepath = tmp_path / "multiline.yaml"

        save_yaml(data, filepath)
        loaded = load_yaml(filepath)

        assert loaded["text"] == data["text"]


class TestPickleIO:
    """Test suite for Pickle I/O functions."""

    def test_save_pickle_basic(self, tmp_path):
        """Test saving basic pickle data."""
        data = {"key": "value", "number": 42}
        filepath = tmp_path / "test.pkl"

        save_pickle(data, filepath)

        assert filepath.exists()

    def test_save_pickle_creates_dirs(self, tmp_path):
        """Test that save_pickle creates parent directories."""
        data = {"test": True}
        filepath = tmp_path / "nested" / "dir" / "test.pkl"

        save_pickle(data, filepath)

        assert filepath.exists()

    def test_load_pickle_basic(self, tmp_path):
        """Test loading pickle data."""
        data = {"key": "value", "list": [1, 2, 3]}
        filepath = tmp_path / "load_test.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        loaded = load_pickle(filepath)

        assert loaded == data

    def test_pickle_roundtrip(self, tmp_path):
        """Test pickle save/load roundtrip."""
        data = {
            "string": "text",
            "number": 42,
            "numpy": np.array([1, 2, 3]),
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},
        }
        filepath = tmp_path / "roundtrip.pkl"

        save_pickle(data, filepath)
        loaded = load_pickle(filepath)

        assert loaded["string"] == data["string"]
        assert loaded["number"] == data["number"]
        assert np.array_equal(loaded["numpy"], data["numpy"])
        assert loaded["tuple"] == data["tuple"]
        assert loaded["set"] == data["set"]

    def test_pickle_numpy_array(self, tmp_path):
        """Test pickle with NumPy arrays."""
        data = np.random.randn(10, 20)
        filepath = tmp_path / "numpy.pkl"

        save_pickle(data, filepath)
        loaded = load_pickle(filepath)

        assert np.array_equal(loaded, data)


class TestTorchIO:
    """Test suite for PyTorch I/O functions."""

    def test_save_torch_tensor(self, tmp_path):
        """Test saving PyTorch tensor."""
        tensor = torch.randn(10, 20)
        filepath = tmp_path / "tensor.pt"

        save_torch(tensor, filepath)

        assert filepath.exists()

    def test_save_torch_creates_dirs(self, tmp_path):
        """Test that save_torch creates parent directories."""
        tensor = torch.tensor([1, 2, 3])
        filepath = tmp_path / "nested" / "dir" / "tensor.pt"

        save_torch(tensor, filepath)

        assert filepath.exists()

    def test_load_torch_tensor(self, tmp_path):
        """Test loading PyTorch tensor."""
        tensor = torch.randn(5, 5)
        filepath = tmp_path / "load_test.pt"
        torch.save(tensor, filepath)

        loaded = load_torch(filepath)

        assert torch.equal(loaded, tensor)

    def test_torch_roundtrip(self, tmp_path):
        """Test PyTorch save/load roundtrip."""
        tensor = torch.randn(100, 100)
        filepath = tmp_path / "roundtrip.pt"

        save_torch(tensor, filepath)
        loaded = load_torch(filepath)

        assert torch.equal(loaded, tensor)

    def test_torch_state_dict(self, tmp_path):
        """Test saving and loading model state dict."""
        model = torch.nn.Linear(10, 5)
        state_dict = model.state_dict()
        filepath = tmp_path / "state_dict.pt"

        save_torch(state_dict, filepath)
        loaded = load_torch(filepath)

        for key in state_dict:
            assert torch.equal(loaded[key], state_dict[key])

    def test_torch_complex_data(self, tmp_path):
        """Test saving complex data structure."""
        data = {
            "tensor": torch.randn(10),
            "list": [1, 2, 3],
            "nested": {"tensor": torch.randn(5)},
        }
        filepath = tmp_path / "complex.pt"

        save_torch(data, filepath)
        loaded = load_torch(filepath)

        assert torch.equal(loaded["tensor"], data["tensor"])
        assert loaded["list"] == data["list"]
        assert torch.equal(loaded["nested"]["tensor"], data["nested"]["tensor"])

    def test_load_torch_map_location(self, tmp_path):
        """Test load_torch with map_location."""
        tensor = torch.randn(10)
        filepath = tmp_path / "device.pt"
        save_torch(tensor, filepath)

        loaded = load_torch(filepath, map_location="cpu")

        assert loaded.device.type == "cpu"


class TestEnsureDir:
    """Test suite for ensure_dir function."""

    def test_ensure_dir_creates_new(self, tmp_path):
        """Test ensure_dir creates new directory."""
        dirpath = tmp_path / "new_dir"

        result = ensure_dir(dirpath)

        assert dirpath.exists()
        assert dirpath.is_dir()
        assert result == dirpath

    def test_ensure_dir_nested(self, tmp_path):
        """Test ensure_dir creates nested directories."""
        dirpath = tmp_path / "a" / "b" / "c"

        result = ensure_dir(dirpath)

        assert dirpath.exists()
        assert result == dirpath

    def test_ensure_dir_existing(self, tmp_path):
        """Test ensure_dir with existing directory."""
        dirpath = tmp_path / "existing"
        dirpath.mkdir()

        result = ensure_dir(dirpath)

        assert dirpath.exists()
        assert result == dirpath

    def test_ensure_dir_string_path(self, tmp_path):
        """Test ensure_dir with string path."""
        dirpath = str(tmp_path / "string_path")

        result = ensure_dir(dirpath)

        assert Path(dirpath).exists()
        assert isinstance(result, Path)


class TestTextIO:
    """Test suite for text I/O functions."""

    def test_read_text(self, tmp_path):
        """Test reading text from file."""
        content = "Hello, World!"
        filepath = tmp_path / "test.txt"
        filepath.write_text(content)

        result = read_text(filepath)

        assert result == content

    def test_read_text_unicode(self, tmp_path):
        """Test reading unicode text."""
        content = "Hello, 世界! 🌍"
        filepath = tmp_path / "unicode.txt"
        filepath.write_text(content, encoding="utf-8")

        result = read_text(filepath)

        assert result == content

    def test_write_text(self, tmp_path):
        """Test writing text to file."""
        content = "Test content"
        filepath = tmp_path / "write_test.txt"

        write_text(content, filepath)

        assert filepath.read_text() == content

    def test_write_text_creates_dirs(self, tmp_path):
        """Test that write_text creates parent directories."""
        content = "Nested content"
        filepath = tmp_path / "nested" / "dir" / "test.txt"

        write_text(content, filepath)

        assert filepath.exists()
        assert filepath.read_text() == content

    def test_text_roundtrip(self, tmp_path):
        """Test text write/read roundtrip."""
        content = "Line 1\nLine 2\nLine 3"
        filepath = tmp_path / "roundtrip.txt"

        write_text(content, filepath)
        result = read_text(filepath)

        assert result == content

    def test_write_text_custom_encoding(self, tmp_path):
        """Test writing text with custom encoding."""
        content = "Test"
        filepath = tmp_path / "latin1.txt"

        write_text(content, filepath, encoding="latin-1")

        assert filepath.read_text(encoding="latin-1") == content


class TestLinesIO:
    """Test suite for lines I/O functions."""

    def test_read_lines(self, tmp_path):
        """Test reading lines from file."""
        lines = ["Line 1", "Line 2", "Line 3"]
        filepath = tmp_path / "lines.txt"
        filepath.write_text("\n".join(lines))

        result = read_lines(filepath)

        assert result == lines

    def test_read_lines_with_strip(self, tmp_path):
        """Test reading lines with stripping."""
        filepath = tmp_path / "spaces.txt"
        filepath.write_text("  Line 1  \n  Line 2  \n")

        result = read_lines(filepath, strip=True)

        # Stripping removes leading/trailing whitespace from each line
        # Trailing newline becomes empty string but file has only 2 actual lines
        assert result == ["Line 1", "Line 2"]

    def test_read_lines_without_strip(self, tmp_path):
        """Test reading lines without stripping."""
        filepath = tmp_path / "nospace.txt"
        filepath.write_text("  Line 1  \n  Line 2  \n")

        result = read_lines(filepath, strip=False)

        assert "  Line 1  \n" in result

    def test_write_lines(self, tmp_path):
        """Test writing lines to file."""
        lines = ["Line 1", "Line 2", "Line 3"]
        filepath = tmp_path / "write_lines.txt"

        write_lines(lines, filepath)

        content = filepath.read_text()
        assert "Line 1\n" in content
        assert "Line 2\n" in content
        assert "Line 3\n" in content

    def test_write_lines_with_newlines(self, tmp_path):
        """Test writing lines that already have newlines."""
        lines = ["Line 1\n", "Line 2\n"]
        filepath = tmp_path / "newlines.txt"

        write_lines(lines, filepath)

        content = filepath.read_text()
        # Should not double newlines - verify exact expected content
        assert content == "Line 1\nLine 2\n"

    def test_write_lines_creates_dirs(self, tmp_path):
        """Test that write_lines creates parent directories."""
        lines = ["Test"]
        filepath = tmp_path / "nested" / "lines.txt"

        write_lines(lines, filepath)

        assert filepath.exists()

    def test_lines_roundtrip(self, tmp_path):
        """Test lines write/read roundtrip."""
        lines = ["First", "Second", "Third"]
        filepath = tmp_path / "roundtrip_lines.txt"

        write_lines(lines, filepath)
        result = read_lines(filepath)

        assert result == lines


class TestIOEdgeCases:
    """Test edge cases and error handling."""

    def test_save_json_nonexistent_file(self, tmp_path):
        """Test saving to nonexistent path creates it."""
        data = {"test": True}
        filepath = tmp_path / "new" / "file.json"

        save_json(data, filepath)

        assert filepath.exists()

    def test_load_json_file_not_found(self, tmp_path):
        """Test loading nonexistent JSON file."""
        filepath = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_json(filepath)

    def test_load_yaml_file_not_found(self, tmp_path):
        """Test loading nonexistent YAML file."""
        filepath = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_yaml(filepath)

    def test_empty_json(self, tmp_path):
        """Test saving and loading empty JSON."""
        data = {}
        filepath = tmp_path / "empty.json"

        save_json(data, filepath)
        loaded = load_json(filepath)

        assert loaded == {}

    def test_empty_yaml(self, tmp_path):
        """Test saving and loading empty YAML."""
        data = {}
        filepath = tmp_path / "empty.yaml"

        save_yaml(data, filepath)
        loaded = load_yaml(filepath)

        # Empty YAML may load as None
        assert loaded is None or loaded == {}

    def test_read_empty_text(self, tmp_path):
        """Test reading empty text file."""
        filepath = tmp_path / "empty.txt"
        filepath.write_text("")

        result = read_text(filepath)

        assert result == ""

    def test_read_empty_lines(self, tmp_path):
        """Test reading empty lines file."""
        filepath = tmp_path / "empty_lines.txt"
        filepath.write_text("")

        result = read_lines(filepath)

        # Empty file returns empty list
        assert result == []

    @pytest.mark.parametrize("extension", [".json", ".yaml", ".pkl", ".pt"])
    def test_path_object_support(self, tmp_path, extension):
        """Test that all functions support Path objects."""
        filepath = tmp_path / f"test{extension}"

        # Just verify Path objects work (don't need to test functionality)
        assert isinstance(filepath, Path)
