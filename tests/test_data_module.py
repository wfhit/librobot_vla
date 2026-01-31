"""Unit tests for data module."""


import numpy as np
import pytest

from librobot.data import (
    AbstractDataset,
    AbstractTokenizer,
    create_dataset,
    get_dataset,
    get_tokenizer,
    list_datasets,
    list_tokenizers,
    register_dataset,
    register_tokenizer,
)


class TestDatasetRegistry:
    """Test dataset registry functionality."""

    def test_list_datasets_empty(self):
        """Test listing datasets when none registered."""
        # Initially may be empty or have some pre-registered datasets
        datasets = list_datasets()
        assert isinstance(datasets, list)

    def test_register_dataset(self):
        """Test registering a custom dataset."""

        @register_dataset(name="test_dataset")
        class TestDataset(AbstractDataset):
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return {
                    "images": np.random.randn(3, 224, 224),
                    "text": "test instruction",
                    "actions": np.random.randn(7),
                    "proprioception": np.random.randn(14),
                }

            def get_statistics(self):
                return {
                    "action_mean": np.zeros(7),
                    "action_std": np.ones(7),
                }

        # Check registration
        assert "test_dataset" in list_datasets()

        # Check retrieval
        dataset_cls = get_dataset("test_dataset")
        assert dataset_cls == TestDataset

    def test_create_dataset(self):
        """Test creating dataset instance."""

        @register_dataset(name="test_dataset_2")
        class TestDataset2(AbstractDataset):
            def __len__(self):
                return 50

            def __getitem__(self, idx):
                return {
                    "images": np.random.randn(3, 224, 224),
                    "text": f"instruction {idx}",
                    "actions": np.random.randn(7),
                }

            def get_statistics(self):
                return {"action_mean": np.zeros(7), "action_std": np.ones(7)}

        dataset = create_dataset("test_dataset_2", data_path="/tmp/test")
        assert len(dataset) == 50

        sample = dataset[0]
        assert "images" in sample
        assert "text" in sample
        assert "actions" in sample


class TestTokenizerRegistry:
    """Test tokenizer registry functionality."""

    def test_list_tokenizers_empty(self):
        """Test listing tokenizers when none registered."""
        tokenizers = list_tokenizers()
        assert isinstance(tokenizers, list)

    def test_register_tokenizer(self):
        """Test registering a custom tokenizer."""

        @register_tokenizer(name="test_tokenizer")
        class TestTokenizer(AbstractTokenizer):
            def encode(self, text, return_tensors=None, **kwargs):
                tokens = [ord(c) % 1000 for c in str(text)]
                return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

            def decode(self, token_ids, skip_special_tokens=True, **kwargs):
                return "".join([chr(t % 128) for t in token_ids])

            def batch_encode(self, texts, return_tensors=None, **kwargs):
                results = [self.encode(t) for t in texts]
                return {
                    "input_ids": [r["input_ids"] for r in results],
                    "attention_mask": [r["attention_mask"] for r in results],
                }

            def batch_decode(self, token_ids_batch, skip_special_tokens=True, **kwargs):
                return [self.decode(t) for t in token_ids_batch]

            @property
            def pad_token_id(self):
                return 0

            @property
            def eos_token_id(self):
                return 1

            @property
            def bos_token_id(self):
                return 2

        # Check registration
        assert "test_tokenizer" in list_tokenizers()

        # Check retrieval
        tokenizer_cls = get_tokenizer("test_tokenizer")
        assert tokenizer_cls == TestTokenizer


class TestAbstractDataset:
    """Test AbstractDataset functionality."""

    def test_dataset_interface(self):
        """Test dataset interface compliance."""

        class ConcreteDataset(AbstractDataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "images": np.random.randn(3, 224, 224),
                    "text": "test",
                    "actions": np.random.randn(7),
                    "proprioception": np.random.randn(14),
                }

            def get_statistics(self):
                return {"action_mean": np.zeros(7), "action_std": np.ones(7)}

        dataset = ConcreteDataset(data_path="/tmp/test")

        # Test __len__
        assert len(dataset) == 10

        # Test __getitem__
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "images" in sample
        assert "actions" in sample

        # Test get_action_dim
        assert dataset.get_action_dim() == 7

        # Test get_state_dim
        assert dataset.get_state_dim() == 14

        # Test iteration
        count = 0
        for sample in dataset:
            count += 1
            if count > 2:
                break
        assert count == 3


class TestAbstractTokenizer:
    """Test AbstractTokenizer functionality."""

    def test_tokenizer_interface(self):
        """Test tokenizer interface compliance."""

        class ConcreteTokenizer(AbstractTokenizer):
            def encode(self, text, return_tensors=None, **kwargs):
                tokens = list(range(len(str(text))))
                return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

            def decode(self, token_ids, skip_special_tokens=True, **kwargs):
                return "decoded"

            def batch_encode(self, texts, return_tensors=None, **kwargs):
                results = [self.encode(t) for t in texts]
                return {
                    "input_ids": [r["input_ids"] for r in results],
                    "attention_mask": [r["attention_mask"] for r in results],
                }

            def batch_decode(self, token_ids_batch, skip_special_tokens=True, **kwargs):
                return [self.decode(t) for t in token_ids_batch]

            @property
            def pad_token_id(self):
                return 0

            @property
            def eos_token_id(self):
                return 1

            @property
            def bos_token_id(self):
                return 2

        tokenizer = ConcreteTokenizer()

        # Test encode
        result = tokenizer.encode("hello")
        assert "input_ids" in result
        assert "attention_mask" in result

        # Test decode
        decoded = tokenizer.decode([0, 1, 2])
        assert isinstance(decoded, str)

        # Test __call__ with single text
        result = tokenizer("hello")
        assert "input_ids" in result

        # Test __call__ with list of texts
        result = tokenizer(["hello", "world"])
        assert "input_ids" in result
        assert len(result["input_ids"]) == 2

        # Test properties
        assert tokenizer.pad_token_id == 0
        assert tokenizer.eos_token_id == 1
        assert tokenizer.bos_token_id == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
