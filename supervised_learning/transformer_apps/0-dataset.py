#!/usr/bin/env python3
"""Load and prepare a Portuguese-to-English translation dataset."""

import transformers
from setup import load_pt2en


class Dataset:
    """Load the translation datasets and create subword tokenizers."""

    def __init__(self):
        """Initialize the training data, validation data, and tokenizers."""
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English tokenizers from a dataset.

        Args:
            data: A dataset containing Portuguese-English sentence pairs.

        Returns:
            A tuple containing the Portuguese tokenizer and English tokenizer.
        """
        pretrained_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        pretrained_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        portuguese_corpus = (
            pt.numpy().decode("utf-8") for pt, _ in data
        )
        tokenizer_pt = pretrained_pt.train_new_from_iterator(
            portuguese_corpus,
            vocab_size=2 ** 13
        )

        english_corpus = (
            en.numpy().decode("utf-8") for _, en in data
        )
        tokenizer_en = pretrained_en.train_new_from_iterator(
            english_corpus,
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
