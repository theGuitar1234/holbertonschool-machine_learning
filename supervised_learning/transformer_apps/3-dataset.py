#!/usr/bin/env python3
"""Dataset module"""

import transformers
from setup import load_pt2en
import tensorflow as tf


class Dataset:
    """Loads and tokenizes the Portuguese-English translation dataset."""

    def __init__(self, batch_size, max_len):
        """Class constructor."""
        self.data_train = load_pt2en(split="train")
        self.data_valid = load_pt2en(split="validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        self.data_train = (
            self.data_train
            .map(self.tf_encode)
            .filter(lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len, tf.size(en) <= max_len))
            .cache()
            .shuffle(20000)
            .padded_batch(batch_size, padded_shapes=([None], [None]))
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        self.data_valid = (
            self.data_valid
            .map(self.tf_encode)
            .filter(lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len, tf.size(en) <= max_len))
            .padded_batch(batch_size, padded_shapes=([None], [None]))
        )

    def tokenize_dataset(self, data):
        """
        Creates subword tokenizers for Portuguese and English.
        """
        pt_sentences = []
        en_sentences = []
        # Iterate over the dataset
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))

        # Create tokenizers for Portuguese and English
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True,
            clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True,
            clean_up_tokenization_spaces=True)

        # Train the tokenizers
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2 ** 13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2 ** 13)

        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        return self.tokenizer_pt, self.tokenizer_en

    def encode(self, pt, en):
        """Encodes `pt` and `en` to `en`."""
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        pt_tokens = self.tokenizer_pt.encode(pt_text,
                                             add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_text,
                                             add_special_tokens=False)

        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        This method acts as a tensorFlow wrapper for the encode
        instance method.
        """
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
