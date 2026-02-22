#!/usr/bin/env python3
"""0-create_confusion.py."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """DocString for create_confusion_matrix."""
    true = np.argmax(labels, axis=1)
    pred = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    np.add.at(confusion, (true, pred), 1)
    return confusion
