#!/usr/bin/env python3
"""Docstring for 1-from_dictionary.py."""

import pandas as pd


dict = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
}
index = "ABCD"
df = pd.DataFrame(dict, index=list(index))
