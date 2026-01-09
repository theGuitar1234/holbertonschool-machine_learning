#!/usr/bin/env python3
"""Docstring for 4-array.py."""

import pandas as pd
import numpy as np


def array(df):
    """Docstring for array."""
    return df[["High", "Close"]].tail().values
