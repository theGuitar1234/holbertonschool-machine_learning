#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def HP(Di, beta):
    """A function that does the trick"""
    P = np.exp(-Di * beta)
    sumP = np.sum(P)
    Pi = P / sumP
    Hi = -np.sum(Pi * np.log2(Pi))
    return (Hi, Pi)
