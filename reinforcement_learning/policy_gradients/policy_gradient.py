#!/usr/bin/env python3
"""Simple Policy function."""
import numpy as np


def policy(matrix, weight):
    """Computes the policy with a weight of a matrix."""
    z = matrix.dot(weight)
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """Computes the Monte-Carlo policy gradient.

    state: matrix representing the current observation of the environment
    weight: matrix of random weight
    Returns: the action and the gradient
    """
    state = state.reshape(1, -1)
    probs = policy(state, weight)[0]
    action = np.random.choice(len(probs), p=probs)

    s = probs.reshape(-1, 1)
    jacobian = np.diagflat(s) - s.dot(s.T)
    dsoftmax = jacobian[action]
    dlog = dsoftmax / probs[action]
    gradient = state.T.dot(dlog.reshape(1, -1))

    return action, gradient
