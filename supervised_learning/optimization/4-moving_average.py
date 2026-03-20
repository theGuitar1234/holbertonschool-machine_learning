#!/usr/bin/env python3
'''
moving average
'''

import numpy as np


def moving_average(data, beta):
    '''
    data is the list of data to calculate the moving average of

    beta is the weight used for the moving average

    Returns: a list containing the moving averages of data
    '''

    moving_averages = []
    v = 0
    for i, x in enumerate(data):
        v = beta * v + (1 - beta) * x
        moving_average = v / (1 - beta ** (i + 1))
        moving_averages.append(moving_average)

    return moving_averages
