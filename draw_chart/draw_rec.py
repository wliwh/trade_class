"""
绘制递归图
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from index_get.get_index_value import other_index_getter
import numpy as np
import matplotlib.pyplot as plt

def recurrence_plot(data, threshold=0.1):
    """
    Generate a recurrence plot from a time series.

    :param data: Time series data
    :param threshold: Threshold to determine recurrence
    :return: Recurrence plot
    """
    # Calculate the distance matrix
    N = len(data)
    distance_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            distance_matrix[i, j] = np.abs(data[i] - data[j])

    # Create the recurrence plot
    recurrence_plot = np.where(distance_matrix <= threshold, 1, 0)

    return recurrence_plot