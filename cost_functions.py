import numpy as np
from numpy import ndarray, dtype
from typing import Any

def euclidean_diff(sample1: ndarray[Any, dtype[float]], sample2: ndarray[Any, dtype[float]]) -> float:
    return float(np.linalg.norm(sample1 - sample2))


def michelson_contrast(sample_grayscale1: ndarray[Any, dtype[float]], sample_grayscale2: ndarray[Any, dtype[float]]) -> float:
    samples = np.array([sample_grayscale1, sample_grayscale2])
    return (np.max(samples) - np.min(samples)) / (np.max(samples) + np.min(samples))


def ciede(sample_lab1: ndarray[Any, dtype[float]], sample_lab2: ndarray[Any, dtype[float]]) -> float:
    return euclidean_diff(sample_lab1, sample_lab2)


def rgbde(sample_rgb1: ndarray[Any, dtype[float]], sample_rgb2: ndarray[Any, dtype[float]]) -> float:
    return euclidean_diff(sample_rgb1, sample_rgb2)
