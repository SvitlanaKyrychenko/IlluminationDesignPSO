import numpy as np

def euclidean_diff(sample1: float, sample2: float) -> float:
    return float(np.linalg.norm(sample1 - sample2))


def michelson_contrast(sample_grayscale1: float, sample_grayscale2: float) -> float:
    samples = np.array([sample_grayscale1, sample_grayscale2])
    return (np.max(samples) - np.min(samples)) / (np.max(samples) + np.min(samples))


def ciede(sample_lab1: float, sample_lab2: float) -> float:
    return euclidean_diff(sample_lab1, sample_lab2)


def rgbde(sample_rgb1: float, sample_rgb2: float) -> float:
    return euclidean_diff(sample_rgb1, sample_rgb2)
