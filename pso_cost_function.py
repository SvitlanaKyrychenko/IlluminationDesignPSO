import numpy as np
from skimage import color

from abc import ABC, abstractmethod
from utils import spim2XYZ, spim2rgb, XYZ2Lab, XYZ2RGB
from numpy import ndarray, dtype
from typing import Any, Tuple
from cost_functions import ciede, rgbde, michelson_contrast

# Abstract Base Class
class PSOCostFunction(ABC):

    @abstractmethod
    def calculate_cost(self, sample1: ndarray[Any, dtype[float]], sample2: ndarray[Any, dtype[float]],
                       wavelength: ndarray[Any, dtype[float]], simulated_illuminants: ndarray[Any, dtype[float]]) \
            -> ndarray[Any, dtype[float]]:
        """Compute the cost."""
        pass



# -----------------------
# Concrete Implementations
# -----------------------
class CiedePSO(PSOCostFunction):

    def calculate_cost(self, sample1: ndarray[Any, dtype[float]], sample2: ndarray[Any, dtype[float]],
                       wavelength: ndarray[Any, dtype[float]], simulated_illuminants: ndarray[Any, dtype[float]]) \
            -> ndarray[Any, dtype[float]]:

            sample1_xyz = np.array([spim2XYZ(sample1, wavelength, sim_ill) for sim_ill in simulated_illuminants])
            sample2_xyz = np.array([spim2XYZ(sample2, wavelength, sim_ill) for sim_ill in simulated_illuminants])
            sample1_lab = np.array([np.squeeze(XYZ2Lab(x))for x in sample1_xyz])
            sample2_lab = np.array([np.squeeze(XYZ2Lab(x))for x in sample2_xyz])
            costs = np.array([
                ciede(s1, s2) for s1, s2 in zip(sample1_lab, sample2_lab)
            ])
            return costs




class RgbdePSO(PSOCostFunction):

    def calculate_cost(self, sample1: ndarray[Any, dtype[float]], sample2: ndarray[Any, dtype[float]],
                       wavelength: ndarray[Any, dtype[float]], simulated_illuminants: ndarray[Any, dtype[float]]) \
            -> ndarray[Any, dtype[float]]:

            sample1_rgb = np.array([spim2rgb(sample1, wavelength, sim_ill) for sim_ill in simulated_illuminants])
            sample2_rgb = np.array([spim2rgb(sample2, wavelength, sim_ill) for sim_ill in simulated_illuminants])
            sample1_grayscale = color.rgb2gray(sample1_rgb)
            sample2_grayscale = color.rgb2gray(sample2_rgb)
            costs = np.array([
                michelson_contrast(s1, s2) for s1, s2 in zip(sample1_grayscale, sample2_grayscale)
            ])
            return costs


class MichelsonContrastPSO(PSOCostFunction):

    def calculate_cost(self, sample1: ndarray[Any, dtype[float]], sample2: ndarray[Any, dtype[float]],
                       wavelength: ndarray[Any, dtype[float]], simulated_illuminants: ndarray[Any, dtype[float]]) \
            -> ndarray[Any, dtype[float]]:

            sample1_xyz = np.array([spim2XYZ(sample1, wavelength, sim_ill) for sim_ill in simulated_illuminants])
            sample2_xyz = np.array([spim2XYZ(sample2, wavelength, sim_ill) for sim_ill in simulated_illuminants])
            sample1_rgb = np.array([np.squeeze(XYZ2RGB(x))for x in sample1_xyz])
            sample2_rgb = np.array([np.squeeze(XYZ2RGB(x))for x in sample2_xyz])
            costs = np.array([
                rgbde(s1, s2) for s1, s2 in zip(sample1_rgb, sample2_rgb)
            ])
            return costs