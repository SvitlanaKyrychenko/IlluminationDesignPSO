import numpy as np
from skimage import color

from abc import ABC, abstractmethod
from utils import spim2XYZ, spim2rgb, XYZ2RGB, XYZ2Lab
from numpy import ndarray, dtype
from typing import Any
from cost_functions import ciede, rgbde, michelson_contrast


class PSOCostFunction(ABC):

    @abstractmethod
    def calculate_cost(self, sample1: ndarray[Any, dtype[float]], sample2: ndarray[Any, dtype[float]],
                       wavelength: ndarray[Any, dtype[float]], simulated_illuminants: ndarray[Any, dtype[float]],
                       worst_value: float) \
            -> ndarray[Any, dtype[float]]:
        # Compute the cost
        pass


class CiedePSO(PSOCostFunction):

    def calculate_cost(self, sample1: ndarray[Any, dtype[float]], sample2: ndarray[Any, dtype[float]],
                       wavelength: ndarray[Any, dtype[float]], simulated_illuminants: ndarray[Any, dtype[float]],
                       worst_value: float) \
            -> ndarray[Any, dtype[float]]:

            num_samples = np.size(simulated_illuminants, axis=0)

            sample1_xyz = np.array([spim2XYZ(sample1, wavelength, sim_ill) for sim_ill in simulated_illuminants])
            sample2_xyz = np.array([spim2XYZ(sample2, wavelength, sim_ill) for sim_ill in simulated_illuminants])

            sample1_rgb = np.array([np.squeeze(XYZ2RGB(x, np.nan, np.nan)) for x in sample1_xyz])
            sample2_rgb = np.array([np.squeeze(XYZ2RGB(x, np.nan, np.nan)) for x in sample2_xyz])

            rgb_invalid_mask_sample1 = np.array([np.any(np.isnan(x)) for x in sample1_rgb])
            rgb_invalid_mask_sample2 = np.array([np.any(np.isnan(x)) for x in sample2_rgb])

            costs = np.zeros(num_samples)
            for i in range(num_samples):
                if rgb_invalid_mask_sample1[i] or rgb_invalid_mask_sample2[i]:
                    costs[i] = worst_value
                else:
                    sample1_lab = np.squeeze(XYZ2Lab(sample1_xyz[i], wavelength, cie_illuminant=simulated_illuminants[i]))
                    sample2_lab = np.squeeze(XYZ2Lab(sample2_xyz[i], wavelength, cie_illuminant=simulated_illuminants[i]))
                    costs[i] = ciede(sample1_lab, sample2_lab)

            return costs


class RgbdePSO(PSOCostFunction):

    def calculate_cost(self, sample1: ndarray[Any, dtype[float]], sample2: ndarray[Any, dtype[float]],
                       wavelength: ndarray[Any, dtype[float]], simulated_illuminants: ndarray[Any, dtype[float]],
                       worst_value: float) \
            -> ndarray[Any, dtype[float]]:

            num_samples = np.size(simulated_illuminants, axis=0)

            sample1_rgb = np.array([spim2rgb(sample1, wavelength, sim_ill, np.nan, np.nan) for sim_ill in simulated_illuminants])
            sample2_rgb = np.array([spim2rgb(sample2, wavelength, sim_ill, np.nan, np.nan) for sim_ill in simulated_illuminants])

            rgb_invalid_mask_sample1 = np.array([np.any(np.isnan(x)) for x in sample1_rgb])
            rgb_invalid_mask_sample2 = np.array([np.any(np.isnan(x)) for x in sample2_rgb])

            costs = np.zeros(num_samples)
            for i in range(num_samples):
                if rgb_invalid_mask_sample1[i] or rgb_invalid_mask_sample2[i]:
                    costs[i] = worst_value
                else:
                    costs[i] = rgbde(sample1_rgb[i], sample2_rgb[i])

            return costs


class MichelsonContrastPSO(PSOCostFunction):

    def calculate_cost(self, sample1: ndarray[Any, dtype[float]], sample2: ndarray[Any, dtype[float]],
                       wavelength: ndarray[Any, dtype[float]], simulated_illuminants: ndarray[Any, dtype[float]],
                       worst_value: float) \
            -> ndarray[Any, dtype[float]]:

            num_samples = np.size(simulated_illuminants, axis=0)

            sample1_rgb = np.array([spim2rgb(sample1, wavelength, sim_ill, np.nan, np.nan) for sim_ill in simulated_illuminants])
            sample2_rgb = np.array([spim2rgb(sample2, wavelength, sim_ill, np.nan, np.nan) for sim_ill in simulated_illuminants])

            sample1_grayscale = color.rgb2gray(sample1_rgb)
            sample2_grayscale = color.rgb2gray(sample2_rgb)

            rgb_invalid_mask_sample1 = np.array([np.any(np.isnan(x)) for x in sample1_rgb])
            rgb_invalid_mask_sample2 = np.array([np.any(np.isnan(x)) for x in sample2_rgb])
            costs = np.zeros(num_samples)
            for i in range(num_samples):
                if rgb_invalid_mask_sample1[i] or rgb_invalid_mask_sample2[i]:
                    costs[i] = worst_value
                else:
                    costs[i] = michelson_contrast(sample1_grayscale[i], sample2_grayscale[i])
            return costs