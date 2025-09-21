import numpy as np
import matplotlib.pyplot as plt
from data_preparation import get_main_data
from utils import spim2rgb



if __name__ == '__main__':

    sample_folder = "./Rosita_Aguirre_Plascencia/capture/"
    sample_name = "sample_scan_0043"
    leds_folder = "./LED_emission_spectra/LED_emission_spectra_expTime_85ms_avg_10_spectra/"

    # Prepera data
    reflectance, ref_wavelengths, leds_spectra = get_main_data(sample_folder, sample_name, leds_folder)
    spot1 = [290, 185]
    spot2 = [250, 180]

    rgbD65Image = spim2rgb(reflectance, ref_wavelengths, 'D65', np.nan, np.nan)

    # Plot points
    plt.figure(figsize=(6, 6))
    plt.imshow(rgbD65Image)
    plt.axis('off')

    # overlay points
    plt.scatter([spot1[1]], [spot1[0]], s=40, c='red',  edgecolors='white', linewidths=1.5)
    plt.scatter([spot2[1]],  [spot2[0]],  s=40, c='blue', edgecolors='white', linewidths=1.5)

    plt.tight_layout()
    plt.show()