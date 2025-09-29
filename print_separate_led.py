import numpy as np

from data_preparation import get_main_data
from visualization import show_rgb_custom_illuminant, plot_spds


if __name__ == '__main__':

    sample_folder = "./Rosita_Aguirre_Plascencia/capture/"
    sample_name = "sample_scan_0043"
    leds_folder = "./LED_emission_spectra/LED_emission_spectra_expTime_85ms_avg_10_spectra/"

    # Prepera data
    spot_number = [290, 185]
    spot_background = [250, 180]
    reflectance, ref_wavelengths, leds_spectra = get_main_data(sample_folder, sample_name, leds_folder)
    indx = 3
    plot_spds([reflectance[290, 185, :], reflectance[250, 180, :], leds_spectra[3]/np.max(leds_spectra[3])], ref_wavelengths, ["Spot 1", "Spot 2", "L optim"])
    show_rgb_custom_illuminant(reflectance, ref_wavelengths, leds_spectra[3])
