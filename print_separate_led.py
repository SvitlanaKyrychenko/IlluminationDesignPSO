from data_preparation import get_main_data
from visualization import show_rgb_custom_illuminant


if __name__ == '__main__':

    sample_folder = "./Rosita_Aguirre_Plascencia/capture/"
    sample_name = "sample_scan_0043"
    leds_folder = "./LED_emission_spectra/LED_emission_spectra_expTime_85ms_avg_10_spectra/"

    # Prepera data
    reflectance, ref_wavelengths, leds_spectra = get_main_data(sample_folder, sample_name, leds_folder)
    show_rgb_custom_illuminant(reflectance, ref_wavelengths, leds_spectra[0])
