import os.path

import numpy as np
from pso import PSOConfig, PSO, OptimizeMode
from pso_cost_function import CiedePSO, RgbdePSO, MichelsonContrastPSO, PSOCostFunction
from cost_functions import ciede, rgbde, michelson_contrast
from data_preparation import get_main_data, get_spots_reflectance
from visualization import show_rgb_custom_illuminant, plot_spds
from generate_emission_spectrum_file import save_emission_spectrum


# Test on random data
def run_random_pso(pso: PSO, pso_cost: PSOCostFunction):
    rand = np.random.default_rng()
    n_wavelength = 200
    low_bound = float(0.0)
    high_bound = float(1.0)
    sample1 = rand.uniform(low_bound, high_bound, size=n_wavelength).astype(np.float64)
    sample2 = rand.uniform(low_bound, high_bound, size=n_wavelength).astype(np.float64)
    wavelength = np.arange(400, 400 + n_wavelength*2, 2, dtype=float)
    led = rand.uniform(low_bound, high_bound, size=(10, n_wavelength)).astype(np.float64)
    global_test_pos, global_test_cost = pso.run_pso(sample1, sample2, wavelength, led, pso_cost, OptimizeMode.MIN)
    print(global_test_pos)
    print(global_test_cost)


if __name__ == '__main__':
    
    sample_folder = "./Rosita_Aguirre_Plascencia/capture/"
    sample_name = "sample_scan_0043"
    leds_folder = "./LED_emission_spectra/LED_emission_spectra_expTime_85ms_avg_10_spectra/"

    # Prepera data
    reflectance, ref_wavelengths, leds_spectra = get_main_data(sample_folder, sample_name, leds_folder)
    spot_number = [290, 185]
    spot_background = [250, 180]

    #spot_number = [290, 185] # Orange
    #spot_background = [275, 355] # Red

    spots = [spot_number, spot_background]
    spots_reflectance = get_spots_reflectance(spots, reflectance)


    # Set PSO parameters
    n_particles = 100
    n_iterations = 500
    c0 = 0.95
    c1 = 5
    c2 = 5
    pso_config = PSOConfig(n_particles, n_iterations, c0, c1, c2)
    pso = PSO(pso_config)

    # Chose PSO cost function
    #pso_cost = CiedePSO()
    pso_cost = RgbdePSO()
    #pso_cost = MichelsonContrastPSO()

    # Run PSO
    global_pos, global_cost = pso.run_pso(spots_reflectance[0], spots_reflectance[1], ref_wavelengths,
                                          leds_spectra, pso_cost, OptimizeMode.MIN)
    print("Final PSO cost")
    print(global_pos)
    print(global_cost)

    # Visualize result
    custom_illuminant = global_pos @ leds_spectra
    # plot_spds([spots_reflectance[0], spots_reflectance[1], custom_illuminant/np.max(custom_illuminant)], ref_wavelengths, ["Spot 1", "Spot 2", "L optim"])
    # show_rgb_custom_illuminant(reflectance, ref_wavelengths, custom_illuminant)

    custom_illuminant_normalized = custom_illuminant / np.max(custom_illuminant)

    # Visualize results
    spds = [spots_reflectance[0], spots_reflectance[1], custom_illuminant_normalized]
    labels = ["Spot 1", "Spot 2", "L optim"]
    plot_spds(spds, ref_wavelengths, labels)

    #cost_function = ciede
    cost_function = rgbde
    # cost_function = michelson_contrast
    show_rgb_custom_illuminant(reflectance, ref_wavelengths, custom_illuminant_normalized, spots, cost_function, global_cost)

    results_folder = "emission_spectrums"
    filename = "test_01.txt"
    save_emission_spectrum(results_folder, filename, ref_wavelengths, custom_illuminant_normalized)