import numpy as np
from pso import PSOConfig, PSO, OptimizeMode, CostFunctionMode
from data_preparation import get_main_data, get_spots_reflectance
from visualization import show_rgb_custom_illuminant

# Test on random data
def run_random_pso():
    rand = np.random.default_rng()
    low_bound = float(0.0)
    high_bound = float(1.0)
    n_iterations = 500
    n_wavelength = 200

    pso_config = PSOConfig(30, n_iterations, 0.95, 5, 5)
    pso = PSO(pso_config)

    sample1 = rand.uniform(low_bound, high_bound, size=n_wavelength).astype(np.float64)
    sample2 = rand.uniform(low_bound, high_bound, size=n_wavelength).astype(np.float64)
    wavelength = np.arange(400, 400 + n_wavelength*2, 2, dtype=float)

    led = rand.uniform(low_bound, high_bound, size=(10, n_wavelength)).astype(np.float64)
    global_pos, global_cost = pso.run_pso(sample1, sample2, wavelength, led, CostFunctionMode.CIEDE, OptimizeMode.MIN)
    print(global_pos)
    print(global_cost)
    

def run_pso(sample1, sample2, wavelength, led):
    n_iterations = 500
    pso_config = PSOConfig(30, n_iterations, 0.95, 5, 5)
    pso = PSO(pso_config)
    global_pos, global_cost = pso.run_pso(sample1, sample2, wavelength, led, CostFunctionMode.CIEDE, OptimizeMode.MIN)
    print(global_pos)
    print(global_cost)
    return global_pos, global_cost

    


if __name__ == '__main__':
    
    sample_folder = "./Practical_work/Rosita_Aguirre_Plascencia/capture/"
    sample_name = "sample_scan_0043"
    
    leds_folder = "./Practical_work/LED_emission_spectra/LED_emission_spectra_expTime_85ms_avg_10_spectra/"
    
    reflectance, ref_wavelengths, leds_spectra = get_main_data(sample_folder, sample_name, leds_folder)
     
    spot_orange = [290, 185] # Orange
    spot_red = [275, 355] # Red
    spots = [spot_orange, spot_red]
    
    spots_reflectance = get_spots_reflectance(spots, reflectance)
    
    # run_random_pso() Run on a rand case
    global_pos, global_cost =  run_pso(spots_reflectance[0], spots_reflectance[1], ref_wavelengths, leds_spectra)
    custom_illuminant = global_pos @ leds_spectra
    
    show_rgb_custom_illuminant(reflectance, ref_wavelengths, custom_illuminant)
