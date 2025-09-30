import numpy as np
from utils import read_envi

def resize_cube(data_cube, new_y_size):
    import numpy as np
    
    mean_y_cube = np.mean(data_cube, axis=0)
    newCube = np.tile(mean_y_cube, (new_y_size, 1, 1))
    
    return newCube


def get_spectral_reflectance_from_files(sample_filename, white_ref_filename, dark_noise_filename, min_wavelength=400, max_wavelength=700, rawFileExt='.raw'):
    sample_cube, sample_wavelengths = read_envi(sample_filename, rawFileExt)
    white_cube, _ = read_envi(white_ref_filename, rawFileExt)
    dark_cube, _ = read_envi(dark_noise_filename, rawFileExt)

    # Crop wavelength and area of interest
    sample_cube, ref_wavelengths = limit_wavelenghts(sample_cube, sample_wavelengths, min_wavelength, max_wavelength)
    white_cube, _ = limit_wavelenghts(white_cube, sample_wavelengths, min_wavelength, max_wavelength)
    dark_cube, _ = limit_wavelenghts(dark_cube, sample_wavelengths, min_wavelength, max_wavelength)

    sample_cube = crop_area_of_interest(sample_cube, 810, 1360, 20, 565)
    white_cube = crop_area_of_interest(white_cube, 810, 1360, 20, 565)
    dark_cube = crop_area_of_interest(dark_cube, 810, 1360, 20, 565)

    # Resize White and Dark cubes to fit the Sample
    ySize, _, _ = sample_cube.shape
    white_cube = resize_cube(white_cube, ySize)
    dark_cube = resize_cube(dark_cube, ySize)
    
    # Calculate pectral reflectance image R
    spectral_reflectance_cube = ((sample_cube - dark_cube) / (white_cube - dark_cube))
    
    return spectral_reflectance_cube, ref_wavelengths


def limit_wavelenghts(spectral_data, original_wavelengths, min_limit = 400, max_limit = 700):
    mask = (original_wavelengths >= min_limit) & (original_wavelengths <= max_limit)
    limited_wavelenghts = original_wavelengths[mask]
    limited_spectral_data = spectral_data[:, :, mask]
    
    return limited_spectral_data, limited_wavelenghts

  
def crop_area_of_interest(spectral_data, min_x, max_x, min_y, max_y):
    croped_spectral_data = spectral_data[min_y:max_y, min_x:max_x, :]
    
    return croped_spectral_data


def read_interp_leds_spectra(folder, reflectance_wavelengths):

    data0 = np.loadtxt(folder + "led1.txt")
    n_wavelengths = data0.shape[0]

    n_leds = 10
    leds_spectra = np.zeros((n_leds, n_wavelengths))

    leds_wavelengths = data0[:, 0]
    leds_spectra[0] = data0[:, 1]

    for i in range(1, n_leds):
        filename = "led" + str(i+1) + ".txt"
        data = np.loadtxt(folder + filename)
        leds_spectra[i] = data[:, 1]

    leds_spectra_interp = np.array([
        np.interp(reflectance_wavelengths, leds_wavelengths, led)
        for led in leds_spectra
    ])
    
    # plot_spds([leds_spectra_interp[1], leds_spectra[1]], [reflectance_wavelengths, leds_wavelengths])

    return leds_spectra_interp


def plot_spds(spds, wavelengths):
    import matplotlib.pyplot as plt
    
    n = len(spds)
    fig, axs = plt.subplots(n, 1, figsize=(6, 3*n), sharex=True)
    
    if n == 1:
        axs = [axs]
    
    for ax, spd, wave in zip(axs, spds, wavelengths):
        ax.plot(wave, spd)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Relative Intensity")
    
    fig.suptitle("Spectral Power Distribution")
    plt.tight_layout()
    plt.show()


def get_main_data(sample_folder, sample_name, leds_folder, min_wavelength=400, max_wavelength=700):
    sample_filename = sample_folder + sample_name
    white_ref_filename = sample_folder + "WHITEREF_" + sample_name
    dark_noise_filename = sample_folder + "DARKREF_" + sample_name
    
    reflectance, ref_wavelengths = get_spectral_reflectance_from_files(sample_filename, white_ref_filename, dark_noise_filename, min_wavelength, max_wavelength)

    leds_spectra = read_interp_leds_spectra(leds_folder, ref_wavelengths)
    
    return reflectance, ref_wavelengths, leds_spectra
    
    
def get_spots_reflectance(spots, all_reflectance):
    
    spots_reflectance = np.array([all_reflectance[spots[0][0], spots[0][1], :], 
                                  all_reflectance[spots[1][0], spots[1][1], :]]
                                 )
    
    spots_reflectance = spots_reflectance / np.max(spots_reflectance, axis=1, keepdims=True)
    
    return spots_reflectance
    