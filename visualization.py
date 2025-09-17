from utils import spim2rgb
from cost_functions import rgbde

def get_reflectances_from_coordinates(dataCube, coordinates):
    import matplotlib.pyplot as plt
    import numpy as np
    
    n = len(coordinates)
    _, _, totalWavelengths = dataCube.shape
    reflectances = np.zeros((n, totalWavelengths))
    
    for i, coord in enumerate(coordinates):
        x, y = coord
        reflectances[i] = dataCube[x, y, :]
    
    return reflectances 
  
  
def show_rgb_custom_illuminant(reflectance, wavelengths, customIlluminant):
    import matplotlib.pyplot as plt
    
    rgbD65Image = spim2rgb(reflectance, wavelengths)
    # rgbD65Diff = rgbde(rgbD65Image[spots[0][0],:], rgbD65Image[spots[1]])
    
    rgbCustomImage = spim2rgb(reflectance, wavelengths, customIlluminant)
    # rgbCustomDiff = rgbde(rgbCustomImage, patches[0], patches[1])
    
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(rgbD65Image)
    axes[0].set_title(f"RBG D65 - Diff: ")
    axes[0].axis("off")
    
    axes[1].imshow(rgbCustomImage)
    axes[1].set_title(f"RBG D65 - Diff:")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()

  
def plot_spds(spds, wavelengths, labels):
    import matplotlib.pyplot as plt
    
    for spd, label in zip(spds, labels):
        plt.plot(wavelengths, spd, label=label)
    
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative Intensity")
    plt.title("Spectral Power Disribution")
    plt.legend()
    plt.show()