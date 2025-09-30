import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import spim2rgb, spim2Lab, spim2gray, spim2rgb_with_adaptation
from cost_functions import ciede, rgbde

def get_reflectances_from_coordinates(dataCube, coordinates):
    n = len(coordinates)
    _, _, totalWavelengths = dataCube.shape
    reflectances = np.zeros((n, totalWavelengths))
    
    for i, coord in enumerate(coordinates):
        x, y = coord
        reflectances[i] = dataCube[x, y, :]
    
    return reflectances 
  

def get_cost_between_spots(cube, spot1, spot2, cost_function):
    spot1values = cube[spot1[0], spot1[1]]
    spot2values = cube[spot2[0], spot2[1]]
    
    return cost_function(spot1values, spot2values)


def show_rgb_custom_illuminant(reflectance, wavelengths, customIlluminant, spots, cost_function, global_cost):
    rgb_image_D65 = spim2rgb(reflectance, wavelengths, 'D65', np.nan, np.nan)
    rgb_image_custom = spim2rgb_with_adaptation(reflectance, wavelengths, customIlluminant, np.nan, np.nan)
    
    if cost_function == rgbde:
        title = "RGB Difference: "
        values_d65 = rgb_image_D65
        values_custom = rgb_image_custom
    elif cost_function == ciede: 
        title = "CIE Lab Difference: "
        values_d65 = spim2Lab(reflectance, wavelengths, 'D65')
        values_custom = spim2Lab(reflectance, wavelengths, customIlluminant)
    else:
        title = "Michelson contrast: "
        values_d65 = spim2gray(reflectance, wavelengths, 'D65', np.nan, np.nan)
        values_custom = spim2gray(reflectance, wavelengths, customIlluminant, np.nan, np.nan)
        
        
    diff_D65 = get_cost_between_spots(values_d65, spots[0], spots[1], cost_function)
    diff_custom = get_cost_between_spots(values_custom, spots[0], spots[1], cost_function)
    
    _, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    axes[0].imshow(rgb_image_D65)
    axes[0].scatter([spots[0][1]], [spots[0][0]], s=40, facecolors='none',  edgecolors='white', linewidths=1.5)
    axes[0].scatter([spots[1][1]],  [spots[1][0]],  s=40, facecolors='none', edgecolors='black', linewidths=1.5)
    axes[0].set_title("RBG Image D65")
    axes[0].axis("off")
    
    axes[1].imshow(rgb_image_custom)
    axes[1].scatter([spots[0][1]], [spots[0][0]], s=40, facecolors='none',  edgecolors='white', linewidths=1.5)
    axes[1].scatter([spots[1][1]],  [spots[1][0]],  s=40, facecolors='none', edgecolors='black', linewidths=1.5)
    axes[1].set_title("RBG Custom Illuminant")
    axes[1].axis("off")
    
    
    # Rectangle parameters
    rect_x, rect_y = 10, 10   # position in plot coords
    rect_w, rect_h = 40, 20   # width and height
    
    color_1_D65 = rgb_image_D65[spots[1][0], spots[1][1], :]
    color_2_D65 = rgb_image_D65[spots[0][0], spots[0][1], :]
    
    rect_1_D65  = patches.Rectangle((rect_x, rect_y), rect_w/2, rect_h,
                            linewidth=1, edgecolor='none', facecolor=color_1_D65)
    rect_2_D65  = patches.Rectangle((rect_x + rect_w/2, rect_y), rect_w/2, rect_h,
                            linewidth=1, edgecolor='none', facecolor=color_2_D65)
    axes[2].add_patch(rect_1_D65)
    axes[2].add_patch(rect_2_D65)
    axes[2].set_xlim(0, rect_x + rect_w + 10)
    axes[2].set_ylim(0, rect_y + rect_h + 10)
    axes[2].set_title(f"{title}{diff_D65:.2f}\nBackground Patch - Number Patch")
    axes[2].axis("off")
    
    
    color_1_custom = rgb_image_custom[spots[1][0], spots[1][1], :]
    color_2_custom = rgb_image_custom[spots[0][0], spots[0][1], :]
    
    rect_1_custom  = patches.Rectangle((rect_x, rect_y), rect_w/2, rect_h,
                            linewidth=1, edgecolor='none', facecolor=color_1_custom)
    rect_2_custom  = patches.Rectangle((rect_x + rect_w/2, rect_y), rect_w/2, rect_h,
                            linewidth=1, edgecolor='none', facecolor=color_2_custom)
    axes[3].add_patch(rect_1_custom)
    axes[3].add_patch(rect_2_custom)
    axes[3].set_xlim(0, rect_x + rect_w + 10)
    axes[3].set_ylim(0, rect_y + rect_h + 10)
    axes[3].set_title(f"{title}{diff_custom:.2f}\nBackground Patch - Number Patch")
    axes[3].axis("off")
    
    
    plt.suptitle(f"C best = {global_cost:.2f}")
    plt.tight_layout()
    plt.show()

  
def plot_spds(spds, wavelengths, labels):
    for spd, label in zip(spds, labels):
        plt.plot(wavelengths, spd, label=label)
    
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative Intensity")
    plt.title("Spectral Power Disribution")
    plt.legend()
    plt.show()