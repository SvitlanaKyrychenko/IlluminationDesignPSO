import os
import numpy as np

def save_emission_spectrum(folder, filename, wavelengths, spectrum):
    data = np.column_stack((wavelengths, spectrum))
    filepath = os.path.join(folder, filename)
    np.savetxt(filepath, data, fmt="%.6f", delimiter=" ")
    print(f"Saved emission spectrum to {filepath}")
