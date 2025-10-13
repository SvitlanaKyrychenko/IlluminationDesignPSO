import os
import numpy as np
from pathlib import Path


def read_txt_folder(folder_path):
    txt_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.txt')])

    all_data = []

    for i, file in enumerate(txt_files):
        curr_path = os.path.join(folder_path, file)
        data = np.loadtxt(curr_path)

        if i == 0:
            # first file: keep both columns (first column will be wavelengths)
            combined = data[:, :1]  # start with first column
            all_data.append(data[:, 1])  # save 2nd col as first data column
            first_col = data[:, 0]
        else:
            all_data.append(data[:, 1])

    # Stack: first column from file1 + all 2nd columns
    all_data = np.column_stack([first_col] + all_data)
    return all_data


def main():
    current_dir = Path(__file__).resolve().parent
    parent_dir  = current_dir.parent
    path_predicted = os.path.join(parent_dir, "emission_spectrums")
    path_measured = os.path.join(parent_dir, "measured_spectrum")

    predicted_led_spectrum = read_txt_folder(path_predicted)
    measured_led_spectrum = read_txt_folder(path_measured)
    print("t")


if __name__ == '__main__':
    main()
