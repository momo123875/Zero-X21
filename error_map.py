import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def load_nifti(file_path):
    nifti_img = nib.load(file_path)
    nifti_data = nifti_img.get_fdata()
    return nifti_data


def calculate_relative_error_slice(data1_slice, data2_slice, epsilon=1e-8):
    # Calculate the relative error for a single slice
    relative_error_slice = np.abs(data1_slice - data2_slice) / (
        np.abs(data1_slice) + epsilon
    )
    return relative_error_slice


def calculate_and_aggregate_error_maps(file1, file2, direction="x", epsilon=1e-8):
    # Load the NIfTI files
    data1 = load_nifti(file1)
    data2 = load_nifti(file2)

    # Initialize an empty map to aggregate errors
    if direction == "x":
        aggregated_error_map = np.zeros((data1.shape[1], data1.shape[2]))
    elif direction == "y":
        aggregated_error_map = np.zeros((data1.shape[0], data1.shape[2]))
    elif direction == "z":
        aggregated_error_map = np.zeros((data1.shape[0], data1.shape[1]))

    # Aggregate errors for each slice
    if direction == "x":
        for i in range(data1.shape[0]):
            slice_error_map = calculate_relative_error_slice(
                data1[i, :, :], data2[i, :, :], epsilon
            )
            aggregated_error_map += slice_error_map

    elif direction == "y":
        for i in range(data1.shape[1]):
            slice_error_map = calculate_relative_error_slice(
                data1[:, i, :], data2[:, i, :], epsilon
            )
            aggregated_error_map += slice_error_map

    elif direction == "z":
        for i in range(data1.shape[2]):
            slice_error_map = calculate_relative_error_slice(
                data1[:, :, i], data2[:, :, i], epsilon
            )
            aggregated_error_map += slice_error_map

    return aggregated_error_map


def save_error_map_image(error_map, direction, save_dir="./img"):
    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot and save the error map
    plt.figure(figsize=(10, 10))
    plt.imshow(error_map, cmap="hot", vmin=0, vmax=np.max(error_map))
    plt.title(f"Aggregated {direction.upper()}-Direction Relative Error Map")
    plt.axis("off")

    # Save the image
    file_name = f"aggregated_relative_error_map_{direction}.png"
    plt.savefig(os.path.join(save_dir, file_name), bbox_inches="tight", pad_inches=0)
    plt.close()


# Example usage
file1 = "BraTS19_CBICA_AZA_1_t1_test.nii.gz"
file2 = "BraTS19_CBICA_AZA_1_t1.nii.gz"

# Calculate and save error map for a specific direction
direction = "z"  # Can be 'x', 'y', or 'z'
aggregated_error_map = calculate_and_aggregate_error_maps(file1, file2, direction)
save_error_map_image(aggregated_error_map, direction)
