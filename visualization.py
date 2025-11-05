import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# 定义输入和输出文件夹路径
input_folder = "test_img_nifti"
output_folder = "./img"

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 获取所有 NIfTI 文件
nifti_files = [
    f for f in os.listdir(input_folder) if f.endswith(".nii.gz") or f.endswith(".nii")
]

# 遍历每个 NIfTI 文件
for file_name in nifti_files:
    # 加载 NIfTI 文件
    file_path = os.path.join(input_folder, file_name)
    nifti_img = nib.load(file_path)

    # 获取图像数据为 NumPy 数组
    img_data = nifti_img.get_fdata()

    # 计算三个方向的中间切片索引
    slice_x = img_data.shape[0] // 2  # X方向
    slice_y = img_data.shape[1] // 2  # Y方向
    slice_z = img_data.shape[2] // 2  # Z方向

    # 标准化图像数据
    img_data_normalized = (img_data - np.min(img_data)) / (
        np.max(img_data) - np.min(img_data)
    )

    # 可视化 X 方向的切片 (Sagittal view)
    # plt.imshow(img_data_normalized[slice_x, :, :], cmap="gray")
    plt.imshow(img_data[slice_x, :, :], cmap="gray")

    plt.title(f"Sagittal (X) Slice {slice_x}")
    plt.axis("off")
    output_path_x = os.path.join(
        output_folder, f"{os.path.splitext(file_name)[0]}_sagittal.png"
    )
    plt.savefig(output_path_x)
    plt.close()

    # 可视化 Y 方向的切片 (Coronal view)
    plt.imshow(img_data[:, slice_y, :], cmap="gray")
    plt.title(f"Coronal (Y) Slice {slice_y}")
    plt.axis("off")
    output_path_y = os.path.join(
        output_folder, f"{os.path.splitext(file_name)[0]}_coronal.png"
    )
    plt.savefig(output_path_y)
    plt.close()

    # 可视化 Z 方向的切片 (Axial view)
    plt.imshow(img_data[:, :, slice_z], cmap="gray")
    plt.title(f"Axial (Z) Slice {slice_z}")
    plt.axis("off")
    output_path_z = os.path.join(
        output_folder, f"{os.path.splitext(file_name)[0]}_axial.png"
    )
    plt.savefig(output_path_z)
    plt.close()

    print(f"Saved slices for {file_name} to {output_folder} (X, Y, Z directions)")
