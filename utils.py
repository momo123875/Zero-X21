import os
import torch
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity  # 从skimage库中导入结构相似性指标
import nibabel as nib  # 导入nibabel库来处理NIfTI文件


def read_img(in_path, single_file=False):
    """
    读取图像，并归一化到[0, 1]。

    参数:
    - in_path: 图像路径或包含图像的文件夹路径。
    - single_file: 是否为单个文件。

    返回:
    - 读取并归一化的图像或图像列表。
    """
    img_list = []  # 初始化图像列表

    if single_file:  # 如果是单个文件
        filenames = [os.path.basename(in_path)]  # 获取文件名
        in_path = os.path.dirname(in_path)  # 获取文件所在目录
    else:  # 如果是文件夹
        filenames = [
            f
            for f in os.listdir(in_path)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ]  # 获取指定类型文件列表

    if not filenames:  # 如果文件列表为空
        raise ValueError(f"No valid image files found in the path: {in_path}")

    for f in filenames:  # 遍历所有文件
        img_path = os.path.join(in_path, f)  # 拼接文件路径
        try:
            img = nib.load(img_path)  # 读取图像，返回Nibabel图像对象
            img_vol = img.get_fdata().astype(
                np.float32
            )  # 转换为NumPy数组，形状: (w, h, d)
            min_val, max_val = img_vol.min(), img_vol.max()  # 获取最小值和最大值
            if max_val - min_val > 0:  # 如果图像中存在多于一个不同的值
                img_vol = (img_vol - min_val) / (max_val - min_val)  # 归一化到[0, 1]
            else:  # 如果图像中所有像素值相同
                img_vol = img_vol - min_val  # 将所有值归一化到0
            img_list.append(img_vol)  # 将处理后的图像添加到列表中
        except Exception as e:  # 捕获读取图像时的异常
            print(f"Error reading {img_path}: {e}")  # 打印错误信息

    if not img_list:  # 检查是否成功读取了图像
        raise ValueError("No images were successfully loaded from the provided paths.")

    return img_list[0] if single_file else img_list  # 返回处理后的图像或图像列表


def min_max_scale(X, s_min, s_max):
    """
    对数组进行归一化。

    参数:
    - X: 输入数组。
    - s_min: 归一化的最小值。
    - s_max: 归一化的最大值。

    返回:
    - 归一化后的数组。
    """
    X_min, X_max = X.min(axis=0), X.max(axis=0)  # 获取数组的最小值和最大值
    return s_min + (X - X_min) * (s_max - s_min) / (
        X_max - X_min
    )  # 归一化到[s_min, s_max]范围内


def make_coord_anatomy(file_path):
    """
    获取NIfTI图像的坐标网格。

    参数:
    - file_path: NIfTI图像路径。

    返回:
    - 坐标张量，形状为 (N, 3)。
    """
    image = nib.load(file_path)  # 读取NIfTI图像

    img_affine = image.affine  # 获取图像的仿射变换矩阵
    img_vol = image.get_fdata()  # 获取图像数据并转换为NumPy数组，形状: (w, h, d)
    w, h, d = img_vol.shape  # 获取图像的维度(w, h, d)

    W = np.linspace(0, w - 1, w)  # 生成w维度的线性点
    H = np.linspace(0, h - 1, h)  # 生成h维度的线性点
    D = np.linspace(0, d - 1, d)  # 生成d维度的线性点
    points = np.meshgrid(W, H, D, indexing="ij")  # 生成网格点
    points = (
        np.stack(points).transpose(1, 2, 3, 0).reshape(-1, 3)
    )  # 将网格点转换为坐标点，形状为(w*h*d, 3)

    coordinates = nib.affines.apply_affine(
        img_affine, points
    )  # 应用仿射变换，将原始坐标转换为新的坐标,形状为(w*h*d, 3)
    coordinates_arr_norm = min_max_scale(
        coordinates, s_min=-1, s_max=1
    )  # 归一化坐标到[-1, 1]
    return torch.tensor(coordinates_arr_norm, dtype=torch.float32)  # 转换为张量并返回


def make_coord(shape, ranges=None, flatten=True):
    """
    生成网格中心的坐标。

    参数:
    - shape: 输入形状。
    - ranges: 范围。
    - flatten: 是否展平。

    返回:
    - 坐标张量。
    """
    coord_seqs = []  # 初始化坐标序列列表
    for i, n in enumerate(shape):  # 遍历输入形状的每个维度
        v0, v1 = (
            ranges[i] if ranges else (-1, 1)
        )  # 默认范围为[-1, 1]，否则使用提供的范围
        r = (v1 - v0) / (2 * n)  # 计算网格中心的步长
        seq = v0 + r + (2 * r) * torch.arange(n).float()  # 生成网格中心的坐标序列
        coord_seqs.append(seq)  # 将坐标序列添加到列表中

    ret = torch.stack(
        torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1
    )  # 生成网格并堆叠为坐标张量
    return (
        ret.view(-1, ret.shape[-1]) if flatten else ret
    )  # 展平坐标张量（如果需要），并返回


def write_img(vol, out_path, ref_path, new_spacing=None):
    """
    写入图像。

    参数:
    - vol: 图像数据。
    - out_path: 输出路径。
    - ref_path: 参考图像路径。
    - new_spacing: 新的间距。

    返回:
    - None
    """
    try:
        img_ref = sitk.ReadImage(ref_path)  # 读取参考图像
        img = sitk.GetImageFromArray(vol)  # 将NumPy数组转换为SimpleITK图像
        img.SetDirection(img_ref.GetDirection())  # 设置图像方向
        img.SetSpacing(
            new_spacing if new_spacing else img_ref.GetSpacing()
        )  # 设置图像间距
        img.SetOrigin(img_ref.GetOrigin())  # 设置图像原点
        sitk.WriteImage(img, out_path)  # 写入图像到输出路径
        print("Save to:", out_path)  # 打印保存路径
    except Exception as e:  # 捕获写入图像时的异常
        print(f"Error writing image: {e}")  # 打印错误信息
