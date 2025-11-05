import os
import utils
from torch.utils import data
from torch.utils.data import Dataset
import torch


class ImgTrain(Dataset):
    def __init__(
        self,
        in_path_lr_1,
        in_path_lr_2,
        in_path_hr,
        coord,
        sample_size,
        is_train,
    ):
        """
        初始化ImgTrain数据集。

        参数:
        - in_path_lr_1: 第一种低清图像文件夹路径。
        - in_path_lr_2: 第二种低清图像文件夹路径。
        - in_path_hr: 高清图像文件夹路径。
        - sample_size: 采样大小，用于训练时随机抽取的体素点数。
        - is_train: 是否为训练模式。
        """
        self.is_train = is_train
        self.sample_size = sample_size

        self.coord = coord
        self.in_path_lr_1 = in_path_lr_1
        self.in_path_lr_2 = in_path_lr_2
        self.in_path_hr = in_path_hr
        self.lr_files_1 = sorted(os.listdir(in_path_lr_1))
        self.lr_files_2 = sorted(os.listdir(in_path_lr_2))
        self.hr_files = sorted(os.listdir(in_path_hr))

    def __len__(self):
        """
        返回数据集的长度。
        """
        return len(self.lr_files_1)

    def __getitem__(self, item):
        """
        获取指定索引的图像数据及其patch。

        参数:
        - item: 索引值。

        返回:
        - lr_img_1: 第一种低清图像。
        - lr_img_2: 第二种低清图像。
        - hr_xyzs: 高清图像的坐标列表。
        - hr_img_1: 第一种高清图像。
        - hr_img_2: 第二种高清图像。
        """

        lr_file_1 = self.lr_files_1[item]
        lr_file_2 = lr_file_1.replace("t1", "t2")

        lr_path_1 = os.path.join(self.in_path_lr_1, lr_file_1)
        lr_path_2 = os.path.join(self.in_path_lr_2, lr_file_2)
        hr_path_1 = os.path.join(self.in_path_hr, lr_file_1)
        hr_path_2 = os.path.join(self.in_path_hr, lr_file_2)

        # 读取高清图像和低清图像，格式为 (w，h，d)
        hr_img_1 = utils.read_img(
            in_path=hr_path_1, single_file=True
        )  # 维度: (w，h，d)
        lr_img_1 = utils.read_img(
            in_path=lr_path_1, single_file=True
        )  # 维度: (w，h，d)

        hr_img_2 = utils.read_img(
            in_path=hr_path_2, single_file=True
        )  # 维度: (w，h，d)
        lr_img_2 = utils.read_img(
            in_path=lr_path_2, single_file=True
        )  # 维度: (w，h，d)

        if self.coord == "普通":
            # 生成高清图像的坐标，格式为 (w，h，d)
            hr_xyzs_1 = utils.make_coord(
                hr_img_1.shape, flatten=True
            )  # 维度: (w*h*d, 3)
            hr_xyzs_2 = utils.make_coord(
                hr_img_2.shape, flatten=True
            )  # 维度: (w*h*d, 3)
        else:
            hr_xyzs_1 = utils.make_coord_anatomy(hr_path_1)
            hr_xyzs_2 = utils.make_coord_anatomy(hr_path_2)

        # 确保图像是单通道的，并转换为Tensor
        hr_img_1 = (
            torch.from_numpy(hr_img_1).unsqueeze(0)
            if len(hr_img_1.shape) == 3
            else torch.from_numpy(hr_img_1)
        )
        hr_img_2 = (
            torch.from_numpy(hr_img_2).unsqueeze(0)
            if len(hr_img_2.shape) == 3
            else torch.from_numpy(hr_img_2)
        )
        lr_img_1 = (
            torch.from_numpy(lr_img_1).unsqueeze(0)
            if len(lr_img_1.shape) == 3
            else torch.from_numpy(lr_img_1)
        )
        lr_img_2 = (
            torch.from_numpy(lr_img_2).unsqueeze(0)
            if len(lr_img_2.shape) == 3
            else torch.from_numpy(lr_img_2)
        )

        return lr_img_1, lr_img_2, hr_xyzs_1, hr_img_1, hr_xyzs_2, hr_img_2


def loader_train(
    in_path_lr_1,
    in_path_lr_2,
    in_path_hr,
    batch_size,
    coord,
    sample_size,
    is_train,
    num_workers,
):
    """
    创建训练数据加载器。

    参数:
    - in_path_lr_1: 第一种低清图像文件夹路径。
    - in_path_lr_2: 第二种低清图像文件夹路径。
    - in_path_hr: 高清图像文件夹路径。
    - batch_size: 批量大小。
    - sample_size: 采样大小。
    - is_train: 是否为训练模式。

    返回:
    - DataLoader实例。
    """
    return data.DataLoader(
        dataset=ImgTrain(
            in_path_lr_1=in_path_lr_1,
            in_path_lr_2=in_path_lr_2,
            in_path_hr=in_path_hr,
            coord=coord,
            sample_size=sample_size,
            is_train=is_train,
        ),
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=True,
        num_workers=num_workers,  # 传递num_workers参数
        pin_memory=True,  # 如果使用GPU，建议设置为True以加速数据转移到GPU
    )


class ImgTest(Dataset):
    def __init__(self, in_path_hr, in_path_lr_1, in_path_lr_2, coord):
        """
        初始化ImgTest数据集。

        参数:
        - in_path_hr: 高清图像文件夹路径。
        - in_path_lr_1: 低清图像文件夹1路径。
        - in_path_lr_2: 低清图像文件夹2路径。
        """
        self.in_path_lr_1 = in_path_lr_1
        self.in_path_lr_2 = in_path_lr_2
        self.in_path_hr = in_path_hr
        self.lr_files_1 = sorted(os.listdir(in_path_lr_1))
        self.lr_files_2 = sorted(os.listdir(in_path_lr_2))
        self.hr_files = sorted(os.listdir(in_path_hr))
        self.coord = coord

    def __len__(self):
        return len(self.lr_files_1)

    def __getitem__(self, item):

        lr_file_1 = self.lr_files_1[item]
        lr_file_2 = lr_file_1.replace("t1", "t2")
        # 确保文件存在
        if lr_file_2 not in self.lr_files_2:
            raise IndexError(f"文件 {lr_file_2} 在路径 {self.in_path_lr_2} 中未找到")

        lr_path_1 = os.path.join(self.in_path_lr_1, lr_file_1)
        lr_path_2 = os.path.join(self.in_path_lr_2, lr_file_2)
        hr_path_1 = os.path.join(self.in_path_hr, lr_file_1)
        hr_path_2 = os.path.join(self.in_path_hr, lr_file_2)

        # 读取高清图像和低清图像，格式为 (w，h，d)
        hr_img_1 = utils.read_img(
            in_path=hr_path_1, single_file=True
        )  # 维度: (w，h，d)
        lr_img_1 = utils.read_img(
            in_path=lr_path_1, single_file=True
        )  # 维度: (w，h，d)
        hr_img_2 = utils.read_img(
            in_path=hr_path_2, single_file=True
        )  # 维度: (w，h，d)
        lr_img_2 = utils.read_img(
            in_path=lr_path_2, single_file=True
        )  # 维度: (w，h，d)

        if self.coord == "普通":
            # 生成高清图像的坐标，格式为 (w，h，d)
            xyz_hr_1 = utils.make_coord(
                hr_img_1.shape, flatten=True
            )  # 维度: (w*h*d, 3)
            xyz_hr_2 = utils.make_coord(
                hr_img_2.shape, flatten=True
            )  # 维度: (w*h*d, 3)
        else:
            xyz_hr_1 = utils.make_coord_anatomy(hr_path_1)
            xyz_hr_2 = utils.make_coord_anatomy(hr_path_2)

        # 确保图像是单通道的，并转换为Tensor
        hr_img_1 = (
            torch.from_numpy(hr_img_1).unsqueeze(0)
            if len(hr_img_1.shape) == 3
            else torch.from_numpy(hr_img_1)
        )
        hr_img_2 = (
            torch.from_numpy(hr_img_2).unsqueeze(0)
            if len(hr_img_2.shape) == 3
            else torch.from_numpy(hr_img_2)
        )
        lr_img_1 = (
            torch.from_numpy(lr_img_1).unsqueeze(0)
            if len(lr_img_1.shape) == 3
            else torch.from_numpy(lr_img_1)
        )
        lr_img_2 = (
            torch.from_numpy(lr_img_2).unsqueeze(0)
            if len(lr_img_2.shape) == 3
            else torch.from_numpy(lr_img_2)
        )

        return (
            lr_img_1,
            lr_img_2,
            xyz_hr_1,
            lr_file_1,
            hr_img_1,
            xyz_hr_2,
            hr_img_2,
            lr_file_2,
        )


def loader_test(in_path_hr, in_path_lr_1, in_path_lr_2, coord):
    """
    创建测试数据加载器。

    参数:
    - in_path_hr: 高清图像文件夹路径。
    - in_path_lr_1: 低清图像文件夹1路径。
    - in_path_lr_2: 低清图像文件夹2路径。

    返回:
    - DataLoader实例。
    """
    dataset = ImgTest(
        in_path_hr=in_path_hr,
        in_path_lr_1=in_path_lr_1,
        in_path_lr_2=in_path_lr_2,
        coord=coord,
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
    )


# 使用方法示例
# train_loader = loader_train(
#     in_path_hr="/data/ssd1/JFhuo/DataSet/train_pad_patch",
#     in_path_lr_1="/data/ssd1/JFhuo/DataSet/train_lr_t1_axial_patch",
#     in_path_lr_2="/data/ssd1/JFhuo/DataSet/train_lr_t2_coronal_patch",
#     coord="普通",
#     batch_size=16,
#     sample_size=1,  # 无所谓
#     is_train=True,
#     num_workers=8,  # 调整此参数以优化数据加载
# )
# val_loader = loader_train(
#     in_path_hr="/data/ssd1/JFhuo/DataSet/validate_pad_patch",
#     in_path_lr_1="/data/ssd1/JFhuo/DataSet/validate_lr_t1_axial_patch",
#     in_path_lr_2="/data/ssd1/JFhuo/DataSet/validate_lr_t2_coronal_patch",
#     coord="普通",
#     batch_size=1,
#     sample_size=1,  # 无所谓
#     is_train=False,
#     num_workers=8,  # 调整此参数以优化数据加载
# )

# test_loader = loader_test(
#     in_path_hr="/data/ssd1/JFhuo/DataSet/validate_pad_patch",
#     in_path_lr_1="/data/ssd1/JFhuo/DataSet/validate_lr_t1_axial_patch",
#     in_path_lr_2="/data/ssd1/JFhuo/DataSet/validate_lr_t2_coronal_patch",
#     coord="普通",
# )
# print("ok")
