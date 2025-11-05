import os
import numpy as np
import torch
import random
import argparse
import data
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import model
import nibabel as nib
import lpips


# 计算3D图像的LPIPS值的函数
def calculate_lpips_for_3d_image(lpips_metric, img_pre, img_hr):
    lpips_metric.eval()  # 确保LPIPS模型处于评估模式
    lpips_values = []
    depth = img_pre.size(2)

    # 对每个深度切片计算LPIPS值
    for i in range(depth):
        slice_pre = img_pre[:, :, i, :, :]  # 提取2D切片
        slice_hr = img_hr[:, :, i, :, :]  # 提取2D切片

        # 计算每个切片的LPIPS值
        lpips_value = lpips_metric(slice_pre, slice_hr)
        lpips_values.append(lpips_value.item())

    # 计算所有切片的LPIPS值的平均值
    avg_lpips_value = sum(lpips_values) / len(lpips_values)
    return avg_lpips_value


# 保存图像为NIfTI文件
def save_image(image, path):
    image = image.cpu().numpy()  # 从 GPU 移动到 CPU
    if image.ndim == 5:
        image = image.squeeze(0)  # [B, C, D, H, W] -> [C, D, H, W]
    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)  # [C, D, H, W] -> [D, H, W]
    image = image.astype(np.float32)  # 确保数据类型为 float32
    nifti_img = nib.Nifti1Image(image, affine=np.eye(4))  # 创建NIfTI对象
    nib.save(nifti_img, path)  # 保存为NIfTI文件


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-encoder", type=str, default="RDN", help="编码器网络类型")
    parser.add_argument("-decoder", type=str, default="MLP", help="解码器网络类型")
    parser.add_argument("-depth", type=int, default=8, help="解码器网络的深度")
    parser.add_argument("-width", type=int, default=256, help="解码器网络的宽度")
    parser.add_argument("-feature_dim", type=int, default=128, help="特征向量的维度")
    parser.add_argument(
        "-pre_trained_model",
        type=str,
        default="/data/hdd1/huojianfei/Arssr_MIA_Test3/exp_1/best_model_param.pkl",
        help="预训练模型路径",
    )
    parser.add_argument(
        "-data_base_path",
        type=str,
        default="/data/hdd1/huojianfei/DataSet",
        help="数据集路径",
    )
    parser.add_argument(
        "-save_path", type=str, default="./result_patch", help="保存路径"
    )

    parser.add_argument("-fourier_dim", type=int, default=16, help="傅里叶特征的维度")

    parser.add_argument("-num_heads", type=int, default=8, help="交叉注意力的头数")
    parser.add_argument(
        "-fusion_hidden_dim", type=int, default=128, help="MLPFusion的隐藏层维度"
    )  # 新增参数
    args = parser.parse_args()

    # 检测是否有可用的GPU，并设置设备
    if torch.cuda.is_available():
        print("使用GPU")
        torch.cuda.set_device(0)  # 使用第0号GPU
        DEVICE = torch.device("cuda:1")
    else:
        DEVICE = torch.device("cpu")

    # 加载模型
    ArSSR = model.ArSSR(
        encoder_name=args.encoder,
        decoder_name=args.decoder,
        feature_dim=args.feature_dim,
        decoder_depth=int(args.depth / 2),  # 优化解码器深度
        decoder_width=args.width,
        fourier_dim=args.fourier_dim,
        num_heads=args.num_heads,
        fusion_hidden_dim=args.fusion_hidden_dim,
    ).to(DEVICE)

    # 加载预训练模型（仅加载权重）
    ArSSR.load_state_dict(
        torch.load(args.pre_trained_model, map_location=DEVICE), strict=False
    )

    os.makedirs(args.save_path, exist_ok=True)

    # 加载测试数据集
    test_loader = data.loader_test(
        in_path_hr=f"{args.data_base_path}/validate_pad_patch",
        in_path_lr_1=f"{args.data_base_path}/validate_lr_t1_axial_patch",
        in_path_lr_2=f"{args.data_base_path}/validate_lr_t2_coronal_patch",
        coord="普通",
    )

    # 将度量指标移动到正确的设备上
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_metric = lpips.LPIPS(net="alex").to(DEVICE)

    # 存储指标的列表
    image_psnr_1, image_ssim_1 = [], []
    image_psnr_2, image_ssim_2 = [], []
    image_lpips_1, image_lpips_2 = [], []
    ArSSR.eval()  # 设置模型为评估模式
    with torch.no_grad():
        for i, (
            img_lr_1,
            img_lr_2,
            xyz_hr_1,
            name_1,
            img_hr_1,
            xyz_hr_2,
            img_hr_2,
            name_2,
        ) in enumerate(test_loader):
            img_lr_1, img_lr_2 = img_lr_1.to(DEVICE), img_lr_2.to(DEVICE)
            img_hr_1, img_hr_2 = img_hr_1.to(DEVICE), img_hr_2.to(DEVICE)
            xyz_hr_1, xyz_hr_2 = xyz_hr_1.to(DEVICE), xyz_hr_2.to(DEVICE)

            # 生成预测图像
            img_pre_1, img_pre_2 = ArSSR(img_lr_1, xyz_hr_1, img_lr_2, xyz_hr_2)

            # 调整维度，使预测图像与真实图像一致
            img_pre_1 = img_pre_1.view_as(img_hr_1)
            img_pre_2 = img_pre_2.view_as(img_hr_2)
            save_image(
                img_pre_1,
                os.path.join(args.save_path, str(name_1[0])),
            )
            save_image(
                img_pre_2,
                os.path.join(args.save_path, str(name_2[0])),
            )

            # 计算PSNR和SSIM
            img_pre_1, img_hr_1 = img_pre_1.float(), img_hr_1.float()
            img_pre_2, img_hr_2 = img_pre_2.float(), img_hr_2.float()

            psnr_value_1 = psnr_metric(img_pre_1, img_hr_1)
            psnr_value_2 = psnr_metric(img_pre_2, img_hr_2)

            # 如果PSNR为inf，跳过该块
            if torch.isinf(psnr_value_1) or torch.isinf(psnr_value_2):
                print(f"第{i}次 PSNR 为无穷大，跳过此块")
                continue

            # 计算SSIM
            image_psnr_1.append(psnr_value_1.item())
            image_ssim_1.append(ssim_metric(img_pre_1, img_hr_1).item())
            image_psnr_2.append(psnr_value_2.item())
            image_ssim_2.append(ssim_metric(img_pre_2, img_hr_2).item())

            print(f"第{i}次 PSNR (img_pre_1): {psnr_value_1.item()}")
            print(f"第{i}次 PSNR (img_pre_2): {psnr_value_2.item()}")
            print(f"第{i}次 SSIM (img_pre_1): {image_ssim_1[-1]}")
            print(f"第{i}次 SSIM (img_pre_2): {image_ssim_2[-1]}")
            img_pre_3ch_1 = img_pre_1.expand(-1, 3, -1, -1, -1)
            img_hr_3ch_1 = img_hr_1.expand(-1, 3, -1, -1, -1)
            img_pre_3ch_2 = img_pre_2.expand(-1, 3, -1, -1, -1)
            img_hr_3ch_2 = img_hr_2.expand(-1, 3, -1, -1, -1)
            lpips_value_1 = calculate_lpips_for_3d_image(
                lpips_metric, img_pre_3ch_1, img_hr_3ch_1
            )
            lpips_value_2 = calculate_lpips_for_3d_image(
                lpips_metric, img_pre_3ch_2, img_hr_3ch_2
            )
            image_lpips_1.append(lpips_value_1)
            image_lpips_2.append(lpips_value_2)

    # 计算平均PSNR和SSIM
    print(f"平均 PSNR (img_pre_1): {np.mean(image_psnr_1)}")
    print(f"平均 PSNR (img_pre_2): {np.mean(image_psnr_2)}")
    print(f"平均 SSIM (img_pre_1): {np.mean(image_ssim_1)}")
    print(f"平均 SSIM (img_pre_2): {np.mean(image_ssim_2)}")
    print(f"平均 lpips(img_pre_1): {np.mean(image_lpips_1)}")
    print(f"平均 lpips (img_pre_2): {np.mean(image_lpips_2)}")
    # 计算合并后的平均值
    combined_psnr = np.mean(image_psnr_1 + image_psnr_2)
    combined_ssim = np.mean(image_ssim_1 + image_ssim_2)
    combined_lpips = np.mean(image_lpips_1 + image_lpips_2)
    print(f"最终合并 PSNR 平均值: {combined_psnr}")
    print(f"最终合并 SSIM 平均值: {combined_ssim}")
    print(f"最终合并 lpips 平均值: {combined_lpips}")
