import data
import torch
import model
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler  # 导入学习率调度器
import os
import warnings
import random
import numpy as np

# 忽略未来警告
warnings.simplefilter("ignore", FutureWarning)


# 设置随机种子函数
def set_seed(seed):
    """
    设置随机种子以确保实验可重复
    """
    random.seed(seed)  # Python 内置随机数生成器的种子
    np.random.seed(seed)  # NumPy 随机数生成器的种子
    torch.manual_seed(seed)  # PyTorch CPU 上的随机数生成器种子
    torch.cuda.manual_seed(seed)  # PyTorch GPU 上的随机数生成器种子
    torch.cuda.manual_seed_all(seed)  # 多 GPU 上的随机数生成器种子
    # 确保每次卷积操作的结果一致，适用于对实验可重复性要求较高的情况
    torch.backends.cudnn.deterministic = True
    # 禁用性能优化算法搜索，使得卷积算法保持一致，但可能会降低性能
    torch.backends.cudnn.benchmark = False


# 在程序开始时调用设置随机种子
set_seed(43)


# 创建日志目录的函数
def create_log_dir(base_dir="./log"):
    """
    动态生成日志目录，使用递增的数字作为目录名。
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    counter = 1
    while True:
        log_dir = os.path.join(base_dir, f"exp_{counter}")
        if not os.path.exists(log_dir):
            break
        counter += 1

    os.makedirs(log_dir)
    return log_dir


# 参数解析函数，定义和解析所有参数
def parse_args():
    parser = argparse.ArgumentParser(description="ArSSR Training Parameters")

    # 模型相关参数
    parser.add_argument(
        "-encoder_name",
        type=str,
        default="RDN",
        help="编码器网络类型 RDN (default), ResCNN, SRResnet",
    )
    parser.add_argument(
        "-decoder_name", type=str, default="MLP", help="MLP 或 SIREN 解码器"
    )
    parser.add_argument("-decoder_depth", type=int, default=8, help="解码器网络深度")
    parser.add_argument("-decoder_width", type=int, default=256, help="解码器网络宽度")
    parser.add_argument("-feature_dim", type=int, default=128, help="特征向量维度")
    parser.add_argument("-fourier_dim", type=int, default=16, help="傅里叶特征维度")
    parser.add_argument(
        "-num_heads", type=int, default=8, help="交叉注意力头的数量"
    )  # 新增参数
    parser.add_argument(
        "-fusion_hidden_dim", type=int, default=128, help="MLPFusion的隐藏层维度"
    )  # 新增参数

    # 数据路径参数
    parser.add_argument(
        "-data_base_path",
        type=str,
        default="/data/ssd1/huojianfei/DataSet_80/",
        help="数据集基础路径",
    )
    parser.add_argument(
        "-hr_data_train",
        type=str,
        default="train_pad_patch",
        help="高分辨率训练数据相对路径",
    )
    parser.add_argument(
        "-lr_data_train_1",
        type=str,
        default="train_lr_t1_axial_patch",
        help="低分辨率训练数据1相对路径",
    )
    parser.add_argument(
        "-lr_data_train_2",
        type=str,
        default="train_lr_t2_coronal_patch",
        help="低分辨率训练数据2相对路径",
    )
    parser.add_argument(
        "-hr_data_val",
        type=str,
        default="validate_pad_patch",
        help="高分辨率验证数据相对路径",
    )
    parser.add_argument(
        "-lr_data_val_1",
        type=str,
        default="validate_lr_t1_axial_patch",
        help="低分辨率验证数据1相对路径",
    )
    parser.add_argument(
        "-lr_data_val_2",
        type=str,
        default="validate_lr_t2_coronal_patch",
        help="低分辨率验证数据2相对路径",
    )

    # 训练超参数
    parser.add_argument("-lr", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("-epoch", type=int, default=1000, help="训练总epoch数")
    parser.add_argument("-batch_size", type=int, default=4, help="批次大小")
    parser.add_argument(
        "-coord", type=str, default="普通", help="坐标系选择：普通或者解剖"
    )
    parser.add_argument("-sample_size", type=int, default=8000, help="采样体素坐标数")

    # 新增日志输出频率参数
    parser.add_argument(
        "-print_freq", type=int, default=8, help="每隔多少次batch输出一次训练日志"
    )

    # 返回解析后的参数
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 初始化 TensorBoard 日志记录器
    log_dir = create_log_dir()
    print(f"Log directory created: {log_dir}")

    # 删除目录中的旧日志文件，保持目录干净
    for filename in os.listdir(log_dir):
        if filename.startswith("event"):
            file_path = os.path.join(log_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    writer = SummaryWriter(log_dir=log_dir)

    # 获取各参数值
    encoder_name = args.encoder_name
    decoder_name = args.decoder_name
    decoder_depth = args.decoder_depth
    decoder_width = args.decoder_width
    feature_dim = args.feature_dim
    fourier_dim = args.fourier_dim
    num_heads = args.num_heads  # 获取 num_heads 参数
    fusion_hidden_dim = args.fusion_hidden_dim  # 获取 fusion_hidden_dim 参数
    data_base_path = args.data_base_path
    hr_data_train = data_base_path + args.hr_data_train
    lr_data_train_1 = data_base_path + args.lr_data_train_1
    lr_data_train_2 = data_base_path + args.lr_data_train_2
    hr_data_val = data_base_path + args.hr_data_val
    lr_data_val_1 = data_base_path + args.lr_data_val_1
    lr_data_val_2 = data_base_path + args.lr_data_val_2
    lr = args.lr
    epoch = args.epoch
    batch_size = args.batch_size
    coord = args.coord
    sample_size = args.sample_size
    print_freq = args.print_freq

    # 打印所有参数
    print("\n".join([f"{arg}: {getattr(args, arg)}" for arg in vars(args)]))
    time.sleep(1)

    # 训练数据加载器
    train_loader = data.loader_train(
        in_path_lr_1=lr_data_train_1,
        in_path_lr_2=lr_data_train_2,
        in_path_hr=hr_data_train,
        batch_size=batch_size,
        coord=coord,
        sample_size=sample_size,
        is_train=True,
        num_workers=4,  # 调整此参数以优化数据加载
    )

    # 验证数据加载器
    val_loader = data.loader_train(
        in_path_hr=hr_data_val,
        in_path_lr_1=lr_data_val_1,
        in_path_lr_2=lr_data_val_2,
        batch_size=batch_size,
        coord=coord,
        sample_size=sample_size,
        is_train=False,
        num_workers=4,  # 调整此参数以优化数据加载
    )

    # 模型 & 优化器
    ArSSR_model = model.ArSSR(
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        feature_dim=feature_dim,
        decoder_depth=int(decoder_depth / 2),  # 修改后的解码器深度，确保与设计需求一致
        decoder_width=decoder_width,
        fourier_dim=fourier_dim,  # 傅里叶特征维度
        num_heads=num_heads,  # 传入 num_heads 参数
        fusion_hidden_dim=fusion_hidden_dim,  # 传入 fusion_hidden_dim 参数
    )

    # GPU 多张处理
    if torch.cuda.is_available():
        device_ids = [0, 1, 2, 3]
        num_gpus = len(device_ids)
        print(f"Number of GPUs specified: {num_gpus}")
        print(f"Using GPUs: {device_ids}")

        # 确保模型的参数在指定的设备上
        ArSSR_model = ArSSR_model.cuda(device_ids[0])

        # 使用 DataParallel 包装模型
        ArSSR_model = nn.DataParallel(ArSSR_model, device_ids=device_ids)
        DEVICE = torch.device(f"cuda:{device_ids[0]}")
    else:
        DEVICE = torch.device("cpu")
        ArSSR_model = ArSSR_model.to(DEVICE)

    # 损失函数和优化器，增加权重衰减作为正则化
    loss_fun = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(
        params=ArSSR_model.parameters(), lr=lr, weight_decay=1e-5  # 添加L2正则化
    )

    # 添加动态学习率调度器，根据验证损失自动调整学习率
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, verbose=True
    )

    # 跟踪最佳验证损失
    best_loss_val = float("inf")

    # 初始化 GradScaler 用于混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    # 训练过程
    for e in range(epoch):
        epoch_start_time = time.time()

        # 训练阶段
        ArSSR_model.train()
        loss_train = 0.0
        for i, (batch) in enumerate(train_loader):
            (
                img_lr_1,
                img_lr_2,
                xyz_hr_1,
                img_hr_1,
                xyz_hr_2,
                img_hr_2,
            ) = batch

            # 将数据转移到 GPU 设备
            img_lr_1 = img_lr_1.to(DEVICE)
            img_lr_2 = img_lr_2.to(DEVICE)
            img_hr_1 = img_hr_1.to(DEVICE).reshape(len(img_hr_1), -1).unsqueeze(-1)
            img_hr_2 = img_hr_2.to(DEVICE).reshape(len(img_hr_2), -1).unsqueeze(-1)
            xyz_hr_1 = xyz_hr_1.to(DEVICE).reshape(len(xyz_hr_1), -1, 3)
            xyz_hr_2 = xyz_hr_2.to(DEVICE).reshape(len(xyz_hr_2), -1, 3)

            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                img_pre_1, img_pre_2 = ArSSR_model(
                    img_lr_1, xyz_hr_1, img_lr_2, xyz_hr_2
                )
                loss_1 = loss_fun(img_pre_1, img_hr_1)
                loss_2 = loss_fun(img_pre_2, img_hr_2)
                loss = (loss_1 + loss_2) / 2

            optimizer.zero_grad()
            # 使用混合精度训练
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_train += loss.item()

            # 每 print_freq 个 batch 输出一次训练日志
            if (i + 1) % print_freq == 0:
                print(
                    f"(TRAIN) Epoch[{e + 1}/{epoch}] Batch[{i + 1}/{len(train_loader)}] - "
                    f"Loss: {loss.item():.6f} （一个batch的）"
                )

        # 将 epoch 级别的损失记录到 TensorBoard
        writer.add_scalar("MES_train", loss_train / len(train_loader), e + 1)

        # 释放内存
        del (
            img_lr_1,
            img_lr_2,
            img_hr_1,
            img_hr_2,
            xyz_hr_1,
            xyz_hr_2,
            img_pre_1,
            img_pre_2,
        )
        torch.cuda.empty_cache()  # 调用 torch.cuda.empty_cache() 释放显存

        # 验证阶段
        ArSSR_model.eval()
        with torch.no_grad():
            loss_val = 0
            for i, (batch) in enumerate(val_loader):
                (
                    img_lr_1,
                    img_lr_2,
                    xyz_hr_1,
                    img_hr_1,
                    xyz_hr_2,
                    img_hr_2,
                ) = batch

                # 将数据转移到 GPU 设备
                img_lr_1 = img_lr_1.to(DEVICE)
                img_lr_2 = img_lr_2.to(DEVICE)
                img_hr_1 = img_hr_1.to(DEVICE).reshape(len(img_hr_1), -1).unsqueeze(-1)
                img_hr_2 = img_hr_2.to(DEVICE).reshape(len(img_hr_2), -1).unsqueeze(-1)
                xyz_hr_1 = xyz_hr_1.to(DEVICE).reshape(len(xyz_hr_1), -1, 3)
                xyz_hr_2 = xyz_hr_2.to(DEVICE).reshape(len(xyz_hr_2), -1, 3)

                # 使用混合精度进行前向传播
                with torch.cuda.amp.autocast():
                    img_pre_1, img_pre_2 = ArSSR_model(
                        img_lr_1, xyz_hr_1, img_lr_2, xyz_hr_2
                    )

                    # 计算验证损失
                    loss_1 = loss_fun(img_hr_1, img_pre_1)
                    loss_2 = loss_fun(img_hr_2, img_pre_2)
                    loss = (loss_1 + loss_2) / 2
                    loss_val += loss.item()

        # 计算验证集上的平均损失
        loss_val_mes = loss_val / len(val_loader)
        writer.add_scalar("MES_val", loss_val_mes, e + 1)

        # 释放内存
        del (
            img_lr_1,
            img_lr_2,
            img_hr_1,
            img_hr_2,
            xyz_hr_1,
            xyz_hr_2,
            img_pre_1,
            img_pre_2,
        )
        torch.cuda.empty_cache()

        # 保存最好的模型
        if loss_val_mes < best_loss_val:
            best_loss_val = loss_val_mes

            # 保存时去掉 "module." 前缀，以确保模型能够被正常加载
            state_dict = ArSSR_model.state_dict()
            if torch.cuda.device_count() > 1:
                state_dict = {
                    k[7:]: v for k, v in state_dict.items()
                }  # 去除 'module.' 前缀

            torch.save(state_dict, f"{log_dir}/best_model_param.pkl")
            print(f"Saved best model with loss: {best_loss_val:.10f}")

        # 更新学习率调度器
        scheduler.step(loss_val_mes)
        print(
            f"Epoch [{e + 1}/{epoch}] completed. Current learning rate: {optimizer.param_groups[0]['lr']}"
        )

    # 保存最后一个模型权重
    state_dict = ArSSR_model.state_dict()
    if torch.cuda.device_count() > 1:
        state_dict = {k[7:]: v for k, v in state_dict.items()}  # 去除 'module.' 前缀

    torch.save(state_dict, f"{log_dir}/last_model_param.pkl")
    print("Saved the last model weights.")

    # 刷新 TensorBoard 缓存
    writer.flush()
