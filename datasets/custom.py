from datetime import datetime, timedelta
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

import netCDF4 as nc
from Register import Registers
from datasets.base import ImagePathDataset
from datasets.utils import get_image_paths_from_dir
import os
import re

@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/B'))
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/A'))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]


@Registers.datasets.register_with_name('Rainfall_aligned')
class RainfallDataset(Dataset):
    def __init__(self, access_dir, awap_dir, start_date, end_date, leading_time_we_use, target_size, stage='train'):
        #print("Initializing RainfallDataset...")
        self.access_dir = access_dir
        self.awap_dir = awap_dir
        self.ensemble = ['e01', 'e02', 'e03', 'e04', 'e05', 'e06', 'e07', 'e08', 'e09']
        self.start_date = datetime.strptime(start_date, "%Y%m%d")
        self.end_date = datetime.strptime(end_date, "%Y%m%d")
        self.leading_time_we_use = leading_time_we_use  # e.g., (1, 7)
        self.target_size = target_size
        self.stage = stage
        self.filename_list = self.create_filename_list()

        #print(f"RainfallDataset initialized with:")
        #print(f"  ACCESS directory: {self.access_dir}")
        #print(f"  AWAP directory: {self.awap_dir}")
        #print(f"  Date range: {self.start_date} to {self.end_date}")
        #print(f"  Leading time range: {self.leading_time_we_use}")
        #print(f"  Total samples: {len(self.filename_list)}")

    def create_filename_list(self):
        """Generate file lists for ACCESS and AWAP."""
        filename_list = []
        current_date = self.start_date

        #print(f"[DEBUG] Checking ACCESS directory: {self.access_dir}")
        #print(f"[DEBUG] Checking AWAP directory: {self.awap_dir}")

        while current_date <= self.end_date:
            for en in self.ensemble:
                access_dir = os.path.join(self.access_dir, en)
                if not os.path.exists(access_dir):
                    #print(f"[WARNING] ACCESS directory not found: {access_dir}")
                    continue

                for file in os.listdir(access_dir):
                    pattern = re.compile(rf"da_pr_({current_date.strftime('%Y%m%d')})_{en}.nc")
                    match = pattern.match(file)
                    if match:
                        access_date_str = match.group(1)
                        for lt in range(self.leading_time_we_use[0], self.leading_time_we_use[1] + 1):
                            awap_date = datetime.strptime(access_date_str, "%Y%m%d") + timedelta(days=lt)
                            awap_file_path = os.path.join(self.awap_dir, f"{awap_date.strftime('%Y-%m-%d')}.nc")
                            if os.path.exists(awap_file_path):
                                filename_list.append((os.path.join(access_dir, file), awap_file_path, lt))
                                #print(f"[INFO] Matched ACCESS file: {file}")
                                #print(f"        Corresponding AWAP file: {awap_file_path}")
                            else:
                                print(f"[WARNING] AWAP file not found: {awap_file_path}")
            current_date += timedelta(days=1)

        #print(f"Generated filename list with {len(filename_list)} entries.")
        return filename_list

    def resize_data(self, data):
        """Resize data to the target dimensions."""
        if len(data.shape) == 3:  # [time, lat, lon]
            scale_factors = [1, self.target_size[0] / data.shape[1], self.target_size[1] / data.shape[2]]
        else:  # [lat, lon]
            scale_factors = [self.target_size[0] / data.shape[0], self.target_size[1] / data.shape[1]]
        resized_data = zoom(data, scale_factors)
        #print(f"[DEBUG] Resized data to shape {resized_data.shape}")
        return resized_data

    def load_nc_data(self, file_path, variable_name, time_idx=None):
        """Load data from NetCDF files."""
        #print(f"[INFO] Loading file: {file_path}")
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                data = dataset.variables[variable_name][:]
                if len(data.shape) == 3 and time_idx is not None:
                    data = data[time_idx, :, :]
                    #print(f"[DEBUG] Extracted time index {time_idx}, data shape: {data.shape}")
                else:
                    print(f"[DEBUG] Loaded data shape: {data.shape}")
                return np.nan_to_num(data, nan=0.0)
        except Exception as e:
            #print(f"[ERROR] Error loading file {file_path}: {e}")
            return None

    def __len__(self):
        return len(self.filename_list)
    def __getitem__(self, idx):
        if idx >= len(self.filename_list):
            raise IndexError("Index out of range in RainfallDataset")

        # 随机采样逻辑
        random_value = np.random.uniform(0, 1)  # 生成 0 到 1 之间的随机数
        if random_value >= 0.4:#在这里设置概率。
            # 如果随机数大于采样概率，跳过当前样本
            next_idx = (idx + 1) % len(self.filename_list)  # 防止索引越界，循环采样
            return self.__getitem__(next_idx)

        access_filepath, awap_filepath, leading_time = self.filename_list[idx]

        # 加载 ACCESS 数据
        #print(f"[INFO] Accessing sample {idx}")
        #print(f"       ACCESS file: {access_filepath}")
        #print(f"       AWAP file: {awap_filepath}")
        #print(f"       Leading time: {leading_time}")

        access_data = self.load_nc_data(access_filepath, 'pr', time_idx=leading_time - 1)
        awap_data = self.load_nc_data(awap_filepath, 'precip')

        # 确保数据非负,并且尺寸一致。
        access_data = self.resize_data(access_data)
        awap_data = self.resize_data(awap_data)
        access_data = np.maximum(access_data, 0)* 86400
        awap_data = np.maximum(awap_data, 0)

        # 日志记录转换前的数据范围
        #print(f"[DEBUG] Access data range before log1p: [{access_data.min()}, {access_data.max()}]")
        #print(f"[DEBUG] AWAP data range before log1p: [{awap_data.min()}, {awap_data.max()}]")

        # 应用对数转换
        access_data = np.log1p(access_data)
        awap_data = np.log1p(awap_data)

        # 日志记录转换后的数据范围
        #print(f"[DEBUG] Access data range after log1p: [{access_data.min()}, {access_data.max()}]")
        #print(f"[DEBUG] AWAP data range after log1p: [{awap_data.min()}, {awap_data.max()}]")

        # 转换为张量
        access_data_tensor = torch.from_numpy(access_data).float().unsqueeze(0)  # [1, H, W]
        awap_data_tensor = torch.from_numpy(awap_data).float()

        # 检查 AWAP 数据维度
        if len(awap_data_tensor.shape) == 2:
            awap_data_tensor = awap_data_tensor.unsqueeze(0)  # 转为 [1, H, W]
        #print(f"[DEBUG] AWAP data shape after tensor conversion: {awap_data_tensor.shape}")

        # 扩展为三通道
        access_data_3_channels = access_data_tensor.repeat(3, 1, 1)  # [3, H, W]
        awap_data_3_channels = awap_data_tensor.repeat(3, 1, 1)      # [3, H, W]

        # 文件名
        access_filename = os.path.basename(access_filepath)
        awap_filename = os.path.basename(awap_filepath)

        #print(f"[INFO] Returning data for sample {idx}:")
        #print(f"       ACCESS data shape: {access_data_3_channels.shape}")
        #print(f"       AWAP data shape: {awap_data_3_channels.shape}")
        #print(f"       ACCESS filename: {access_filename}")
        #print(f"       AWAP filename: {awap_filename}")

        return (awap_data_3_channels, awap_filename), (access_data_3_channels, access_filename)


# class RainfallDataset(Dataset):

#     def __init__(self, access_dir, awap_dir, start_date, end_date, leading_time_we_use, target_size, stage='train'):
#         #print("using rainfall aligned dataset!")
#         self.access_dir = access_dir
#         self.awap_dir = awap_dir
#         self.start_date = datetime.strptime(start_date, "%Y%m%d")
#         self.end_date = datetime.strptime(end_date, "%Y%m%d")
#         self.leading_time_we_use = leading_time_we_use
#         self.ensemble = ['e01','e02','e03','e04','e05','e06','e07','e08','e09']
#         self.stage = stage
#         self.filename_list = self.create_filename_list()
#         self.target_size = target_size



#     def create_filename_list(self):
#         if self.stage == 'val':
#             # 假设验证集是时间范围内延后3个月
#             self.start_date = self.end_date 
#             self.end_date = self.end_date + timedelta(days=120)
#         elif self.stage == 'test':
#             self.start_date = datetime.strptime("20070101", "%Y%m%d")
#             self.end_date = datetime.strptime("20071231", "%Y%m%d")
#             #为了测试年份，需要改动这里，可以加到yaml文件的配置中。
#         filename_list = []
#         current_date = self.start_date
#         while current_date <= self.end_date:
#             for en in self.ensemble:
#                 pattern = re.compile(rf"da_pr_({current_date.strftime('%Y%m%d')})_{en}_.*_LT(\d+).nc")
#                 for file in os.listdir(os.path.join(self.access_dir, en)):
#                     match = pattern.match(file)
#                     if match:
#                         access_date_str, lt_str = match.groups()
#                         lt = int(lt_str)
#                         if self.leading_time_we_use[0] <= lt <= self.leading_time_we_use[1]:
#                             access_date = datetime.strptime(access_date_str, "%Y%m%d")
#                             awap_date = access_date + timedelta(days=lt)
#                             awap_filename = os.path.join(self.awap_dir, f"{awap_date.strftime('%Y-%m-%d')}.nc")
#                             if os.path.exists(os.path.join(self.access_dir, en, file)) and os.path.exists(awap_filename):
#                                 filename_list.append((os.path.join(self.access_dir, en, file), awap_filename))
#             current_date += timedelta(days=1)
#         return filename_list

#     def resize_data(self, data):
#         # 插值数据到目标尺寸
#         scale_factors = [self.target_size[0] / data.shape[0], self.target_size[1] / data.shape[1]]
#         resized_data = zoom(data, scale_factors)
#         return resized_data


#     def load_nc_data(self, file_path, variable_name):
#         with nc.Dataset(file_path, 'r') as dataset:
#             data = dataset.variables[variable_name][:]
#             data = data[0, :, :] if len(data.shape) == 3 else data
#             data = np.nan_to_num(data, nan=0.0)
#             return data

#     def __len__(self):
#         return len(self.filename_list)

#     def __getitem__(self, idx):
#         if idx < len(self.filename_list):
#             access_filepath, awap_filepath = self.filename_list[idx]

#             # 加载数据
#             access_data = self.load_nc_data(access_filepath, 'pr')
#             awap_data = self.load_nc_data(awap_filepath, 'precip')
#             # #print(access_filepath)
#             # #print(awap_filepath)
#             # #print("让我们重新开始吧！")
#             # #print(f"原始 ACCESS 数据形状: {access_data.shape}, AWAP 数据形状: {awap_data.shape}")
#             access_data = self.resize_data(access_data) * 86400 # 保持单位一致
#             awap_data = self.resize_data(awap_data)
#             awap_data = np.where(awap_data > -1, awap_data, 0)  # 将小于-1的值设为0或其他小正值
#             access_data = np.where(access_data > -1, access_data, 0)  # 将小于-1的值设为0或其他小正值

#             access_data = np.log1p(access_data)
#             awap_data = np.log1p(awap_data)
#             # #print(access_data,awap_data)


#             # 假设access_data和awap_data已经是单通道的torch张量，并且它们的形状为[N, H, W]，其中
#             # N是批次大小，H是高度，W是宽度。

#             # 将单通道数据复制三次并沿着通道维度（dim=1）堆叠起来
#           # 将NumPy数组转换为PyTorch张量
#             access_data_tensor = torch.from_numpy(access_data).float()
#             awap_data_tensor = torch.from_numpy(awap_data).float()

#             access_data = access_data_tensor.unsqueeze(0)
#             awap_data = awap_data_tensor.unsqueeze(0)


#             access_data_3_channels = access_data.repeat(3, 1, 1)
#             awap_data_3_channels = awap_data.repeat(3, 1, 1)


#             access_filename = os.path.basename(access_filepath)
#             awap_filename = os.path.basename(awap_filepath)

#             # 返回数据和文件名
#             return (awap_data_3_channels, awap_filename),(access_data_3_channels, access_filename) 
#         else:
#             raise IndexError("Index out of range in RainfallDataset")