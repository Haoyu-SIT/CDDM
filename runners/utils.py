import os

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from datetime import datetime

from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.basemap import Basemap
from torchvision.utils import make_grid, save_image
from Register import Registers
from datasets.custom import CustomAlignedDataset,RainfallDataset
import netCDF4 as nc
def remove_file(fpath):
    if os.path.exists(fpath):
        os.remove(fpath)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)
    return dir


def make_save_dirs(args, prefix, suffix=None, with_time=False):
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S") if with_time else ""
    suffix = suffix if suffix is not None else ""

    result_path = make_dir(os.path.join(args.result_path, prefix, suffix, time_str))
    image_path = make_dir(os.path.join(result_path, "image"))
    log_path = make_dir(os.path.join(result_path, "log"))
    checkpoint_path = make_dir(os.path.join(result_path, "checkpoint"))
    sample_path = make_dir(os.path.join(result_path, "samples"))
    sample_to_eval_path = make_dir(os.path.join(result_path, "sample_to_eval"))
    print("create output path " + result_path)
    return image_path, checkpoint_path, log_path, sample_path, sample_to_eval_path


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Parameter') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_optimizer(optim_config, parameters):
    if optim_config.optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay,
                                betas=(optim_config.beta1, 0.999))
    elif optim_config.optimizer == 'RMSProp':
        return torch.optim.RMSprop(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay)
    elif optim_config.optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=optim_config.lr, momentum=0.9)
    else:
        return NotImplementedError('Optimizer {} not understood.'.format(optim_config.optimizer))


def get_dataset(data_config):
    #这里后期改一下，把train dataset和testdataset改成不一样的。
    dataset_params = data_config.dataset_config
    access_dir = dataset_params.access_dir
    awap_dir = dataset_params.awap_dir
    start_date = dataset_params.start_date
    end_date = dataset_params.end_date
    leading_time_we_use = dataset_params.leading_time_we_use
    target_size = dataset_params.target_size
    # 创建数据集实例
    train_dataset = Registers.datasets[data_config.dataset_type](
        access_dir, awap_dir, start_date, end_date, leading_time_we_use,  target_size, stage='train')
    val_dataset = Registers.datasets[data_config.dataset_type](
        access_dir, awap_dir, start_date, end_date, leading_time_we_use,  target_size, stage='val')
    test_dataset = Registers.datasets[data_config.dataset_type](
        access_dir, awap_dir, start_date, end_date, leading_time_we_use,  target_size, stage='test')

    return train_dataset, val_dataset, test_dataset


import netCDF4 as nc
import os
import torch


@torch.no_grad()
def save_single_NC(tensor, save_path, file_name, variable_name="pr"):
        if isinstance(tensor, torch.Tensor):
            data = tensor.detach().cpu().numpy()
        else:
            data = tensor
        # 假定prcp_colours和prcp_colormap已经定义
        prcp_colours = [
            "#FFFFFF", '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4',
            '#1d91c0', '#225ea8', '#253494', '#4B0082', "#800080", '#8B0000'
        ]
        if isinstance(save_path, torch.Tensor):
            save_path = save_path.item()  # 假设是标量Tensor
        save_path = str(save_path)  # 确保是字符串

        # 同样确保文件名是字符串类型
        file_name = str(file_name)
        file_name, _ = os.path.splitext(file_name)#去除文件的后缀名
        prcp_colormap = matplotlib.colors.ListedColormap(prcp_colours)
        levels = [0, 0.1, 1, 5, 10, 20, 30, 40, 60, 100,150]  # 自定义降水级别

        # 确保数据是numpy数组
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        # 创建输出目录
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, file_name)

        with nc.Dataset(file_path, 'w', format='NETCDF4') as dataset:
            lat_data = np.linspace(-39.2, -18.6, data.shape[1])
            lon_data = np.linspace(140.6, 153.9, data.shape[2])
            dataset.createDimension('lat', len(lat_data))
            dataset.createDimension('lon', len(lon_data))
            lat_var = dataset.createVariable('lat', np.float32, ('lat',))
            lon_var = dataset.createVariable('lon', np.float32, ('lon',))
            # 如果数据包含时间维度，首先创建时间维度
            if 'time' not in dataset.dimensions:
                time_dim = dataset.createDimension('time', None)  # None表示这是一个无限维度

            lat_var[:] = lat_data
            lon_var[:] = lon_data

            # 创建并填充pr变量
            pr_var = dataset.createVariable(variable_name, np.float32, ('time', 'lat', 'lon'))
            pr_var[:] = data  # 假设data形状为[时间(层数), 纬度, 经度]

        plt.figure()
        m = Basemap(projection='mill', llcrnrlat=min(lat_data), urcrnrlat=max(lat_data),
                    llcrnrlon=min(lon_data), urcrnrlon=max(lon_data), resolution='c')
        m.drawcoastlines()
        m.drawparallels(np.arange(np.floor(min(lat_data)), np.ceil(max(lat_data)), 5), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(np.floor(min(lon_data)), np.ceil(max(lon_data)), 5), labels=[0, 0, 0, 1])
        x, y = m(*np.meshgrid(lon_data, lat_data))
        norm = BoundaryNorm(levels, ncolors=len(prcp_colours), clip=True)
        cs = m.pcolormesh(x, y, data[0], cmap=prcp_colormap, norm=norm)
        plt.colorbar(cs, shrink=0.5, aspect=5)
        plt.title(f'Precipitation')
        plt.savefig(os.path.join(save_path, f'{file_name}.png'))
        plt.close()

@torch.no_grad()
def get_image_grid(batch, grid_size=4, to_normal=True):
    batch = batch.detach().clone()
    image_grid = make_grid(batch, nrow=grid_size)
    if to_normal:
        image_grid = image_grid.mul_(0.5).add_(0.5).clamp_(0, 1.)
    image_grid = image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return image_grid
