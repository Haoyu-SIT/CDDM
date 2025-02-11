import os
import netCDF4 as nc
import numpy as np
import os
import properscoring as ps
import math
import numpy as np
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import netCDF4 as nc
from PIL import Image
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
import numpy as np
import netCDF4 as nc
import os
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
import matplotlib
import matplotlib.pyplot as plt

from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_NC
from tqdm.autonotebook import tqdm

import numpy as np
import netCDF4 as nc
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import BoundaryNorm
import matplotlib
import matplotlib.pyplot as plt

def mae(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((ens - hr)).sum(axis=0) / ens.shape[0]


def bias(ens,hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return(ens - hr).sum(axis=0) / ens.shape[0]


def bias_median(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.median(ens, axis=0) - hr


def rmse(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.sqrt(((ens - hr) ** 2).sum(axis=(0)) / ens.shape[0])

def mae_mean(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((ens.mean(axis=0) - hr))

def mae_median(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((np.median(ens, axis=0) - hr))


@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])

        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler)
)
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            if self.config.training.use_DDP:
                model_states['ori_latent_mean'] = self.net.module.ori_latent_mean
                model_states['ori_latent_std'] = self.net.module.ori_latent_std
                model_states['cond_latent_mean'] = self.net.module.cond_latent_mean
                model_states['cond_latent_std'] = self.net.module.cond_latent_std
            else:
                model_states['ori_latent_mean'] = self.net.ori_latent_mean
                model_states['ori_latent_std'] = self.net.ori_latent_std
                model_states['cond_latent_mean'] = self.net.cond_latent_mean
                model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        print(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        print(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)
            # break

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        print(self.net.ori_latent_mean)
        print(self.net.ori_latent_std)
        print(self.net.cond_latent_mean)
        print(self.net.cond_latent_std)

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        (x, x_name), (x_cond, x_cond_name) = batch
        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])

        loss, additional_info = net(x, x_cond)
        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            if additional_info.__contains__('recloss_noise'):
                self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
            if additional_info.__contains__('recloss_xy'):
                self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
        return loss




    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        print("Sampling started...")
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        condition_path = make_dir(os.path.join(sample_path, 'condition'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))
        metrics_path = os.path.join(sample_path, 'metrics_log.txt')

        (x, x_name), (x_cond, x_cond_name) = batch

        batch_size = x.shape[0] 
        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond = x_cond[0:batch_size].to(self.config.training.device[0])

        # Generate samples
        sample = net.sample(x_cond, clip_denoised=self.config.testing.clip_denoised).to('cpu')
        with open(metrics_path, 'a') as log_file:
            log_file.write("x_name[i], mse, rmse_val, mae_val,  bias_median_val, skil_BI_val, mae_mean_val, mae_median_val\n")
            for i in range(batch_size):
                # Calculate metrics before saving
                x_cpu = x[i].to('cpu')
                sample_cpu = sample[i].to('cpu')
                x_cpu = torch.exp(x_cpu) - 1

                
                sample_cpu = torch.exp(sample_cpu) - 1

                # 现有的损失计算
                mse = torch.mean((x_cpu - sample_cpu) ** 2).item()  # MSE计算后转化为标量
                rmse_val = math.sqrt(mse)  # RMSE
                mae_val = torch.mean(torch.abs(x_cpu - sample_cpu)).item()  # MAE

                sample_cpu = sample_cpu.numpy()  # 同上
                x_cpu = x_cpu.numpy()  # 将 PyTorch 张量转换为 NumPy 数组
                # 新增的损失计算
                # 假设这些函数已经定义好了，且接受的参数为(x_cpu, sample_cpu)
                bias_median_val = np.mean(bias_median(x_cpu, sample_cpu))
                skil_BI_val = np.mean(ps.crps_ensemble(sample_cpu, x_cpu) ) # 直接使用numpy数组# 注意调整维度以匹配函数期望的输入
                mae_mean_val = np.mean(mae_mean(x_cpu, sample_cpu))
                mae_median_val = np.mean(mae_median(x_cpu, sample_cpu))

                # Save NC files
                save_single_NC(sample_cpu, sample_path, f'sample_{x_cond_name[i]}_{x_name[i]}', variable_name="sample")
                save_single_NC(torch.exp(x_cond[i]) - 1, condition_path, f'condition_{x_cond_name[i]}', variable_name="condition")
                save_single_NC(x_cpu, gt_path, f'gt_{x_name[i]}', variable_name="ground_truth")

                log_file.write(f"{x_name[i]}, {mse}, {rmse_val}, {mae_val}, {bias_median_val}, {skil_BI_val},  {mae_mean_val},{mae_median_val}\n")



    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):

        print("start sample_to_eval")
        condition_path = make_dir(os.path.join(sample_path, 'condition'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))
        result_path = make_dir(os.path.join(sample_path, str(self.config.model.BB.params.sample_step)))

        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        batch_size = self.config.data.test.batch_size
        sample_num = self.config.testing.sample_num+1
        for test_batch in batch_size:
            (x, x_name), (x_cond, x_cond_name) = test_batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            for j in range(sample_num):
                sample = net.sample(x_cond, clip_denoised=False)
                for i in range(batch_size):
                    condition = x_cond[i].detach().clone()
                    gt = x[i]
                    result = sample[i]
                    if j == 0:
                        save_single_NC(condition, condition_path, f'{x_cond_name[i]}.nc')
                        save_single_NC(gt, gt_path, f'{x_name[i]}.nc')
                    result_path_i = make_dir(os.path.join(result_path, x_name[i])) if sample_num > 1 else result_path
                    save_single_NC(result, result_path_i, f'output_{j}.nc' if sample_num > 1 else f'{x_name[i]}.nc')

