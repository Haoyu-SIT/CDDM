import argparse
import datetime
import pdb
import time

import yaml
import os
import traceback

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm


from runners.base.EMA import EMA
from runners.utils import make_save_dirs, make_dir, get_dataset, remove_file


class BaseRunner(ABC):
    def __init__(self, config):
        self.net = None  # Neural Network
        self.optimizer = None  # optimizer
        self.scheduler = None  # scheduler
        self.config = config  # config from configuration file

        # set training params
        self.global_epoch = 0  # global epoch
        if config.args.sample_at_start:
            self.global_step = -1  # global step
        else:
            self.global_step = 0

        self.GAN_buffer = {}  # GAN buffer for Generative Adversarial Network
        self.topk_checkpoints = {}  # Top K checkpoints

        # set log and save destination
        self.config.result = argparse.Namespace()
        self.config.result.image_path, \
        self.config.result.ckpt_path, \
        self.config.result.log_path, \
        self.config.result.sample_path, \
        self.config.result.sample_to_eval_path = make_save_dirs(self.config.args,
                                                                prefix=self.config.data.dataset_name,
                                                                suffix=self.config.model.model_name)

        self.save_config()  # save configuration file
        self.writer = SummaryWriter(self.config.result.log_path)  # initialize SummaryWriter

        # initialize model
        self.net, self.optimizer, self.scheduler = self.initialize_model_optimizer_scheduler(self.config)

        self.print_model_summary(self.net)

        # initialize EMA
        self.use_ema = False if not self.config.model.__contains__('EMA') else self.config.model.EMA.use_ema
        if self.use_ema:
            self.ema = EMA(self.config.model.EMA.ema_decay)
            self.update_ema_interval = self.config.model.EMA.update_ema_interval
            self.start_ema_step = self.config.model.EMA.start_ema_step
            self.ema.register(self.net)

        # load model from checkpoint
        self.load_model_from_checkpoint()

        # initialize DDP
        if self.config.training.use_DDP:
            self.net = DDP(self.net, device_ids=[self.config.training.local_rank], output_device=self.config.training.local_rank)
        else:
            self.net = self.net.to(self.config.training.device[0])
        # self.ema.reset_device(self.net)

    # save configuration file
    def save_config(self):
        save_path = os.path.join(self.config.result.ckpt_path, 'config.yaml')
        save_config = self.config
        with open(save_path, 'w') as f:
            yaml.dump(save_config, f)

    def initialize_model_optimizer_scheduler(self, config, is_test=False):
        """
        get model, optimizer, scheduler
        :param args: args
        :param config: config
        :param is_test: is_test
        :return: net: Neural Network, nn.Module;
                 optimizer: a list of optimizers;
                 scheduler: a list of schedulers or None;
        """
        net = self.initialize_model(config)
        optimizer, scheduler = None, None
        if not is_test:
            optimizer, scheduler = self.initialize_optimizer_scheduler(net, config)
        return net, optimizer, scheduler

    # load model, EMA, optimizer, scheduler from checkpoint
    def load_model_from_checkpoint(self):
        model_states = None
        if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
            print(f"load model {self.config.model.model_name} from {self.config.model.model_load_path}")
            model_states = torch.load(self.config.model.model_load_path, map_location='cpu')

            self.global_epoch = model_states['epoch']
            self.global_step = model_states['step']

            # load model
            self.net.load_state_dict(model_states['model'])

            # load ema
            if self.use_ema:
                self.ema.shadow = model_states['ema']
                self.ema.reset_device(self.net)

            # load optimizer and scheduler
            if self.config.args.train:
                if self.config.model.__contains__('optim_sche_load_path') and self.config.model.optim_sche_load_path is not None:
                    optimizer_scheduler_states = torch.load(self.config.model.optim_sche_load_path, map_location='cpu')
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(optimizer_scheduler_states['optimizer'][i])

                    if self.scheduler is not None:
                        for i in range(len(self.optimizer)):
                            self.scheduler[i].load_state_dict(optimizer_scheduler_states['scheduler'][i])
        return model_states

    def get_checkpoint_states(self, stage='epoch_end'):
        optimizer_state = []
        for i in range(len(self.optimizer)):
            optimizer_state.append(self.optimizer[i].state_dict())

        scheduler_state = []
        for i in range(len(self.scheduler)):
            scheduler_state.append(self.scheduler[i].state_dict())

        optimizer_scheduler_states = {
            'optimizer': optimizer_state,
            'scheduler': scheduler_state
        }

        model_states = {
            'step': self.global_step,
        }

        if self.config.training.use_DDP:
            model_states['model'] = self.net.module.state_dict()
        else:
            model_states['model'] = self.net.state_dict()

        if stage == 'exception':
            model_states['epoch'] = self.global_epoch
        else:
            model_states['epoch'] = self.global_epoch + 1

        if self.use_ema:
            model_states['ema'] = self.ema.shadow
        return model_states, optimizer_scheduler_states

    # EMA part
    def step_ema(self):
        with_decay = False if self.global_step < self.start_ema_step else True
        if self.config.training.use_DDP:
            self.ema.update(self.net.module, with_decay=with_decay)
        else:
            self.ema.update(self.net, with_decay=with_decay)

    def apply_ema(self):
        if self.use_ema:
            if self.config.training.use_DDP:
                self.ema.apply_shadow(self.net.module)
            else:
                self.ema.apply_shadow(self.net)

    def restore_ema(self):
        if self.use_ema:
            if self.config.training.use_DDP:
                self.ema.restore(self.net.module)
            else:
                self.ema.restore(self.net)

    # Evaluation and sample part
    @torch.no_grad()
    def validation_step(self, val_batch, epoch, step):
        self.apply_ema()
        self.net.eval()
        loss = self.loss_fn(net=self.net,
                            batch=val_batch,
                            epoch=epoch,
                            step=step,
                            opt_idx=0,
                            stage='val_step')
        if len(self.optimizer) > 1:
            loss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=1,
                                stage='val_step')
        self.restore_ema()

    @torch.no_grad()
    def validation_epoch(self, val_loader, epoch):
        self.apply_ema()
        self.net.eval()

        pbar = tqdm(val_loader, total=len(val_loader), smoothing=0.01)
        step = 0
        loss_sum = 0.
        dloss_sum = 0.
        for val_batch in pbar:
            loss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=0,
                                stage='val',
                                write=False)
            loss_sum += loss
            if len(self.optimizer) > 1:
                loss = self.loss_fn(net=self.net,
                                    batch=val_batch,
                                    epoch=epoch,
                                    step=step,
                                    opt_idx=1,
                                    stage='val',
                                    write=False)
                dloss_sum += loss
            step += 1
        average_loss = loss_sum / step
        self.writer.add_scalar(f'val_epoch/loss', average_loss, epoch)
        if len(self.optimizer) > 1:
            average_dloss = dloss_sum / step
            self.writer.add_scalar(f'val_dloss_epoch/loss', average_dloss, epoch)
        self.restore_ema()
        return average_loss

    @torch.no_grad()
    def sample_step(self, train_batch, val_batch):
        self.apply_ema()
        self.net.eval()
        sample_path = make_dir(os.path.join(self.config.result.image_path, str(self.global_step)))
        if self.config.training.use_DDP:
            self.sample(self.net.module, train_batch, sample_path, stage='train')
            self.sample(self.net.module, val_batch, sample_path, stage='val')
        else:
            self.sample(self.net, train_batch, sample_path, stage='train')
            self.sample(self.net, val_batch, sample_path, stage='val')
        self.restore_ema()

    # abstract methods
    @abstractmethod
    def print_model_summary(self, net):
        pass

    @abstractmethod
    def initialize_model(self, config):
        pass

    @abstractmethod
    def initialize_optimizer_scheduler(self, net, config):
        pass

    @abstractmethod
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        pass

    @abstractmethod
    def sample(self, net, batch, sample_path, stage='train'):
        pass

    @abstractmethod
    def sample_to_eval(self, net, test_loader, sample_path):
        pass

    def on_save_checkpoint(self, net, train_loader, val_loader, epoch, step):
        pass


    @torch.no_grad()
    def test(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        if test_dataset is None:
            test_dataset = val_dataset
        # test_dataset = val_dataset
        if self.config.training.use_DDP:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     drop_last=False,
                                     sampler=test_sampler)
        else:
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     drop_last=True)

        if self.use_ema:
            self.apply_ema()

        self.net.eval()
        if self.config.args.sample_to_eval:
            sample_path = self.config.result.sample_to_eval_path
            if self.config.training.use_DDP:
                self.sample_to_eval(self.net.module, test_loader, sample_path)
            else:
                self.sample_to_eval(self.net, test_loader, sample_path)
        else:
            test_iter = iter(test_loader)
            for i in tqdm(range(1), initial=0, dynamic_ncols=True, smoothing=0.01):
                test_batch = next(test_iter)
                sample_path = os.path.join(self.config.result.sample_path, str(i))
                if self.config.training.use_DDP:
                    self.sample(self.net.module, test_batch, sample_path, stage='test')
                else:
                    self.sample(self.net, test_batch, sample_path, stage='test')

