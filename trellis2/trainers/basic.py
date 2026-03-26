from abc import abstractmethod
import os
import time
import json
import copy
import threading
from functools import partial
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict as edict

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler
from ..utils.dist_utils import *
from ..utils import grad_clip_utils, elastic_utils

class BasicTrainer:
    def __init__(self, models, dataset, *, output_dir, load_dir, step, max_steps, batch_size=None, batch_size_per_gpu=None, batch_split=None, optimizer={}, lr_scheduler=None, elastic=None, grad_clip=None, ema_rate=0.9999, fp16_mode=None, mix_precision_mode='inflat_all', mix_precision_dtype='float16', fp16_scale_growth=1e-3, parallel_mode='ddp', finetune_ckpt=None, log_param_stats=False, prefetch_data=True, snapshot_batch_size=4, i_print=1, i_log=500, i_sample=1000, i_save=500, i_ddpcheck=10000, **kwargs):
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'
        self.models, self.dataset, self.batch_split, self.max_steps = models, dataset, batch_split if batch_split is not None else 1, max_steps
        self.optimizer_config, self.lr_scheduler_config, self.grad_clip = optimizer, lr_scheduler, grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else (ema_rate if ema_rate else [])
        self.mix_precision_mode, self.mix_precision_dtype = mix_precision_mode, str_to_dtype(mix_precision_dtype)
        self.output_dir, self.i_print, self.i_log, self.i_sample, self.i_save = output_dir, i_print, i_log, i_sample, i_save
        
        if dist.is_initialized():
            self.world_size, self.rank = dist.get_world_size(), dist.get_rank()
            self.is_master = self.rank == 0
        else:
            self.world_size, self.rank, self.is_master = 1, 0, True
            
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)
        self.step = 0
        if load_dir is not None and step is not None: self.load(load_dir, step)
        
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, 'tb_logs'))

    def init_models_and_more(self, **kwargs):
        self.training_models = self.models
        self.model_params = sum([[p for p in model.parameters() if p.requires_grad] for model in self.models.values()], [])
        self.master_params = self.model_params
        self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate] if self.is_master and self.ema_rate else []
        opt_cfg = self.optimizer_config
        self.optimizer = getattr(torch.optim, opt_cfg.get('name', 'AdamW'))(self.master_params, **opt_cfg.get('args', {}))
        self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(self.optimizer, **self.lr_scheduler_config['args']) if self.lr_scheduler_config else None

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = ResumableSampler(self.dataset, shuffle=True)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size_per_gpu, num_workers=0, pin_memory=True, drop_last=True, sampler=self.data_sampler)
        self.data_iterator = cycle(self.dataloader)

    def load(self, load_dir, step):
        path = os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu')
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.step = ckpt['step']

    def update_ema(self):
        if not self.is_master or not self.ema_params: return
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(master_param, alpha=1.0 - ema_rate)

    def save(self):
        if not self.is_master: return
        misc_ckpt = {'optimizer': self.optimizer.state_dict(), 'step': self.step}
        torch.save(misc_ckpt, os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt'))
        for name, model in self.models.items(): 
            torch.save(model.state_dict(), os.path.join(self.output_dir, 'ckpts', f'{name}_step{self.step:07d}.pt'))

    @abstractmethod
    def training_losses(self, **mb_data): pass

    def run(self):
        print(f"[*] Training Loop Active. Target: {self.max_steps} steps.")
        model_val = next(iter(self.models.values()))
        device = next(model_val.parameters()).device
        
        while self.step < self.max_steps:
            start_time = time.time()
            # Ambil data batch
            batch = next(self.data_iterator)
            if isinstance(batch, list): batch = batch[0]
            batch = recursive_to_device(batch, device)
            
            # Reset Gradien (set_to_none=False penting untuk DeepSpeed ZeRO)
            self.optimizer.zero_grad(set_to_none=False)
            
            # Hitung Loss
            losses_out = self.training_losses(**batch)
            
            # --- EKSTRAKSI LOSS (Fix untuk TRELLIS Tuple Output) ---
            loss = None
            
            # 1. Bongkar Tuple jika output adalah (dict_loss, dict_extra)
            if isinstance(losses_out, (tuple, list)):
                actual_losses = losses_out[0]
            else:
                actual_losses = losses_out

            # 2. Cari tensor loss di dalam hasil bongkaran
            if isinstance(actual_losses, (dict, edict)):
                # Cari key prioritas: 'loss' atau 'mse'
                for key in ['loss', 'mse', 'loss_total', 'total_loss']:
                    if key in actual_losses:
                        loss = actual_losses[key]
                        break
                # Fallback: Ambil tensor pertama yang tersedia
                if loss is None:
                    loss = next((v for v in actual_losses.values() if torch.is_tensor(v)), None)
            elif torch.is_tensor(actual_losses):
                loss = actual_losses

            # Proteksi Loss Valid
            if loss is None or not torch.is_tensor(loss):
                print(f"[!] Warning Step {self.step}: Loss invalid. Skipping batch.")
                self.step += 1
                continue
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[!] Warning Step {self.step}: Loss is NaN/Inf. Skipping batch.")
                self.step += 1
                continue

            # --- BACKWARD DEEPSPEED ---
            # WAJIB menggunakan self.optimizer.backward(loss) jika menggunakan DeepSpeed
            if hasattr(self.optimizer, 'backward'): 
                self.optimizer.backward(loss)
            else: 
                loss.backward()
            
            # Gradient Clipping
            if self.grad_clip and not hasattr(self.optimizer, 'backward'):
                torch.nn.utils.clip_grad_norm_(self.master_params, self.grad_clip)
            
            # Optimizer Step
            self.optimizer.step()
            if self.lr_scheduler: self.lr_scheduler.step()
            
            self.step += 1
            self.update_ema()
            
            # Logging ke Terminal
            if self.step % self.i_print == 0:
                print(f"[Step {self.step:05d}] Loss: {loss.item():.4f} | Time: {time.time()-start_time:.2f}s", flush=True)
            
            # Autosave Checkpoint
            if self.step % self.i_save == 0: 
                self.save()
                
        print("[*] All training steps completed!")