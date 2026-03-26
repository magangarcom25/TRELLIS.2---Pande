from abc import abstractmethod
import os
import time
import json
import copy
import threading
import functools
from functools import partial
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from torchvision import utils
from torch.utils.tensorboard import SummaryWriter

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler
from ..utils.dist_utils import *
from ..utils import grad_clip_utils

class BasicTrainer:
    def __init__(self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=1,
        optimizer={},
        lr_scheduler=None,
        grad_clip=None,
        ema_rate=0.9999,
        i_print=1,
        i_log=500,
        i_save=500,
        **kwargs
    ):
        # --- 1. SETUP ATRIBUT DASAR ---
        # Nama 'training_models' harus digunakan agar subclass flow_matching tidak error
        self.training_models = models 
        self.models = models # Aliasing untuk keamanan
        self.dataset = dataset
        self.max_steps = max_steps
        self.batch_split = batch_split  
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.grad_clip = grad_clip
        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_save = i_save
        
        if ema_rate is None or (isinstance(ema_rate, list) and len(ema_rate) == 0):
            self.ema_rate = []
        else:
            self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.is_master = self.rank == 0
        else:
            self.world_size = 1
            self.rank = 0
            self.is_master = True

        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size

        # --- 2. JALANKAN INISIALISASI ---
        self.init_models_and_more(**kwargs)
        
        # Panggil prepare_dataloader SETELAH semua atribut (batch_split, models) siap
        self.prepare_dataloader()
        
        self.step = 0
        if load_dir is not None and step is not None:
            self.load(load_dir, step)
        
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, 'tb_logs'))

    def init_models_and_more(self, **kwargs):
        # Ambil parameter yang membutuhkan gradien dari training_models
        self.model_params = sum([[p for p in model.parameters() if p.requires_grad] 
                                for model in self.training_models.values()], [])
        
        opt_cfg = self.optimizer_config
        self.optimizer = getattr(torch.optim, opt_cfg.get('name', 'AdamW'))(self.model_params, **opt_cfg.get('args', {}))
        
        if self.lr_scheduler_config:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(self.optimizer, **self.lr_scheduler_config['args'])
        else:
            self.lr_scheduler = None

        if self.is_master and self.ema_rate:
            self.ema_params = [copy.deepcopy(self.model_params) for _ in self.ema_rate]
        else:
            self.ema_params = []

    def prepare_dataloader(self):
        # Default dataloader jika tidak di-override subclass
        self.data_sampler = ResumableSampler(self.dataset, shuffle=True)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size_per_gpu, 
            num_workers=0, 
            pin_memory=True, 
            drop_last=True, 
            sampler=self.data_sampler
        )
        self.data_iterator = cycle(self.dataloader)

    def load(self, load_dir, step):
        path = os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu')
            if 'optimizer' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            self.step = ckpt.get('step', 0)

    def update_ema(self):
        if not self.is_master or not self.ema_params: return
        for i, rate in enumerate(self.ema_rate):
            for master, ema in zip(self.model_params, self.ema_params[i]):
                ema.detach().mul_(rate).add_(master, alpha=1.0 - rate)

    def save(self):
        if not self.is_master: return
        torch.save({'optimizer': self.optimizer.state_dict(), 'step': self.step}, 
                   os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt'))
        for name, model in self.training_models.items():
            torch.save(model.state_dict(), os.path.join(self.output_dir, 'ckpts', f'{name}_step{self.step:07d}.pt'))

    @abstractmethod
    def training_losses(self, **mb_data):
        pass

    def run(self):
        """DEEPSPEED ZeRO-3 STABLE RUN"""
        print(f"[*] Training Active. Target: {self.max_steps} steps.")
        
        # DeepSpeed engine biasanya adalah denoiser
        engine = self.training_models['denoiser']
        device = next(engine.parameters()).device
        
        while self.step < self.max_steps:
            start_time = time.time()
            self.step += 1
            
            # Ambil Batch
            batch = next(self.data_iterator)
            if isinstance(batch, list): batch = batch[0]
            batch = recursive_to_device(batch, device)

            # Forward Pass
            raw_loss = self.training_losses(**batch)
            loss = raw_loss[0] if isinstance(raw_loss, (tuple, list)) else raw_loss
            if isinstance(loss, dict): 
                loss = sum(l for l in loss.values() if isinstance(l, torch.Tensor))

            # DeepSpeed Engine handles Backward & Step
            engine.backward(loss)
            engine.step() 
            
            self.update_ema()

            # Terminal Log
            if self.step % self.i_print == 0:
                print(f"[Step {self.step:05d}] Loss: {loss.item():.4f} | Time: {time.time()-start_time:.2f}s", flush=True)

            if self.step % self.i_save == 0:
                self.save()

            if self.is_master and self.step % self.i_log == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.step)
            
            if self.step % 100 == 0:
                torch.cuda.empty_cache()

        print("[*] Training completed.")