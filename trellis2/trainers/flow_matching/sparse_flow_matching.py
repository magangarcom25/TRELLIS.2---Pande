from typing import *
import os
import copy
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin, MultiImageConditionedMixin

class SparseFlowMatchingTrainer(FlowMatchingTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Mapping otomatis agar tidak KeyError pada trainer
        if 'denoiser' not in self.models and 'slat' in self.models:
            self.models['denoiser'] = self.models['slat']
            
        if 'denoiser' not in self.training_models and 'slat' in self.training_models:
            self.training_models['denoiser'] = self.training_models['slat']

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        # --- STRATEGI ANTI-BFLOAT16: NUCLEAR OPTION ---
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        with torch.autocast(device_type='cuda', enabled=False):
            # 1. Pastikan seluruh bobot model dalam Float32
            self.training_models['denoiser'].float()
            
            # 2. Siapkan input x_0 ke Float32
            x_0 = x_0.replace(x_0.feats.to(torch.float32))
            
            # 3. Generate Noise & Time Step
            noise_feats = torch.randn_like(x_0.feats).to(torch.float32)
            noise = x_0.replace(noise_feats)
            # Ambil t untuk batch size asli (N=1)
            t = self.sample_t(x_0.shape[0]).to(x_0.device).to(torch.float32)
            
            # 4. Kalkulasi Diffusion (x_t)
            x_t = self.diffuse(x_0, t, noise=noise)
            x_t = x_t.replace(x_t.feats.to(torch.float32))
            
            # 5. Handling Kondisi (DINOv2)
            cond = self.get_cond(cond, **kwargs)

            # --- FORCE SYNC BATCH FOR CFG ---
            # Jika cond memiliki 2 batch (asli + null) tapi x_t cuma 1
            # Kita paksa duplikasi di sini agar atensi tidak crash 1, 2, 2
            target_batch = cond.shape[0] if isinstance(cond, torch.Tensor) else x_0.shape[0] * 2
            
            if x_t.shape[0] == 1 and target_batch == 2:
                # Duplikasi x_t (SparseTensor) menjadi 2 batch
                x_t_input = sp.sparse_cat([x_t, x_t])
                # Duplikasi t menjadi 2 batch
                t_input = torch.cat([t, t], dim=0)
            else:
                x_t_input = x_t
                t_input = t
            
            # 6. Forward Pass Denoiser
            pred_raw = self.training_models['denoiser'](
                x_t_input, 
                (t_input * 1000).to(torch.float32), 
                cond, 
                **kwargs
            )

            # Jika tadi kita duplikasi untuk CFG, kita ambil setengah hasilnya 
            # (chunk pertama) untuk dihitung loss-nya terhadap target asli
            if x_t.shape[0] == 1 and target_batch == 2:
                # Ambil fitur setengah pertama
                half_feats = pred_raw.feats.shape[0] // 2
                pred = pred_raw.replace(pred_raw.feats[:half_feats])
            else:
                pred = pred_raw
            
            # 7. Ground Truth Velocity
            target = self.get_v(x_0, noise, t)
            target_feats = target.feats.to(torch.float32)
            
            # 8. Hitung Loss
            terms = edict()
            terms["mse"] = F.mse_loss(pred.feats.to(torch.float32), target_feats)
            terms["loss"] = terms["mse"]

            # Monitor Loss per objek dalam batch
            try:
                mse_per_instance = np.array([
                    F.mse_loss(pred.feats[x_0.layout[i]].to(torch.float32), target_feats[x_0.layout[i]]).item()
                    for i in range(x_0.shape[0])
                ])
                
                t_np = t.detach().cpu().numpy()
                time_bin = np.digitize(t_np, np.linspace(0, 1, 11)) - 1
                for i in range(10):
                    if (time_bin == i).sum() != 0:
                        terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}
            except:
                pass # Hindari crash hanya karena monitoring loss

            return terms, {}

    @torch.no_grad()
    def run_snapshot(self, num_samples: int, batch_size: int, verbose: bool = False) -> Dict:
        with torch.autocast(device_type='cuda', enabled=False):
            self.models['denoiser'].float()
            
            dataloader = DataLoader(
                copy.deepcopy(self.dataset),
                batch_size=num_samples,
                shuffle=True,
                num_workers=0,
                collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            )
            data = next(iter(dataloader))
            sampler = self.get_sampler()
            sample = []
            cond_vis = []
            
            for i in range(0, num_samples, batch_size):
                batch_data = {k: v[i:i+batch_size] for k, v in data.items()}
                batch_data = recursive_to_device(batch_data, 'cuda')
                
                x_0_snap = batch_data['x_0']
                x_0_snap = x_0_snap.replace(x_0_snap.feats.to(torch.float32))
                
                noise = x_0_snap.replace(torch.randn_like(x_0_snap.feats).to(torch.float32))
                cond_vis.append(self.vis_cond(**batch_data))
                del batch_data['x_0']
                
                args = self.get_inference_cond(**batch_data)
                args = {k: v.to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in args.items()}
                
                res = sampler.sample(
                    self.models['denoiser'],
                    noise=noise,
                    **args,
                    steps=12, guidance_strength=3.0, verbose=verbose,
                )
                sample.append(res.samples)
            
            sample = sp.sparse_cat(sample)
            sample_gt = {k: v for k, v in data.items()}
            sample = {k: v if k != 'x_0' else sample for k, v in data.items()}
            
            sample_dict = {
                'sample_gt': {'value': sample_gt, 'type': 'sample'},
                'sample': {'value': sample, 'type': 'sample'},
            }
            sample_dict.update(dict_reduce(cond_vis, None, {
                'value': lambda x: torch.cat(x, dim=0),
                'type': lambda x: x[0],
            }))
            
            return sample_dict

class SparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer): pass
class TextConditionedSparseFlowMatchingCFGTrainer(TextConditionedMixin, SparseFlowMatchingCFGTrainer): pass
class ImageConditionedSparseFlowMatchingCFGTrainer(ImageConditionedMixin, SparseFlowMatchingCFGTrainer): pass
class MultiImageConditionedSparseFlowMatchingCFGTrainer(MultiImageConditionedMixin, SparseFlowMatchingCFGTrainer): pass