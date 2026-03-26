import os
import sys
import json
import argparse
import torch
import torch.distributed as dist
import deepspeed
from easydict import EasyDict as edict
from safetensors.torch import load_file
from deepspeed.ops.adam import DeepSpeedCPUAdam

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
from feature_extractors import DinoV2FeatureExtractor
from trellis2 import datasets, trainers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    opt = parser.parse_args()

    # 0. Load Config
    with open(opt.config, 'r') as f:
        cfg = edict(json.load(f))
    
    os.makedirs(opt.output_dir, exist_ok=True)
    if not dist.is_initialized():
        deepspeed.init_distributed()
    
    dtype = torch.float32 
    target_steps = cfg.trainer.args.get('max_steps', 1000)

    # 1. DinoV2 Setup
    print(f"[*] Loading DINOv2 Feature Extractor...")
    model_dinov2 = DinoV2FeatureExtractor("dinov2_vitb14").to(dtype).cuda()
    model_dinov2.eval()
    for param in model_dinov2.parameters():
        param.requires_grad = False
    
    # 2. Ganesha Model Setup
    model_slat = SparseStructureFlowModel(
        resolution=32, 
        in_channels=8, 
        model_channels=128, 
        out_channels=32, 
        num_res_blocks=12
    ).to(dtype).cuda()

    # Load Weights
    ckpt_path = "ckpts/slat_flow_img2shape_dit_600M_ms1024.safetensors"
    if os.path.exists(ckpt_path):
        print(f"[*] Loading Weights from {ckpt_path}...")
        try:
            state_dict = load_file(ckpt_path)
            model_slat.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[!] Info: {e}. Output layers will be fine-tuned.")

    # 3. Optimized Optimizer for CPU Offload
    opt_cfg = cfg.trainer.args.get('optimizer', {'args': {'lr': 5e-5}})
    trainable_params = [p for p in model_slat.parameters() if p.requires_grad]
    
    # DeepSpeedCPUAdam wajib digunakan jika memakai Zero-Offload
    optimizer_obj = DeepSpeedCPUAdam(
        trainable_params, 
        lr=opt_cfg.get('args', {}).get('lr', 5e-5),
        betas=opt_cfg.get('args', {}).get('betas', (0.9, 0.999)),
        eps=opt_cfg.get('args', {}).get('eps', 1e-8),
        weight_decay=opt_cfg.get('args', {}).get('weight_decay', 0.01)
    )

    print(f"[*] Initializing DeepSpeed Stage 2 with Optimized CPU Adam...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=opt, 
        model=model_slat, 
        model_parameters=trainable_params,
        optimizer=optimizer_obj
    )

    # Patch Forward
    orig_forward = model_engine.forward
    def patched_forward(*args, **kwargs):
        outputs = orig_forward(*args, **kwargs)
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    model_engine.forward = patched_forward

    # 4. Trainer Setup
    dataset = getattr(datasets, cfg.dataset.name)(**cfg.dataset.args)
    models_dict = {'denoiser': model_engine, 'image_cond_model': model_dinov2}
    
    cfg.trainer.args['max_steps'] = target_steps

    trainer = getattr(trainers, cfg.trainer.name)(
        models=models_dict,
        image_cond_model=model_dinov2,
        output_dir=opt.output_dir,
        dataset=dataset, 
        load_dir=None, 
        step=0, 
        **cfg.trainer.args
    )
    
    trainer.max_steps = target_steps

    if dist.get_rank() == 0:
        print("\n" + "="*60)
        print(f"   TRELLIS 2: RECONSTRUCTION STARTING")
        print(f"   Mode: DeepSpeed ZeRO Stage 2 + CPU Adam")
        print(f"   Target: {target_steps} steps")
        print("="*60 + "\n")
    
    # 5. Run
    trainer.run()

if __name__ == '__main__':
    main()