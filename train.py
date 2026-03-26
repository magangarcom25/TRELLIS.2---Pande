import os
import json
import argparse
import torch
import deepspeed
from easydict import EasyDict as edict
from safetensors.torch import load_file
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

    with open(opt.config, 'r') as f:
        cfg = edict(json.load(f))
    
    os.makedirs(opt.output_dir, exist_ok=True)
    deepspeed.init_distributed()
    dtype = torch.bfloat16 

    model_dinov2 = DinoV2FeatureExtractor("dinov2_vitb14").to(dtype).cuda()
    model_dinov2.eval()
    
    model_slat = SparseStructureFlowModel(
        resolution=32, in_channels=32, model_channels=1024, 
        cond_channels=768, out_channels=32, num_blocks=12, num_heads=16
    ).to(dtype).cuda()

    if os.path.exists("ckpts/slat_flow_img2shape_dit_600M_ms1024.safetensors"):
        print(f"[*] Loading Ganesha 600M Checkpoint...")
        state_dict = load_file("ckpts/slat_flow_img2shape_dit_600M_ms1024.safetensors")
        model_slat.load_state_dict(state_dict, strict=False)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=opt, model=model_slat, model_parameters=model_slat.parameters()
    )

    # MONKEY PATCH: Agar DeepSpeed selalu mengembalikan Tensor Tunggal ke Trainer
    orig_forward = model_engine.forward
    def patched_forward(*args, **kwargs):
        outputs = orig_forward(*args, **kwargs)
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    model_engine.forward = patched_forward

    dataset = getattr(datasets, cfg.dataset.name)(**cfg.dataset.args)
    trainer = getattr(trainers, cfg.trainer.name)(
        models={'denoiser': model_engine, 'image_cond_model': model_dinov2},
        image_cond_model=model_dinov2, 
        output_dir=opt.output_dir,
        dataset=dataset, load_dir=None, step=0, **cfg.trainer.args
    )
    
    print("\n" + "="*50 + "\n   GANESHA 600M: STARTING TRAINING (SUCCESS)\n" + "="*50)
    trainer.run()

if __name__ == '__main__':
    main()