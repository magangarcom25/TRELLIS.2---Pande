import torch.nn as nn
from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import json

from ....utils import dist_utils

class DinoV2FeatureExtractor(nn.Module):
    """
    Feature extractor for DINOv2 models.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        # Load model dari Torch Hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def forward(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        return self.__call__(image)

    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)
        elif isinstance(image, list):
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            # Ambil device dari parameter model
            device = next(self.parameters()).device
            image = torch.stack(image).to(device)
        
        image = self.transform(image)
        # Ambil fitur prenorm (patch tokens)
        features = self.model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens

class DinoV3FeatureExtractor(nn.Module):
    """
    Feature extractor for DINOv3 models - Manual Offline Version.
    """
    def __init__(self, model_name: str, image_size=512):
        super().__init__()
        from transformers import DINOv3ViTModel, DINOv3ViTConfig
        from safetensors.torch import load_file
        
        self.model_name = model_name
        self.image_size = image_size
        
        # Baca config secara manual
        config_path = os.path.join(model_name, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = DINOv3ViTConfig.from_dict(config_dict)
        
        # Load model kosong & suntik bobot safetensors
        self.model = DINOv3ViTModel(config)
        weight_path = os.path.join(model_name, "model.safetensors")
        if os.path.exists(weight_path):
            state_dict = load_file(weight_path)
            self.model.load_state_dict(state_dict)
        
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        # Gunakan dtype dari parameter model
        dtype = next(self.parameters()).dtype
        image = image.to(dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for layer_module in self.model.layer:
            hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
        
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        if isinstance(image, list):
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            device = next(self.parameters()).device
            image = torch.stack(image).to(device)
        
        image = self.transform(image)
        return self.extract_features(image)

class ImageConditionedMixin:
    def __init__(self, *args, image_cond_model: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_config = image_cond_model
        # Inisialisasi awal agar tidak AttributeError
        self.image_cond_model = None

    def _init_image_cond_model(self):
        # Cek apakah model sudah dimuat oleh train.py di self.models
        if hasattr(self, 'models') and 'image_cond_model' in self.models:
            self.image_cond_model = self.models['image_cond_model']
            return

        model = getattr(self, 'image_cond_model', None)
        if model is None or isinstance(model, dict):
            if isinstance(self.image_cond_model_config, dict):
                name = self.image_cond_model_config['name']
                args = self.image_cond_model_config.get('args', {})
                if name in globals():
                    self.image_cond_model = globals()[name](**args)
                    self.image_cond_model.cuda()
                    print(f"[*] {name} initialized via Mixin.")
                else:
                    raise AttributeError(f"Model class '{name}' not found.")

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        model = getattr(self, 'image_cond_model', None)
        if model is None or isinstance(model, dict):
            self._init_image_cond_model()
        return self.image_cond_model(image)
        
    def get_cond(self, cond, **kwargs):
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        return super().get_cond(cond, **kwargs)

    def get_inference_cond(self, cond, **kwargs):
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        return super().get_inference_cond(cond, **kwargs)

    def vis_cond(self, cond, **kwargs):
        return {'image': {'value': cond, 'type': 'image'}}

class MultiImageConditionedMixin:
    def __init__(self, *args, image_cond_model: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_config = image_cond_model
        # Inisialisasi awal agar tidak AttributeError
        self.image_cond_model = None
        
    def _init_image_cond_model(self):
        if hasattr(self, 'models') and 'image_cond_model' in self.models:
            self.image_cond_model = self.models['image_cond_model']
            return

        model = getattr(self, 'image_cond_model', None)
        if model is None or isinstance(model, dict):
            if isinstance(self.image_cond_model_config, dict):
                name = self.image_cond_model_config['name']
                args = self.image_cond_model_config.get('args', {})
                if name in globals():
                    self.image_cond_model = globals()[name](**args)
                    self.image_cond_model.cuda()
                else:
                    raise AttributeError(f"Model class '{name}' not found.")
    
    @torch.no_grad()
    def encode_images(self, images: Union[List[torch.Tensor], List[List[Image.Image]]]) -> List[torch.Tensor]:
        model = getattr(self, 'image_cond_model', None)
        if model is None or isinstance(model, dict):
            self._init_image_cond_model()
            
        seqlen = [len(i) for i in images]
        images_flat = torch.cat(images, dim=0) if isinstance(images[0], torch.Tensor) else sum(images, [])
        features = self.image_cond_model(images_flat)
        features_split = torch.split(features, seqlen)
        return [f.reshape(-1, f.shape[-1]) for f in features_split]
        
    def get_cond(self, cond, **kwargs):
        cond = self.encode_images(cond)
        kwargs['neg_cond'] = [torch.zeros_like(c[:1, :]) for c in cond]
        return super().get_cond(cond, **kwargs)

    def get_inference_cond(self, cond, **kwargs):
        cond = self.encode_images(cond)
        kwargs['neg_cond'] = [torch.zeros_like(c[:1, :]) for c in cond]
        return super().get_inference_cond(cond, **kwargs)

    def vis_cond(self, cond, **kwargs):
        H, W = cond[0].shape[-2:]
        vis = []
        for images in cond:
            canvas = torch.zeros(3, H * 2, W * 2, device=images.device, dtype=images.dtype)
            for i, image in enumerate(images[:4]):
                kh, kw = i // 2, i % 2
                canvas[:, kh*H:(kh+1)*H, kw*W:(kw+1)*W] = image
            vis.append(canvas)
        return {'image': {'value': torch.stack(vis), 'type': 'image'}}