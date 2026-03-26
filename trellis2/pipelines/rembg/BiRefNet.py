import torch
import torch.nn as nn
from transformers import AutoModelForImageSegmentation

class BiRefNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Kita tentukan langsung nama modelnya agar tidak ada lagi 'missing argument'
        model_name = "briaai/RMBG-2.0"
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, 
            low_cpu_mem_usage=False, 
            trust_remote_code=True
        )

    def forward(self, x):
        return self.model(x)
