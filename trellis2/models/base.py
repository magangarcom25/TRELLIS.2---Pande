import torch
import torch.nn as nn
from abc import abstractmethod

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Wajib diimplementasikan oleh subclass"""
        pass

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')
