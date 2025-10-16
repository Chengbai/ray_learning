import torch

from dataclasses import dataclass


@dataclass
class Config:
    # Training settings
    epochs: int = 5

    # Optimizer settings
    lr: float = 0.001
    momentum: float = 0.9

    # device used for model training and inference
    device: torch.device = torch.device("cpu")
