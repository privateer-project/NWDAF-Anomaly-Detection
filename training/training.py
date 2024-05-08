from tqdm import tqdm

import torch
import torch.optim as optim

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class TrainingHistory:
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    mal_losses: List[float] = field(default_factory=list)
    gradient_norms: dict = field(default_factory=lambda: defaultdict(list))
    model_weights: Dict[str, Any] = field(default_factory=dict)
    epochs_trained: int = 0
    
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print('Early stopping')
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0