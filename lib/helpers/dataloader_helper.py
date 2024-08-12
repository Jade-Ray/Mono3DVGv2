
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator

from lib.datasets.mono3drefer import build_dataset


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers: int=0, accelerator: Accelerator=None):
    train_dataset, valid_dataset, test_dataset, id2label, label2id = build_dataset(cfg.dataset, accelerator)
    print('train, val, test: ',len(train_dataset), len(valid_dataset), len(test_dataset))
    
    dataloader_common_args = {
        "num_workers": workers,
        "worker_init_fn": my_worker_init_fn,
    }
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=cfg.train_batch_size, 
        **dataloader_common_args
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        shuffle=False, 
        batch_size=cfg.valid_batch_size,  
        **dataloader_common_args
    )
    test_dataloader = DataLoader(
        test_dataset, 
        shuffle=False, 
        batch_size=cfg.test_batch_size, 
        **dataloader_common_args
    )
    
    return train_dataloader, valid_dataloader, test_dataloader, id2label, label2id