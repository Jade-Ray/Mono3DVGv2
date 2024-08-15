from .mono3drefer import build_dataset as build_dataset_from_hub
from .mono3drefer_dataset import Mono3DReferDataset

def build_dataset(cfg):
    
    if hasattr(cfg, "root_dir"):
        train_dataset = Mono3DReferDataset('train', cfg)
        val_dataset = Mono3DReferDataset('val', cfg)
        test_dataset = Mono3DReferDataset('test', cfg)
        return train_dataset, val_dataset, test_dataset, train_dataset.id2label, train_dataset.label2id
    elif hasattr(cfg, "name"):
        return build_dataset_from_hub(cfg)
    
    raise ValueError("You must provide either a root_dir or a name in the config to build the dataset.")
    