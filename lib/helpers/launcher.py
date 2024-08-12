import gc
import logging
import math
import os
from datetime import timedelta
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
import numpy as np

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from huggingface_hub import HfApi
import wandb
import transformers

from dataset import build_dataset
from models import summary
from pipelines.anchor_pipeline import get_anchor_pipeline
import utils.checkpoint as cu

logger = get_logger(__name__, log_level="info")


class Launcher(ABC):
    """Abstract class for launcher. 
    Should implemente the following methods: `_env_init`, `_model_init`, `_data_init` and `_logger_init`. 
    
    The default `__init__` order is `_env_init` -> `_logger_init` -> `_data_init` -> `_model_init`."""
    
    def __init__(self, cfg: Namespace):
        self.cfg = cfg
        self._env_init(cfg)
        self._logger_init(cfg)
        self._data_init(cfg)
        self._model_init(cfg)
    
    @property
    def output_dir(self) -> Path:
        return self._output_dir
    
    @output_dir.setter
    def output_dir(self, output_dir):
        if output_dir is not None:
            self._output_dir = Path(output_dir)
            self._output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _env_init(self, cfg: Namespace):
        """Set up environment."""
        raise NotImplementedError
    
    @abstractmethod
    def _model_init(self, cfg: Namespace):
        """Initialize model."""
        raise NotImplementedError
    
    @abstractmethod
    def _data_init(self, cfg: Namespace):
        """Initialize data."""
        raise NotImplementedError
    
    @abstractmethod
    def _logger_init(self, cfg: Namespace):
        """Initialize logger."""
        raise NotImplementedError
    
    def clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()


class AcceleratorLauncher(Launcher):
    """Auto set up accelerator in `_env_init`, but need to implement `_accelerator_init` in child class. 
    Also set up accelerator logger in `_logger_init` and support a method to save imgs to logger.
    
    The init order changes to `_env_init`(setted) -> `_logger_init`(setted) -> `_data_init` -> `_model_init` -> `_accelerator_init`."""

    def __init__(self, cfg: Namespace):
        super().__init__(cfg)
        self._accelerator_init(cfg)

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    def _env_init(self, cfg):
        self.output_dir = cfg.output_dir
        self.logging_dir = self.output_dir / cfg.logging_dir
        gradient_accumulation_steps = getattr(cfg, "gradient_accumulation_steps", 1)
        mixed_precision = getattr(cfg, "mixed_precision", None)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=cfg.logger,
            project_config=ProjectConfiguration(
                project_dir=self.output_dir,
                logging_dir=self.logging_dir,
            ),
            kwargs_handlers=[
                InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
            ]
        )
        
        # If passed along, set the training seed now.
        # We set device_specific to True as we want different data augmentation per device.
        if cfg.seed is not None:
            set_seed(cfg.seed, device_specific=True)
        
        if hasattr(cfg, 'hub'):
            self._huggingface_hub_init(cfg.hub)

    def _logger_init(self, cfg):
        if cfg.logger == "tensorboard":
            if not is_tensorboard_available():
                raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

        elif cfg.logger == "wandb":
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
    
    def _huggingface_hub_init(self, cfg):
        """Handle the repository creation."""
        if self.accelerator.is_main_process and cfg.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = cfg.hub_model_id
            if repo_name is None:
                repo_name = Path(self.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=cfg.hub_token).repo_id

            with open(os.path.join(self.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
    
    def _save_to_logger(self, imgs, title='',
                        epoch=None, global_step=None):
        """Save images to logger of accelerator."""
        if not isinstance(imgs, list):
            imgs = [imgs]
        if self.cfg.logger == 'tensorboard':
            tracker = self.accelerator.get_tracker("tensorboard", unwrap=True)
            tracker.add_images(title, imgs, 
                               global_step=epoch)
        elif self.cfg.logger == 'wandb':
            log_dict = {title: [wandb.Image(_) for _ in imgs]}
            if epoch is not None:
                log_dict['epoch'] = epoch 
            self.accelerator.get_tracker("wandb").log(
                log_dict,
                step=global_step,
            )
    
    def images_to_logger(self, images: list, title='', 
                         captions: list[str]=None, epoch=None):
        """Log images to accelerator logger."""
        if not isinstance(images, list):
            images = [images]
        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.array(image) for image in images])
                tracker.writer.add_images(title, np_images, epoch, dataformats="NHWC")
            elif tracker.name == "wandb":
                wandb_images = []
                for i, image in enumerate(images):
                    caption = captions[i] if captions is not None else f"{i}"
                    if isinstance(image, wandb.Image):
                        image._caption = caption
                    else:
                        image = wandb.Image(image, caption=caption)
                    wandb_images.append(image)   
                tracker.log({title: wandb_images,}, step=epoch)
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")
    
    def tabel_to_logger(self, columns: list, data: list[list], title='', epoch=None):
        """Log table to accelerator logger."""
        for tracker in self.accelerator.trackers:
            if tracker.name == "wandb":
                wandb_table = wandb.Table(columns=columns, data=data)
                tracker.log({title: wandb_table}, step=epoch)
            else:
                logger.warn(f"table logging not implemented for {tracker.name}")

    def _accelerator_init(self, cfg):
        # Prepare everything with our `accelerator`.
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        raise NotImplementedError
    
    def _unwrap_model(self, model):
        """Unwrapping if model was compiled with `torch.compile`."""
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    def clear_memory(self):
        """Clear memory and torch cache, then waiting for every tracker of accelerator to update."""
        gc.collect()
        torch.cuda.empty_cache()
        
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
