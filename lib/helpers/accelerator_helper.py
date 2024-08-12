import logging
from datetime import timedelta
from argparse import Namespace

import datasets
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import MultiProcessAdapter
from accelerate.utils import ProjectConfiguration, set_seed, is_tensorboard_available, is_wandb_available

from utils.logging import get_logger
from utils.parser import namespace_to_dict


def get_mp_logger(name: str, log_level: str = 'info', log_format: str = None):
    """
    Returns a `logging.LoggerAdapter` for `name` that can handle multiprocessing.
    
    See more details in the `accelerate.logging` module.
    """
    
    return MultiProcessAdapter(get_logger(name, log_level, log_format), {})


def build_accelerator(cfg: Namespace) -> Accelerator:
    """
    Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers in the environment
    """
    accelerator_log_kwargs = {}
    project_config_kwargs = {"project_dir": cfg.output_dir}
    
    if cfg.with_tracking:
        if cfg.report_to == "tensorboard":
            if not is_tensorboard_available():
                raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")
        elif cfg.report_to == "wandb":
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        accelerator_log_kwargs["log_with"] = cfg.report_to
        project_config = ProjectConfiguration()
        project_config.set_directories(cfg.output_dir) # Set the project directory and logging directory to the output_dir
        accelerator_log_kwargs["project_config"] = project_config
    
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
        ],
        **accelerator_log_kwargs,
    )
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    # If passed along, set the training seed now.
    # We set device_specific to True as we want different data augmentation per device.
    if cfg.seed is not None:
        set_seed(cfg.seed, device_specific=True)
    
    return accelerator


def init_accelerator(accelerator: Accelerator, cfg:Namespace):
    if cfg.with_tracking:
        accelerator.init_trackers(
            project_name=cfg.project_name, 
            config=namespace_to_dict(cfg),
        )

