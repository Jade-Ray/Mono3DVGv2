"""Functions that handle saving and loading of checkpoints."""
import shutil
from pathlib import Path


class CustomCheckpoint(object):
    """Custom checkpoint data to be saved by the accelerator. Include the epoch, best result, and best epoch."""
    def __init__(self):
        self._epoch = 0
        self._best_result = 0.
        self._best_epoch = 0
    
    @property
    def epoch(self):
        """The current epoch."""
        return self._epoch
    
    @epoch.setter
    def epoch(self, value):
        if not isinstance(value, int) and value < 0:
            raise ValueError("The epoch must be a positive integer.")
        self._epoch = value
    
    @property
    def best_result(self):
        """The best result in evluation."""
        return self._best_result
    
    @best_result.setter
    def best_result(self, value):
        if not isinstance(value, (int, float)) and value >= self._best_result:
            raise ValueError("The best result must be a number less than the previous best result.")
        self._best_result = float(value)
    
    @property
    def best_epoch(self):
        """The epoch where the best result was achieved."""
        return self._best_epoch
    
    @best_epoch.setter
    def best_epoch(self, value):
        if not isinstance(value, int) and value < 0:
            raise ValueError("The best epoch must be a positive integer.")
        self._best_epoch = value
    
    def state_dict(self):
        return {"epoch": self.epoch, "best_result": self.best_result, "best_epoch": self.best_epoch}
    
    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.best_result = state_dict["best_result"]
        self.best_epoch = state_dict["best_epoch"]


def get_resume_chekpoint_path(resume_chekpoint_path: str = None, output_path: str = None):
    if resume_chekpoint_path is not None or resume_chekpoint_path != "":
        path = Path(resume_chekpoint_path)
    else:
        path = get_last_checkpoint(output_path)
    
    assert path.exists(), f"Checkpoint file not found: {path}"
    return path


def make_checkpoint_dir(path_to_job, is_master_proc=True):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    checkpoint_dir = Path(path_to_job) / "checkpoints"
    # Create the checkpoint dir from the master process
    if is_master_proc:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_checkpoint_dir(path_to_job) -> Path:
    """
    Get path for storing checkpoints.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return Path(path_to_job) / "checkpoints"


def get_path_to_checkpoint(path_to_job, epoch, task=""):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        epoch (int): the number of epoch for the checkpoint.
    """
    if task != "":
        name = "{}_checkpoint_epoch_{:05d}.pyth".format(task, epoch)
    else:
        name = "checkpoint_epoch_{:05d}.pyth".format(epoch)
    return get_checkpoint_dir(path_to_job) / name


def get_last_checkpoint(path_to_job, task=""):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """

    d = get_checkpoint_dir(path_to_job)
    names = [_.name for _ in d.iterdir()] if d.exists() else []
    if 'latest' in names:
        return d / 'latest'
    if task != "":
        names = [f for f in names if "{}_checkpoint".format(task) in f]
    else:
        names = [f for f in names if f.startswith("checkpoint")]
    if len(names) == 0:
        return None
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return d / name


def get_checkpoint_epoch(path_to_job: Path):
    return int(path_to_job.stem.split("_")[-1])


def has_checkpoint(path_to_job):
    """
    Determines if the given directory contains a checkpoint.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = get_checkpoint_dir(path_to_job)
    files = list(map(str, d.iterdir())) if d.exists() else []
    return any("checkpoint" in f for f in files)


def limit_checkpoints_number(path_to_job, checkpoints_total_limit, task="", logger=None):
    
    d = get_checkpoint_dir(path_to_job)
    checkpoints = [_ for _ in d.iterdir()] if d.exists() else []
    if task != "":
        checkpoints = [f for f in checkpoints if "{}_checkpoint".format(task) in f]
    else:
        checkpoints = [f for f in checkpoints if f.stem.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda f: get_checkpoint_epoch(f))
    
    if len(checkpoints) >= checkpoints_total_limit:
        num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
        removing_checkpoints = checkpoints[:num_to_remove]
        
        logger.info(f"{len(checkpoints)} checkpoints already exist, removing {num_to_remove} checkpoints")
        logger.info(f"removing cheeckpoints: {', '.join([f.name for f in removing_checkpoints])}")
        
        for removing_checkpoint in removing_checkpoints:
            shutil.rmtree(removing_checkpoint)

