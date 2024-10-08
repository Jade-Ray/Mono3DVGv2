import math
import datetime
from pathlib import Path

from tqdm.auto import tqdm

from utils.logging import get_file_handler
from utils.parser import parse_args, load_config
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.accelerator_helper import build_accelerator, init_accelerator, get_mp_logger as get_logger
from lib.helpers.checkpoint_helper import CustomCheckpoint, get_resume_chekpoint_path
from lib.helpers.metric_helper import evaluation
from lib.models.configuration_mono3dvg_v2 import Mono3DVGv2Config
from lib.models.mono3dvg_v2 import Mono3DVGv2ForSingleObjectDetection as Mono3DVG
from lib.models.image_processsing_mono3dvg import Mono3DVGImageProcessor


logger = get_logger(__name__)


def main():
    args = parse_args()
    cfg = load_config(args, args.cfg_file)
    cfg.with_tracking = False
    
    accelerator = build_accelerator(cfg)
    # Handle the hugingface hub repo creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)    
    accelerator.wait_for_everyone()
    
    logger.logger.addHandler(get_file_handler(Path(cfg.output_dir) / f'test.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log'))
    
    logger.info(f'Init accelerator...\n{accelerator.state}', main_process_only=False)
    
    logger.info("Init DataLoader...")
    _, _, test_dataloader, id2label, label2id = build_dataloader(
        cfg, workers=cfg.dataloader_num_workers, accelerator=accelerator
    )
    
    logger.info("Init Model...")
    config = Mono3DVGv2Config(
        label2id=label2id, id2label=id2label, **vars(cfg.model)
    )
    model = Mono3DVG(config)
    image_processor = Mono3DVGImageProcessor()
    
    
    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )
    
    # We need to recalculate our total test steps as the size of the testing dataloader may have changed.
    cfg.max_test_steps = math.ceil(len(test_dataloader) / cfg.gradient_accumulation_steps)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    init_accelerator(accelerator, cfg)
    
    # ------------------------------------------------------------------------------------------------
    # Run testing
    # ------------------------------------------------------------------------------------------------

    total_batch_size = cfg.test_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("##################  Running testing  ##################")
    logger.info(f"  Num examples = {len(test_dataloader.dataset)}")
    logger.info(f"  Instantaneous batch size per device = {cfg.test_batch_size}")
    logger.info(f"  Total test batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total testing steps = {cfg.max_test_steps}")

    # custom checkpoint data registe to accelerator
    extra_state = CustomCheckpoint()
    
    # Potentially load in the weights and states from a previous save
    if cfg.pretrain_model:
        checkpoint_path = get_resume_chekpoint_path(cfg.pretrain_model, cfg.output_dir)
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.register_for_checkpointing(extra_state)
        accelerator.load_state(checkpoint_path)
        logger.info(f"Loading Checkpoint... Best Result:{extra_state.best_result}, Best Epoch:{extra_state.best_epoch}")
    
    logger.info("***** Running evaluation *****")
    metrics = evaluation(model, image_processor, accelerator, test_dataloader, 
                         logger=logger, only_overall=False)
    for split_name in ['Overall', 'Unique', 'Multiple', 'Near', 'Medium', 'Far', 'Easy', 'Moderate', 'Hard']:
        logger.info(f"------------{split_name}------------")
        msg = (
            f'Accuracy@0.25: {metrics[f"{split_name}_Acc@0.25"]:.2f}%\t'
            f'Accuracy@0.5: {metrics[f"{split_name}_Acc@0.5"]:.2f}%\t'
            f'Mean IoU: {metrics[f"{split_name}_MeanIoU"]:.2f}%\t'
        )
        logger.info(msg)


if __name__ == '__main__':
    main()
