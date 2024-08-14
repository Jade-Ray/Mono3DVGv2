import math
import datetime
from pathlib import Path

import torch
from tqdm.auto import tqdm

from utils.logging import get_file_handler
from utils.parser import parse_args, load_config
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.accelerator_helper import build_accelerator, init_accelerator, get_mp_logger as get_logger
from lib.helpers.schedule_helper import build_lr_scheduler
from lib.helpers.checkpoint_helper import CustomCheckpoint, get_resume_chekpoint_path, get_checkpoint_epoch, get_checkpoint_dir
from lib.helpers.metric_helper import evaluation
from lib.helpers.huggingface_hub_helper import create_huggingface_hub_repo, upload_output_folder
from lib.models.configuration_mono3dvg_v2 import Mono3DVGv2Config
from lib.models.mono3dvg_v2 import Mono3DVGv2ForSingleObjectDetection as Mono3DVG
from lib.models.image_processsing_mono3dvg import Mono3DVGImageProcessor


logger = get_logger(__name__)


def main():
    args = parse_args()
    cfg = load_config(args, args.cfg_file)
    
    accelerator = build_accelerator(cfg)
    # Handle the hugingface hub repo creation
    if accelerator.is_main_process:
        if cfg.push_to_hub:
            api, repo_id, hub_token = create_huggingface_hub_repo(cfg)
        elif cfg.output_dir is not None:
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)    
    accelerator.wait_for_everyone()
    
    logger.logger.addHandler(get_file_handler(Path(cfg.output_dir) / f'train.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log'))
    
    logger.info(f'Init accelerator...\n{accelerator.state}', main_process_only=False)
    
    logger.info("Init DataLoader...")
    train_dataloader, valid_dataloader, _, id2label, label2id = build_dataloader(
        cfg, workers=cfg.dataloader_num_workers, accelerator=accelerator
    )
    
    logger.info("Init Model...")
    config = Mono3DVGv2Config(
        label2id=label2id, id2label=id2label, **vars(cfg.model)
    )
    if hasattr(cfg, 'mono3dvg_model') and cfg.mono3dvg_model is not None:
        model = Mono3DVG._load_mono3dvg_pretrained_model(cfg.mono3dvg_model, config, logger=logger)
    else:
        model = Mono3DVG(config)
    image_processor = Mono3DVGImageProcessor()
    
    # Optimizer
    weights, biases = [], []
    for name, param in model.named_parameters():
        if 'bias' in name:
            biases += [param]
        else:
            weights += [param]
    optimizer = torch.optim.AdamW(
        [{'params': biases, 'weight_decay': 0}, 
         {'params': weights, 'weight_decay': cfg.weight_decay}],
        lr=cfg.learning_rate,
        betas=[cfg.adam_beta1, cfg.adam_beta2],
        eps=cfg.adam_epsilon,
    )
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = build_lr_scheduler(
        cfg=cfg.lr_scheduler,
        optimizer=optimizer,
        num_processes=accelerator.num_processes,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
        max_train_steps=cfg.max_train_steps,
    )
    
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    init_accelerator(accelerator, cfg)
    
    # ------------------------------------------------------------------------------------------------
    # Run training with evaluation on each epoch
    # ------------------------------------------------------------------------------------------------

    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("##################  Running training  ##################")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    
    # custom checkpoint data registe to accelerator
    extra_state = CustomCheckpoint()
    accelerator.register_for_checkpointing(extra_state)
    
    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        checkpoint_path = get_resume_chekpoint_path(cfg.resume_from_checkpoint, cfg.output_dir)
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        if extra_state.epoch > 0:
            starting_epoch = extra_state.epoch + 1
        else:
            starting_epoch = get_checkpoint_epoch(checkpoint_path) + 1
        completed_steps = starting_epoch * num_update_steps_per_epoch
    
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    
    for epoch in range(starting_epoch, cfg.num_train_epochs):
        model.train()
        if cfg.with_tracking:
            total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                batch = Mono3DVGImageProcessor.prepare_batch(batch)
                outputs = model(**batch, output_attentions=False, output_hidden_states=False)
                loss = outputs.loss
                if cfg.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
            if step % 30 == 0:
                losses_dict = {k: v.item() for k, v in outputs.loss_dict.items()}
                losses_dict['loss'] = loss.item()
                msg = (
                    f'Epoch: [{epoch}][{step}/{len(train_dataloader)}]\t'
                    f'Loss_mono3dvg: {losses_dict['loss']:.2f}\t'
                    f'loss_ce: {losses_dict['loss_ce']:.2f}\t'
                    f'loss_bbox: {losses_dict['loss_bbox']:.2f}\t'
                    f'loss_depth: {losses_dict['loss_depth']:.2f}\t'
                    f'loss_dim: {losses_dict['loss_dim']:.2f}\t'
                    f'loss_angle: {losses_dict['loss_angle']:.2f}\t'
                    f'loss_center: {losses_dict['loss_center']:.2f}\t'
                    f'loss_depth_map: {losses_dict['loss_depth_map']:.2f}\t'
                )
                logger.info(msg)
        logger.info(f'Final Training Loss: {losses_dict['loss']}')
        
        logger.info("***** Running evaluation *****")
        metrics = evaluation(model, image_processor, accelerator, valid_dataloader, epoch, logger)
        msg = (
            f'Accuracy@0.25: {metrics['Overall_Acc@0.25']}%\t'
            f'Accuracy@0.5: {metrics['Overall_Acc@0.5']}%\t'
            f'Mean IoU: {metrics['Overall_MeanIoU']}%\t'
        )
        logger.info(f"Final Evaluation Result: " + msg)
        eval_result = metrics['Overall_Acc@0.25'] + metrics['Overall_Acc@0.25']
        if eval_result > extra_state.best_result:
            extra_state.best_eval_result = eval_result
            extra_state.best_epoch = epoch
            accelerator.save_state(get_checkpoint_dir(cfg.output_dir) / 'best')
            logger.info(f"Best Result: {extra_state.best_eval_result}, epoch: {epoch}")
        
        if cfg.with_tracking:
            accelerator.log(
                {
                    "lr": lr_scheduler.get_last_lr()[0],
                    **losses_dict,
                    **metrics,
                    "epoch": epoch,
                    "step": completed_steps,
                }, 
                step=completed_steps,
            )
        
        # Svae model
        if cfg.push_to_hub and epoch < cfg.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                cfg.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                image_processor.save_pretrained(cfg.output_dir)
                upload_output_folder(
                    api, repo_id, hub_token, commit_message=f"Training in progress epoch {epoch}", output_dir=cfg.output_dir
                )
        
        # Save checkpoint
        extra_state.epoch = epoch
        accelerator.save_state(get_checkpoint_dir(cfg.output_dir) / 'latest')


if __name__ == '__main__':
    main()
