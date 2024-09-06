from typing import Dict, List, Sequence, Mapping
from collections import defaultdict
import logging
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

from lib.models.image_processsing_mono3dvg import Mono3DVGImageProcessor


def nested_to_cpu(objects):
    """Move nested tesnors in objects to CPU if they are on GPU"""
    if isinstance(objects, torch.Tensor):
        return objects.cpu()
    elif isinstance(objects, Mapping):
        return type(objects)({k: nested_to_cpu(v) for k, v in objects.items()})
    elif isinstance(objects, (list, tuple)):
        return type(objects)([nested_to_cpu(v) for v in objects])
    elif isinstance(objects, (np.ndarray, str, int, float, bool)):
        return objects
    raise ValueError(f"Unsupported type {type(objects)}")


def calculate_3DIoU(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU of two axis-aligned 3D bboxes.
        
    Args:
        box_a (np.ndarray): 3D bbox in 6D-format (x, y, z, h, w, l)
        box_b (np.ndarray): 3D bbox in 6D-format (x, y, z, h, w, l)
    """
    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union


class IOU3DMetric(object):
    
    _TEST_INSTANCEID_SPLIT_FILE = 'Mono3DRefer/test_instanceID_split.json'
    
    def __init__(self, test_instanceID_split_file: str = None):
        self.iou_3dbox = defaultdict(list)
        split_file = test_instanceID_split_file or self._TEST_INSTANCEID_SPLIT_FILE
        with open(split_file, 'r') as f:
            self.test_instanceID_split = json.load(f)
    
    def _input_validator(
        self, 
        preds: Sequence[Dict[str, np.ndarray]], 
        targets: Sequence[Dict[str, np.ndarray]],
    ) -> None:
        """Ensure the correct input format of `preds` and `targets`."""
        if not isinstance(preds, Sequence):
            raise ValueError(f"Expected argument `preds` to be of type Sequence, but got {preds}")
        if not isinstance(targets, Sequence):
            raise ValueError(f"Expected argument `target` to be of type Sequence, but got {targets}")
        if len(preds) != len(targets):
            raise ValueError(
                f"Expected argument `preds` and `target` to have the same length, but got {len(preds)} and {len(targets)}"
            )
    
    def update(self, preds: List[Dict[str, np.ndarray]], target: List[Dict[str, np.ndarray]], num_processes: int = 1) -> None:
        """Update metric state."""
        
        self._input_validator(preds, target)
        
        for pred_item, gt_item in zip(preds, target):
            pred_box3D = pred_item['boxes_3d'] 
            gt_box3D = gt_item['boxes_3d']
            instanceID = gt_item['instance_id']
            # for distributed evaluation, grouped item should manully operate
            if num_processes > 1:
                assert instanceID.shape[0] == pred_box3D.shape[0]
                # gathers input_data and potentially drops duplicates in the last batch if on a distributed system
                # so recalculate gt
                box_dim = pred_box3D.shape[1]
                num_gt = torch.div(gt_box3D.shape[0], box_dim, rounding_mode='floor')
                num_drop = gt_box3D.shape[0] % box_dim
                if not num_drop == 0:
                    gt_box3D = gt_box3D[:-num_drop]
                gt_box3D = gt_box3D.split([box_dim for _ in range(num_gt)])
                for pred, gt, id in zip(pred_box3D, gt_box3D, instanceID):
                    self._update_item(pred, gt, id)
            else:
                pred_box3D = pred_box3D[0] # Only calculate for top-1 prediction
                self._update_item(pred_box3D, gt_box3D, instanceID)
    
    def _update_item(self, pred_box3D, gt_box3D, instanceID):
        # (dim, loc) -> (loc, dim)
        pred_box3D = np.array([pred_box3D[3], pred_box3D[4], pred_box3D[5], pred_box3D[0], pred_box3D[1], pred_box3D[2]], dtype=np.float32)
        gt_box3D = np.array([gt_box3D[3], gt_box3D[4], gt_box3D[5], gt_box3D[0], gt_box3D[1], gt_box3D[2]], dtype=np.float32)
        # real 3D center in 3D space is (x, y, z) - (0, h/2, 0)
        pred_box3D[1] -= pred_box3D[3] / 2
        gt_box3D[1] -= gt_box3D[3] / 2
            
        iou = calculate_3DIoU(pred_box3D, gt_box3D)
        self.iou_3dbox['Overall'].append(iou)
        if instanceID in self.test_instanceID_split['Unique']:
            self.iou_3dbox['Unique'].append(iou)
        else:
            self.iou_3dbox['Multiple'].append(iou)
            
        if instanceID in self.test_instanceID_split['Near']:
            self.iou_3dbox['Near'].append(iou)
        elif instanceID in self.test_instanceID_split['Medium']:
            self.iou_3dbox['Medium'].append(iou)
        elif instanceID in self.test_instanceID_split['Far']:
            self.iou_3dbox['Far'].append(iou)
            
        if instanceID in self.test_instanceID_split['Easy']:
            self.iou_3dbox['Easy'].append(iou)
        elif instanceID in self.test_instanceID_split['Moderate']:
            self.iou_3dbox['Moderate'].append(iou)
        elif instanceID in self.test_instanceID_split['Hard']:
            self.iou_3dbox['Hard'].append(iou)

    def compute(self, only_overall: bool = True) -> dict:
        """Compute the metric."""
        result_dict = {}
        for split_name, iou_3dbox in self.iou_3dbox.items():
            if only_overall and split_name != 'Overall':
                continue
            length = len(iou_3dbox)
            acc5 = np.sum(np.array(iou_3dbox) > 0.5) / length
            acc25 = np.sum(np.array(iou_3dbox) > 0.25) / length
            miou = np.mean(iou_3dbox)
            result_dict.update({
                f'{split_name}_Acc@0.5': acc5,
                f'{split_name}_Acc@0.25': acc25,
                f'{split_name}_MeanIoU': miou,
            })
        return result_dict


def evaluation(
    model: torch.nn.Module,
    image_processor: Mono3DVGImageProcessor,
    accelerator: Accelerator,
    dataloader: DataLoader,
    epoch: int = None,
    logger: logging.Logger = None,
    only_overall: bool = True,
) -> dict:
    model.eval()
    metric = IOU3DMetric()
    
    for step, batch in enumerate(tqdm(dataloader, disable=not accelerator.is_local_main_process)):
        batch = Mono3DVGImageProcessor.prepare_batch(batch, return_info=True)
        info = batch.pop("info")
        with torch.no_grad():
            outputs = model(**batch)
        
        # For metric computation we need to collect ground truth and predicted 3D boxes in the same format

        # 1. Collect predicted 3Dboxes, classes, scores
        # image_processor convert boxes from size_3d, box and depth predict to Real 3D box format [cx, cy, cz, h, w, l] in absolute coordinates.
        predictions = image_processor.post_process_3d_object_detection(outputs, batch["calibs"], target_sizes=batch["img_sizes"], top_k=1)

        # 2. Collect ground truth boxes in the same format for metric computation
        # Do the same, convert 3D boxes to Pascal VOC format
        target = []
        for gt_3dbox, instance_id in zip(info['gt_3dbox'], info['instance_id']):
            target.append({"boxes_3d": gt_3dbox, "instance_id": instance_id})
        
        all_predictions, all_targets = accelerator.gather_for_metrics((predictions, target))
        all_predictions = nested_to_cpu(all_predictions)
        all_targets = nested_to_cpu(all_targets)
        
        metric.update(all_predictions, all_targets, num_processes=accelerator.num_processes)
        
        if step % 30 == 0 and logger is not None:
            metrics = metric.compute(only_overall=True)
            msg = (
                    f'Evaluation: [{epoch}][{step}/{len(dataloader)}]\t' if epoch is not None else f'Epoch: [{step}/{len(dataloader)}]\t'
                    f'Loss_mono3dvg: {outputs.loss.item():.2f}\t'
                    f'Accu25: {metrics['Overall_Acc@0.25']*100:.2f}%\t'
                    f'Accu5: {metrics['Overall_Acc@0.5']*100:.2f}%\t'
                    f'Mean_iou: {metrics['Overall_MeanIoU']*100:.2f}%\t'
                )
            logger.info(msg)
    
    metrics = metric.compute(only_overall=only_overall)

    # Convert metrics to percentage
    metrics = {k: round(v.item(), 4) * 100 for k, v in metrics.items()}

    return metrics

