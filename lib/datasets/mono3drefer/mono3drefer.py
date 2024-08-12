from functools import partial
from typing import Any, Mapping
from argparse import Namespace

import albumentations as A
import numpy as np
from datasets import load_dataset

from accelerate import Accelerator
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import corners_to_center_format

from lib.datasets.utils import angle2class
from lib.datasets.mono3drefer.utils import Calibration, generate_corners3d, affine_transform
from lib.datasets.transforms import PhotometricDistort, HorizontalFlip, Affine, NoOp
from lib.models.image_processsing_mono3dvg import Mono3DVGImageProcessor
from utils.parser import dict_to_namespace


def format_object3d_annotations(caption: str, calib: Calibration, object: Namespace, info: Namespace) -> dict:
    # encoding 2d/3d boxes
    center_box2d = corners_to_center_format(object.box2d) # (cx, cy, w, h)
    box3d = np.concatenate([object.center_3d, object.center_3d - object.box2d[0:2], object.box2d[2:4] - object.center_3d], axis=0) # (cx, cy, l, r, t, b)
    
    # encoding heading angle
    heading_angle = calib.ry2alpha(object.ry, (object.box2d[0] + object.box2d[2]) / 2)
    if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
    if heading_angle < -np.pi: heading_angle += 2 * np.pi
    heading_bin, heading_res = angle2class(heading_angle)
    heading_bin = np.array([heading_bin], dtype=np.int64)
    heading_res = np.array([heading_res], dtype=np.float32)
    
    # encoding size_3d
    size_3d = np.array([object.h, object.w, object.l], dtype=np.float32)
    
    # encoding corner_3d
    if hasattr(object, 'is_flipped'):
        info.projected_corner_3d[:, 0] = info.img_size[0] - info.projected_corner_3d[:, 0]
    if hasattr(object, 'trans'):
        info.projected_corner_3d = np.stack([affine_transform(corner, object.trans) for corner in info.projected_corner_3d], axis=0)
    
    if object.trucation <= 0.5 and object.occlusion <= 2:
        mask_2d = np.ones((1), dtype=bool)
    else:
        mask_2d = np.zeros((1), dtype=bool)
    
    # language encoding
    caption = caption.lower()
    
    return {
        'targets': {
            'labels': np.array([info.category], dtype=np.int8), # (1,)
            'boxes': center_box2d[np.newaxis, ...], # (1, 4)
            'boxes_3d': box3d[np.newaxis, ...], # (1, 6)
            'depth': np.array([object.pos[-1]], dtype=np.float32), # (1, 1)
            'size_3d': size_3d[np.newaxis, ...], # (1, 3)
            'heading_bin': heading_bin[np.newaxis, ...], # (1, 1)
            'heading_res': heading_res[np.newaxis, ...], # (1, 1)
            'mask_2d': mask_2d, # (1,)
        },
        'caption': caption,
        'calib': calib.P2,
        'info': {
            'img_id': info.img_id,
            'img_size': info.img_size,
            'instance_id': info.instance_id,
            'anno_id': info.ann_id,
            'bbox_downsample_ratio': info.bbox_downsample_ratio,
            'gt_3dbox': info.gt_3dbox,
            'corners_3d': info.projected_corner_3d[np.newaxis, ...], # (1, 8, 2)
        },
    }


def augment_and_transform_batch(
    examples: Mapping[str, Any],
    transform: A.Compose,
    image_processor: Mono3DVGImageProcessor,
    return_pixel_mask: bool = False,
) -> BatchFeature:
    """Apply augmentations for 3D object grounding task"""

    images = []
    annotations = []
    for image, caption, calib, object, info in zip(examples["image"], examples['description'], examples['calib'], examples["objects"], examples["info"]):
        image = np.array(image.convert("RGB"))
        calib = Calibration({k: np.array(v, dtype=np.float32) for k, v in calib.items()})
        
        # 3D object
        object = dict_to_namespace({k: np.array(v[0]) if isinstance(v[0], list) else v[0] for k, v in object.items()})
        center_3d = object.pos + [0, -object.h / 2, 0]  # real 3D center in 3D space
        center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
        center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
        object.center_3d = center_3d[0]  # shape adjustment
        
        # info
        info = dict_to_namespace({k: np.array(v[0]) if isinstance(v[0], list) else v[0] for k, v in info.items()})
        info.bbox_downsample_ratio = info.img_size / (np.array([1280, 384]) // 32)
        info.gt_3dbox = [object.h, object.w, object.l, float(object.pos[0]), float(object.pos[1]), float(object.pos[2])]
        # corner_3d        
        corner_3d = generate_corners3d(object.l, object.h, object.w, object.ry, object.pos)
        info.projected_corner_3d, _ = calib.rect_to_img(corner_3d) # (8, 2)

        # apply augmentations
        output = transform(image=image, object3d=object)
        images.append(output["image"])
        
        # format annotations
        formatted_annotations = format_object3d_annotations(
            caption=caption, calib=calib, object=output["object3d"], info=info
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


def build_dataset(cfg, accelerator: Accelerator = None):
    # Load dataset
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset = load_dataset(cfg.name, cache_dir=cfg.cache_dir)
    
    # Get dataset categories and prepare mappings for label_name <-> label_id
    categories = dataset['train'].features['info'].feature['category'].names
    id2label = dict(enumerate(categories))
    label2id = {v: k for k, v in id2label.items()}

    # Load image processor
    image_processor = Mono3DVGImageProcessor()
    
    # Define image augmentations and dataset transforms
    train_augment_and_transform = A.Compose(
        [
            PhotometricDistort() if cfg.aug_pd else NoOp(),
            HorizontalFlip(p=cfg.random_flip),
            Affine(scale=cfg.scale,shift=cfg.shift,p=cfg.random_crop,) if cfg.aug_crop else NoOp(),
        ], 
        strict=False,
    )
    validation_transform = A.Compose([NoOp()], strict=False,)
    
    # Make transform functions for batch and apply for dataset splits
    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
    )
    
    if accelerator is None:
        train_dataset = dataset["train"].with_transform(train_transform_batch)
        valid_dataset = dataset["validation"].with_transform(validation_transform_batch)
        test_dataset = dataset["test"].with_transform(validation_transform_batch)
    else:
        with accelerator.main_process_first():
            train_dataset = dataset["train"].with_transform(train_transform_batch)
            valid_dataset = dataset["validation"].with_transform(validation_transform_batch)
            test_dataset = dataset["test"].with_transform(validation_transform_batch)
        
    
    return train_dataset, valid_dataset, test_dataset, id2label, label2id
