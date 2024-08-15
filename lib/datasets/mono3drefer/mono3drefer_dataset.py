import ast
import time
import json
from pathlib import Path
from argparse import Namespace

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .pd import PhotometricDistort
from .utils import Calibration, get_affine_transform, affine_transform
from lib.datasets.utils import angle2class


class Object3d(object):
    """ 3d object label """
    def __init__(self, label):
        self.src = label
        self.trucation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.pos = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()
    
    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if self.trucation == -1:
            self.level_str = 'DontCare'
            return 0

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos
        return corners3d

    def to_bev_box2d(self, oblique=True, voxel_size=0.1):
        """
        :param bev_shape: (2) for bev shape (h, w), => (y_max, x_max) in image
        :param voxel_size: float, 0.1m
        :param oblique:
        :return: box2d (4, 2)/ (4) in image coordinate
        """
        if oblique:
            corners3d = self.generate_corners3d()
            xz_corners = corners3d[0:4, [0, 2]]
            box2d = np.zeros((4, 2), dtype=np.int32)
            box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            box2d[:, 1] = Object3d.BEV_SHAPE[0] - 1 - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            box2d[:, 0] = np.clip(box2d[:, 0], 0, Object3d.BEV_SHAPE[1])
            box2d[:, 1] = np.clip(box2d[:, 1], 0, Object3d.BEV_SHAPE[0])
        else:
            box2d = np.zeros(4, dtype=np.int32)
            # discrete_center = np.floor((self.pos / voxel_size)).astype(np.int32)
            cu = np.floor((self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            cv = Object3d.BEV_SHAPE[0] - 1 - ((self.pos[2] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            half_l, half_w = int(self.l / voxel_size / 2), int(self.w / voxel_size / 2)
            box2d[0], box2d[1] = cu - half_l, cv - half_w
            box2d[2], box2d[3] = cu + half_l, cv + half_w

        return box2d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.pos, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.trucation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.pos[0], self.pos[1], self.pos[2],
                       self.ry)
        return kitti_str
    
    def to_dataframe(self):
        return pd.DataFrame([{
            'trucation': self.trucation,
            'occlusion': self.occlusion,
            'alpha': self.alpha,
            'box2d': self.box2d,
            'h': self.h,
            'w': self.w,
            'l': self.l,
            'pos': self.pos,
            'ry': self.ry,
            'score': self.score,
            'level': self.level,
        }])


class Mono3DRefer():
    def __init__(self, root_dir, split: str):
        # load dataset
        self.anns, self.imgs, self.cailbs = dict(), dict(), dict()
        print('loading annotations into memory...')
        tic = time.time()
        root_dir = Path(root_dir) if isinstance(root_dir, str) else root_dir
        assert split in ['train', 'val', 'test']
        split_file = root_dir / f'Mono3DRefer_{split}_image.txt'
        ann_path = root_dir / 'Mono3DRefer.json'
        anns = self._split_ann(ann_path, split_file)
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        print('creating index...')
        pbar = tqdm(enumerate(anns))
        for i, ann in pbar:
            pbar.set_description(f'Index {i}')
            im_name = ann['im_name']
            self.anns[i] = {
                'img_id': ann['im_name'],
                'instance_id': ann['instanceID'],
                'ann_id': ann['ann_id'],
                'category': self.label2id[ann['objectName']],
                'description': ann['description'],
                'label_2': ast.literal_eval(ann['label_2']),
            }
            self.imgs[i] = root_dir / 'images' / f'{im_name}.png'
            self.cailbs[i] = root_dir / 'calib' / f'{im_name}.txt'
        print('index created!')
    
    def _split_ann(self, ann_path, split_file)-> list[dict]:
        split_anns = []
        image_list = [x.strip() for x in open(split_file).readlines()]
        with open(ann_path, 'r') as f:
            anns = json.load(f)
        for ann in anns:
            if ann['im_name'] in image_list:
                split_anns.append(ann)
        return split_anns
    
    @property
    def label2id(self):
        return {'pedestrian': 0, 'car': 1, 'cyclist': 2,'van':3, 'truck':4, 'tram':5, 'bus':6, 'person_sitting':7, 'motorcyclist':8 }
    
    def loadImg(self, ids):
        return self.imgs[ids]
    
    def loadCalib(self, id):
        return self.cailbs[id]
    
    def loadAnn(self, id) -> dict:
        return self.anns[id]


class Mono3DReferDataset(Dataset):
    def __init__(self, split, cfg: Namespace):
        
        self.mono3d_refer = Mono3DRefer(cfg.root_dir, split)
        self.ids = list(sorted(self.mono3d_refer.imgs.keys()))
        self.id2label = self.mono3d_refer.label2id
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.split = split
        self.max_objs = 1
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = getattr(cfg, 'use_3d_center', True)
        self.meanshape = getattr(cfg, 'meanshape', False)
        
        # data augmentation configuration
        self.data_augmentation = True if split in ['train'] else False
        
        self.aug_pd = getattr(cfg, 'aug_pd', False)
        self.aug_crop = getattr(cfg, 'aug_crop', False)
        self.aug_calib = getattr(cfg, 'aug_calib', False)
        
        self.random_flip = getattr(cfg, 'random_flip', 0.5)
        self.random_crop = getattr(cfg, 'random_crop', 0.5)
        self.scale = getattr(cfg, 'scale', 0.4)
        self.shift = getattr(cfg, 'shift', 0.1)
        
        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls_mean_size = np.array([
            [1.76255119    ,0.66068622   , 0.84422524   ],
            [1.52563191462 ,1.62856739989, 3.88311640418],
            [1.73698127    ,0.59706367   , 1.76282397   ],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
        ])
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 32
        self.pd = PhotometricDistort()
        self.clip_2d = getattr(cfg, 'clip_2d', False)
    
    def _load_image(self, id):
        image_path = self.mono3d_refer.loadImg(id)
        assert image_path.exists(), f'Image path {image_path} does not exist'
        return Image.open(image_path)
    
    def _load_calib(self, id) -> Calibration:
        calib_path = self.mono3d_refer.loadCalib(id)
        assert calib_path.exists(), f'Calib path {calib_path} does not exist'
        calib = Calibration(calib_path)
        return calib
    
    def _load_objects(self, labels: str) -> Object3d:
        object = Object3d(labels)
        return object
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index: int):
        #  ============================   get inputs   ===========================
        img = self._load_image(index)
        img_size = np.array(img.size)
        features_size = self.resolution // self.downsample    # W * H
        
        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag, random_crop_flag = False, False
        
        if self.data_augmentation:
            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.aug_crop:
                if np.random.random() < self.random_crop:
                    random_crop_flag = True
                    crop_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                    crop_size = img_size * crop_scale
                    center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                    center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
        
        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W
        
        ann = self.mono3d_refer.loadAnn(index)
        object = self._load_objects(ann['label_2'])
        calib = self._load_calib(index)
        
        info = {'img_id': index,
                'img_size': img_size,
                'instance_id': ann['instance_id'],
                'anno_id': ann['ann_id'],
                'bbox_downsample_ratio': img_size / features_size,
                'gt_3dbox': [object.h,object.w,object.l,float(object.pos[0]),float(object.pos[1]),float(object.pos[2])] }

        #  ============================   get labels   ==============================
        if random_flip_flag:
            if self.aug_calib:
                calib.flip(img_size)
            [x1, _, x2, _] = object.box2d
            object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
            object.alpha = np.pi - object.alpha
            object.ry = np.pi - object.ry
            if self.aug_calib:
                object.pos[0] *= -1
            if object.alpha > np.pi:  object.alpha -= 2 * np.pi  # check range
            if object.alpha < -np.pi: object.alpha += 2 * np.pi
            if object.ry > np.pi:  object.ry -= 2 * np.pi
            if object.ry < -np.pi: object.ry += 2 * np.pi

        # labels encoding
        calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32) 
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)

        # process 2d bbox & get 2d center
        bbox_2d = object.box2d.copy()

        # add affine transformation for 2d boxes.
        bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
        bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

        # process 3d center
        center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                             dtype=np.float32)  # W * H
        corner_2d = bbox_2d.copy()

        center_3d = object.pos + [0, -object.h / 2, 0]  # real 3D center in 3D space
        center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
        center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
        center_3d = center_3d[0]  # shape adjustment
        if random_flip_flag and not self.aug_calib:  # random flip for center3d
            center_3d[0] = img_size[0] - center_3d[0]
        center_3d = affine_transform(center_3d.reshape(-1), trans)
        
        # class
        labels[0] = ann['category']
        
        # encoding 2d/3d boxes
        w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
        size_2d[0] = 1. * w, 1. * h

        center_2d_norm = center_2d / self.resolution
        size_2d_norm = size_2d[0] / self.resolution

        corner_2d_norm = corner_2d
        corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
        corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
        center_3d_norm = center_3d / self.resolution

        l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
        t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

        if l < 0 or r < 0 or t < 0 or b < 0:
            if self.clip_2d:
                l = np.clip(l, 0, 1)
                r = np.clip(r, 0, 1)
                t = np.clip(t, 0, 1)
                b = np.clip(b, 0, 1)

        boxes[0] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
        boxes_3d[0] = center_3d_norm[0], center_3d_norm[1], l, r, t, b
        
        # encoding depth
        depth[0] = object.pos[-1] * crop_scale
        
        # encoding heading angle
        heading_angle = calib.ry2alpha(object.ry, (object.box2d[0] + object.box2d[2]) / 2)
        if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
        if heading_angle < -np.pi: heading_angle += 2 * np.pi
        heading_bin[0], heading_res[0] = angle2class(heading_angle)
        
        # encoding size_3d
        src_size_3d[0] = np.array([object.h, object.w, object.l], dtype=np.float32)
        mean_size = self.cls_mean_size[ann['category']]
        size_3d[0] = src_size_3d[0] - mean_size
        
        if object.trucation <= 0.5 and object.occlusion <= 2:
            mask_2d[0] = 1
            
        calibs[0] = calib.P2
        
        # language encoding
        text = ann['description'].lower()
        
        return {
            'pixel_values': img,
            'captions': text,
            'calibs': calibs,
            'targets': {
                'labels': labels,
                'boxes': boxes,
                'boxes_3d': boxes_3d,
                'depth': depth,
                'size_3d': size_3d,
                'heading_bin': heading_bin,
                'heading_res': heading_res,
                'mask_2d': mask_2d,
            },
            'info': info,
        }
        
