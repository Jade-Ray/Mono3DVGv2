from typing import Union, List, Optional, Tuple, Dict
from collections import OrderedDict
import torch
import PIL
import numpy as np
import cv2 as cv
from torchvision.transforms.v2 import functional as F
from transformers.image_transforms import (
    to_pil_image, 
    ExplicitEnum,
    center_to_corners_format
)
from transformers.image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    get_image_size,
)

from lib.datasets.utils import draw_msra_gaussian, draw_umich_gaussian, gaussian_radius


def convert_pil2mat(img: PIL.Image.Image) -> np.ndarray:
    data = np.array(img)
    assert data.ndim in (2, 3), f"Expected 2D or 3D array, got {data.ndim}D array instead."
    if data.ndim == 2:
        return data
    return cv.cvtColor(data, cv.COLOR_RGB2BGR)


def convert_mat2pil(mat: np.ndarray) -> PIL.Image.Image:
    assert mat.ndim in (2, 3), f"Expected 2D or 3D array, got {mat.ndim}D array instead."
    if mat.ndim == 2:
        return PIL.Image.fromarray(np.uint8(mat))
    return PIL.Image.fromarray(cv.cvtColor(mat, cv.COLOR_BGR2RGB))


def cv_rectangle(mat, pt1, pt2, color, thickness=1, style='', alpha=1.):
    """Extend opencv rectangle function.

    Args:
        mat (np.array): The Mat image.
        pt1 ([x, y]): The left-top corner point.
        pt2 ([x, y]): The right-bottom corner point.
        color (list): BGR Color of the rectangle.
        thickness (int, optional): Thickness of the rectangle. Defaults to 1.
        style (str, optional): Style of the rectangle with 3 options.`dashed` is draw dashed line of rectangle, `dotted` is draw dotted line of rectangle, `''` is norm rectangle. Defaults to ''.
        alpha (float, optional): Alpha of the rectangle. Defaults to `1.`.
    """
    if pt1[0] == pt2[0] or pt1[1] == pt2[1]:
        return
    overlay = mat.copy()
    if style in ('dashed', 'dotted'):
        pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
        drawpoly(overlay, pts, color, thickness, style)
    else:
        cv.rectangle(overlay, pt1, pt2, color, thickness)
    cv.addWeighted(overlay, alpha, mat, 1 - alpha, 0, mat)


def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)


def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=10):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv.line(img,s,e,color,thickness)
            i+=1


def get_polygon_pts(corners_3d):
    corners_3d = corners_3d.copy().transpose() # (8, 2) -> (2, 8)
    rb, _ = np.argmax(corners_3d, axis=-1)
    rt = rb + 4
    lb, _ = np.argmin(corners_3d, axis=-1)
    lt = lb + 4
    argmaxes = np.argpartition(corners_3d, -4, axis=-1)
    bottom = argmaxes[1, -2:]
    bottom = bottom[-2] if bottom[-1] == rb or bottom[-1] == lb else bottom[-1]
    top = argmaxes[1, :2]
    top = top[1] if top[0] == rt or top[0] == lt else top[0]
    return [rb, bottom, lb, lt, top, rt]


def draw_projected_box3d(image, corners3d, color=(255, 255, 255), thickness=1, fill=False):
    ''' Draw 3d bounding box in image
    input:
        image: RGB image
        corners3d: (8, 2) array of vertices (in image plane) for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7

    '''

    corners3d = corners3d.astype(np.int32)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv.LINE_AA)
        i, j = k + 4, (k + 1) % 4 + 4
        cv.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv.LINE_AA)
        i, j = k, k + 4
        cv.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness, lineType=cv.LINE_AA)
    
    if fill:
        overlay = image.copy()
        alpha = 0.3
        idx = get_polygon_pts(corners3d)
        pts = corners3d[idx].reshape(1, -1, 2)
        cv.fillPoly(overlay, pts, color)
        cv.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return image


def infer_image_format(image: Union[np.ndarray, PIL.Image.Image]) -> str:
    if isinstance(image, np.ndarray):
        return ImageFormat.MAT
    elif isinstance(image, PIL.Image.Image):
        return ImageFormat.PIL
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


class RCG(object):
    """random color generator"""
    def __init__(self, max_len = 1000):
        self.color_map = OrderedDict()
        self.max_len = max_len
    
    def _random_generator(self, id):
        color = tuple(np.uint8(np.random.choice(range(256), size=3)))
        while len(self.color_map) >=  self.max_len:
            self.color_map.popitem(last=False)
        self.color_map[id] = color
    
    def __call__(self, id):
        if id not in self.color_map.keys():
            self._random_generator(id)
        return self.color_map[id]


class ImageFormat(ExplicitEnum):
    MAT = "cv_mat"
    PIL = "pil_image"


class Mono3DVGPlotter:
    
    _image_mean = IMAGENET_DEFAULT_MEAN
    _image_std = IMAGENET_DEFAULT_STD
    _rcg = RCG()
    
    @classmethod
    def denormalize(
        cls, 
        image: torch.Tensor,
        mean: Optional[Union[float, List[float]]] = None,
        std: Optional[Union[float, List[float]]] = None,
        rescale: bool = True,
    ) -> torch.Tensor:
        """
        Denormalize the image tensor.
        Args:
            image: The image tensor to denormalize.
            mean: The mean to use for denormalization.
            std: The standard deviation to use for denormalization.
            rescale: If True, rescale the image to be in [0, 255].
        Returns:
            The denormalized image tensor.
        """
        mean = cls._image_mean if mean is None else mean
        std = cls._image_std if std is None else std
        if image.dtype.is_floating_point and image.min() < 0:
            if mean is not None and std is not None:
                # Undo normalization for better visualization
                mean = torch.as_tensor(mean)
                std = torch.as_tensor(std)
                std_inv = 1 / (std + 1e-7)
                mean_inv = -mean * std_inv
                # (tensor - (-mean / std)) / 1/std = std * tensor + mean
                image = F.normalize(image, mean=mean_inv, std=std_inv)
            else:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                image -= image.min()
                image /= image.max()
        
        if rescale:
            image = F.to_dtype(image, torch.uint8, scale=True)
        
        return image
    
    @classmethod
    def denormalize_annotation(
        cls,
        annotation: Dict[str, torch.Tensor],
        image_size: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        W, H = image_size
        new_annotation = {}
        for key, value in annotation.items():
            if key == "boxes":
                new_annotation[key] = center_to_corners_format(value.cpu().numpy() * np.array([W, H, W, H]))
            elif key == "boxes_3d":
                new_annotation[key] = value.cpu().numpy() * np.array([W, H, W, W, H, H])
            else:
                new_annotation[key] = value.cpu().numpy()
        return new_annotation
    
    @classmethod
    def to_pil_images(cls, images: torch.Tensor) -> List[PIL.Image.Image]:
        """
        Convert a tensor size of (batch_size, C, H, W) to a list of PIL images.
        Args:
            images: The tensor to convert.
        Returns:
            A list of PIL images.
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.dim() != 4:
            raise ValueError(f"Images must have 3 or 4 dimensions, got {images.dim()}.")
        return [to_pil_image(cls.denormalize(image)) for image in images]
    
    @classmethod
    def draw_boxes2d(
        cls,
        image: Union[np.ndarray, PIL.Image.Image],
        boxes2d: np.ndarray,
        color: Union[List[int], Tuple[int]] = (0, 255, 0),
        random_color: bool = False,
        thickness: int = 2,
        format: str = ImageFormat.PIL,
    ) -> Union[np.ndarray, PIL.Image.Image]:
        """
        Draw a bounding box on an image.
        Args:
            image: The image to draw the bounding box on.
            boxes2d: The bounding boxes to draw, (N, 4) array of (x1, y1, x2, y2). 
            color: The color of the bounding box.
            random_color: If True, use a random color for each bounding box.
            thickness: The thickness of the bounding box.
            format: The format of the image.
        Returns:
            The image with the bounding box drawn on it.
        """
        if format == ImageFormat.PIL:
            image = convert_pil2mat(image)
        
        boxes2d = boxes2d.astype(int)
        for i, (l, t, r, b) in enumerate(boxes2d):
            if random_color:
                color = cls._rcg(i)
            cv.rectangle(image, (l, t), (r, b), color, thickness)
        
        if format == ImageFormat.PIL:
            image = convert_mat2pil(image)
        
        return image
    
    @classmethod
    def draw_projected_boxes3d(
        cls, 
        image: Union[np.ndarray, PIL.Image.Image],
        boxes3d: np.ndarray,
        color: Union[List[int], Tuple[int]] = (0, 255, 0),
        random_color: bool = False,
        thickness: int = 2,
        fill: bool = False,
        format: str = ImageFormat.PIL,
    ) -> Union[np.ndarray, PIL.Image.Image]:
        """
        Draw a projected 3D corners on an image.
        Args:
            image: The image to draw the 3D bounding box on.
            boxes3d: The 3D corners to draw, (N, 8, 2) array of vertices (in image plane) for the 3d box.
            color: The color of the bounding box.
            random_color: If True, use a random color for each bounding box.
            thickness: The thickness of the bounding box.
            fill: If True, fill the 3D bounding box with 0.3 alpha.
            format: The format of the image.
        Returns:
            The image with the 3D bounding box drawn on it.
        """
        if format == ImageFormat.PIL:
            image = convert_pil2mat(image)
        
        boxes3d = boxes3d.astype(int)
        for i, box3d in enumerate(boxes3d):
            if random_color:
                color = cls._rcg(i)
            draw_projected_box3d(image, box3d, color, thickness, fill=fill)
        
        if format == ImageFormat.PIL:
            image = convert_mat2pil(image)
        
        return image

    @classmethod
    def draw_depth_map(
        cls,
        image: Union[np.ndarray, PIL.Image.Image],
        boxes2d: np.ndarray,    
        projected_centers3d: np.ndarray,
        depthes: np.ndarray,
        max_depth: float = 60.0,
        gaussian_format: str = "msra",
        format: str = ImageFormat.PIL,
    ):
        if format == ImageFormat.PIL:
            image = convert_pil2mat(image)
        
        heatmap = np.zeros((image.shape[0], image.shape[1]))
        boxes2d = boxes2d.astype(int)
        projected_centers3d = projected_centers3d.astype(int)
        for (l, t, r, b), (cx, cy), depth in zip(boxes2d, projected_centers3d, depthes):
            heatmap[t:b, l:r] = max_depth - min(depth, max_depth)
            if gaussian_format == "msra":
                sigma = max(b - t, r - l)
                heatmap = draw_msra_gaussian(heatmap, (cx, cy), sigma)
            elif gaussian_format == "umich":
                radius = gaussian_radius((b - t, r - l),)
                heatmap = draw_umich_gaussian(heatmap, (cx, cy), int(radius))
        
        if format == ImageFormat.PIL:
            heatmap = convert_mat2pil(heatmap)
        
        return heatmap
