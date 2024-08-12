from enum import Enum
from typing import Any, Callable
from argparse import Namespace

import cv2
import numpy as np
from PIL import Image
from numpy import random
from albucore.utils import is_rgb_image, get_num_channels
from pydantic import Field

import albumentations as A
import albumentations.augmentations.functional as fmain
import albumentations.augmentations.geometric.functional as fgeometric
from albumentations import random_utils
from albumentations.core.utils import to_tuple
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import (
    ScaleFloatType,
    SizeType,
)

from lib.datasets.mono3drefer.utils import get_affine_transform, affine_transform


class Targets(Enum):
    IMAGE = "Image"
    OBJECT3D = "Object3D"


class PhotometricDistort(A.ColorJitter):
    
    def __init__(
        self,
        brightness: ScaleFloatType = (0.87451, 1.12549),
        contrast: ScaleFloatType = (0.5, 1.5),
        saturation: ScaleFloatType = (0.5, 1.5),
        hue: ScaleFloatType = (-0.05, 0.05),
    ):
        super().__init__(brightness, contrast, saturation, hue, p=1.0)
    
    def apply(
        self,
        img: np.ndarray,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        order: list[int],
        **params: Any,
    ) -> np.ndarray:
        if random.randint(2):
            order = [0, 1, 2, 3] # Brightness, Contrast, Saturation, Hue
        else:
            order = [0, 2, 3, 1] # Brightness, Saturation, Hue, Contrast
        if not is_rgb_image(img):
            msg = "PhotometricDistort transformation expects 3-channel images."
            raise TypeError(msg)
        color_transforms = [brightness, contrast, saturation, hue]
        for i in order:
            # random apply
            if random.randint(2):
                img = self.transforms[i](img, color_transforms[i])
        
        # random apply channel shuffle
        if random.randint(2):
            ch_arr = list(range(img.shape[2]))
            ch_arr = random_utils.shuffle(ch_arr)
            img = fmain.channel_shuffle(img, ch_arr)
        return img


class CustomDualTransform(DualTransform):
    
    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
            "object3d": self.apply_to_object3d,
        }
    
    def apply_to_object3d(self, object3d: Namespace, *args: Any, **params: Any) -> Namespace:
        msg = f"Method apply_to_object is not implemented in class {self.__class__.__name__}"
        raise NotImplementedError(msg)


class HorizontalFlip(CustomDualTransform):
    """Flip the input horizontally around the y-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, object3d

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.OBJECT3D)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if get_num_channels(img) > 1 and img.dtype == np.uint8:
            # Opencv is faster than numpy only in case of
            # non-gray scale 8bits images
            return fgeometric.hflip_cv2(img)

        return fgeometric.hflip(img)
    
    def apply_to_object3d(self, object3d: Namespace, *args: Any, **params: Any) -> Namespace:
        width = params["cols"]
        x1, _, x2, _ = object3d.box2d
        object3d.box2d[0], object3d.box2d[2] = width - x2, width - x1
        object3d.alpha = np.pi - object3d.alpha
        object3d.ry = np.pi - object3d.ry
        if object3d.alpha > np.pi:  object3d.alpha -= 2 * np.pi  # check range
        if object3d.alpha < -np.pi: object3d.alpha += 2 * np.pi
        if object3d.ry > np.pi:  object3d.ry -= 2 * np.pi
        if object3d.ry < -np.pi: object3d.ry += 2 * np.pi
        object3d.center_3d[0] = width - object3d.center_3d[0]
        object3d.is_flipped = True
        return object3d

    def get_transform_init_args_names(self) -> tuple[()]:
        return ()


class Affine(CustomDualTransform):
    """Augmentation to apply affine transformations to images.

    Affine transformations involve:

        - Translation ("move" image on the x-/y-axis)
        - Rotation
        - Scaling ("zoom" in/out)
        - Shear (move one side of the image, turning a square into a trapezoid)

    All such transformations can create "new" pixels in the image without a defined content, e.g.
    if the image is translated to the left, pixels are created on the right.
    A method has to be defined to deal with these pixel values.
    The parameters `cval` and `mode` of this class deal with this.

    Some transformations involve interpolations between several pixels
    of the input image to generate output pixel values. The parameters `interpolation` and
    `mask_interpolation` deals with the method of interpolation used for this.

    Args:
        scale (number, tuple of number or dict): Scaling factor to use, where ``1.0`` denotes "no change" and
            ``0.5`` is zoomed out to ``50`` percent of the original size.
                * If a single number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``.
                  That the same range will be used for both x- and y-axis. To keep the aspect ratio, set
                  ``keep_ratio=True``, then the same value will be used for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes. Note that when
                  the ``keep_ratio=True``, the x- and y-axis ranges should be the same.
        translate_percent (None, number, tuple of number or dict): Translation as a fraction of the image height/width
            (x-translation, y-translation), where ``0`` denotes "no change"
            and ``0.5`` denotes "half of the axis size".
                * If ``None`` then equivalent to ``0.0`` unless `translate_px` has a value other than ``None``.
                * If a single number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``.
                  That sampled fraction value will be used identically for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        translate_px (None, int, tuple of int or dict): Translation in pixels.
                * If ``None`` then equivalent to ``0`` unless `translate_percent` has a value other than ``None``.
                * If a single int, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from
                  the discrete interval ``[a..b]``. That number will be used identically for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        rotate (number or tuple of number): Rotation in degrees (**NOT** radians), i.e. expected value range is
            around ``[-360, 360]``. Rotation happens around the *center* of the image,
            not the top left corner as in some other frameworks.
                * If a number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``
                  and used as the rotation value.
        shear (number, tuple of number or dict): Shear in degrees (**NOT** radians), i.e. expected value range is
            around ``[-360, 360]``, with reasonable values being in the range of ``[-45, 45]``.
                * If a number, then that value will be used for all images as
                  the shear on the x-axis (no shear on the y-axis will be done).
                * If a tuple ``(a, b)``, then two value will be uniformly sampled per image
                  from the interval ``[a, b]`` and be used as the x- and y-shear value.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        interpolation (int): OpenCV interpolation flag.
        mask_interpolation (int): OpenCV interpolation flag.
        cval (number or sequence of number): The constant value to use when filling in newly created pixels.
            (E.g. translating by 1px to the right will create a new 1px-wide column of pixels
            on the left of the image).
            The value is only used when `mode=constant`. The expected value range is ``[0, 255]`` for ``uint8`` images.
        cval_mask (number or tuple of number): Same as cval but only for masks.
        mode (int): OpenCV border flag.
        fit_output (bool): If True, the image plane size and position will be adjusted to tightly capture
            the whole image after affine transformation (`translate_percent` and `translate_px` are ignored).
            Otherwise (``False``),  parts of the transformed image may end up outside the image plane.
            Fitting the output shape can be useful to avoid corners of the image being outside the image plane
            after applying rotations. Default: False
        keep_ratio (bool): When True, the original aspect ratio will be kept when the random scale is applied.
            Default: False.
        rotate_method (Literal["largest_box", "ellipse"]): rotation method used for the bounding boxes.
            Should be one of "largest_box" or "ellipse"[1]. Default: "largest_box"
        balanced_scale (bool): When True, scaling factors are chosen to be either entirely below or above 1,
            ensuring balanced scaling. Default: False.

            This is important because without it, scaling tends to lean towards upscaling. For example, if we want
            the image to zoom in and out by 2x, we may pick an interval [0.5, 2]. Since the interval [0.5, 1] is
            three times smaller than [1, 2], values above 1 are picked three times more often if sampled directly
            from [0.5, 2]. With `balanced_scale`, the  function ensures that half the time, the scaling
            factor is picked from below 1 (zooming out), and the other half from above 1 (zooming in).
            This makes the zooming in and out process more balanced.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32

    Reference:
        [1] https://arxiv.org/abs/2109.13488

    """

    _targets = (Targets.IMAGE, Targets.OBJECT3D)

    class InitSchema(BaseTransformInitSchema):
        scale: ScaleFloatType | dict[str, Any] | None = Field(
            default=None,
            description="Scaling factor or dictionary for independent axis scaling.",
        )
        shift: ScaleFloatType | dict[str, Any] | None = Field(
            default=None,
            description="Translation factor or dictionary for independent axis translation.",
        )

    def __init__(
        self,
        scale: ScaleFloatType | dict[str, Any] | None = None,
        shift: ScaleFloatType | dict[str, Any] | None = None,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        
        self.scale = scale
        self.shift = shift

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "scale",
            "shift",
        )

    def apply(
        self,
        img: np.ndarray,
        trans_inv: cv2.typing.MatLike,
        **params: Any,
    ) -> np.ndarray:
        img = Image.fromarray(img)
        img = img.transform(img.size,
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        img = np.array(img)
        return img

    def apply_to_object3d(
        self, 
        object3d: Namespace,
        trans: cv2.typing.MatLike,
        crop_scale: float,
        **params: Any,
    ) -> Namespace:
        object3d.box2d[:2] = affine_transform(object3d.box2d[:2], trans)
        object3d.box2d[2:] = affine_transform(object3d.box2d[2:], trans)
        object3d.center_3d = affine_transform(object3d.center_3d.reshape(-1), trans)
        object3d.pos[-1] = object3d.pos[-1] * crop_scale # depth
        object3d.trans = trans
        return object3d

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        height, width = params["image"].shape[:2]
        img_size = np.array([width, height])
        center = np.array([width, height]) / 2
        crop_size, crop_scale = img_size, 1
        
        if self.scale is not None:
            crop_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
            crop_size = img_size * crop_scale
        if self.shift is not None:
            center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
            center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
            
        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, img_size, inv=1)
        
        return {
            "trans": trans,
            "trans_inv": trans_inv,
            "crop_scale": crop_scale,
        }


class NoOp(CustomDualTransform):
    """Identity transform (does nothing).

    Targets:
        image, object3d
    """

    _targets = (Targets.IMAGE, Targets.OBJECT3D)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return img

    def apply_to_object3d(self, object3d: Namespace, **params: Any) -> Namespace:
        return object3d

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()
