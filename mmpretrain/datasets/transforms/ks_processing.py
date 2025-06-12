import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
from enum import Enum
import torch
from torch import Tensor
from mmcv.transforms.utils import cache_randomness
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
import mmengine.fileio as fileio
import numpy as np

import mmcv
from mmpretrain.registry import TRANSFORMS
from torchvision import transforms

@TRANSFORMS.register_module()
class MyLoadImageFromFile(BaseTransform):

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        audio = np.load(results['audio_path'])
        audio = np.expand_dims(audio, axis=-1)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['audio'] = audio
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class MyRandomResizedCrop(BaseTransform):

    def __init__(self,
                 scale: Union[Sequence, int],
                 crop_ratio_range: Tuple[float, float] = (0.08, 1.0),
                 aspect_ratio_range: Tuple[float, float] = (3. / 4., 4. / 3.),
                 max_attempts: int = 10,
                 interpolation: str = 'bilinear',
                 backend: str = 'cv2') -> None:
        if isinstance(scale, Sequence):
            assert len(scale) == 2
            assert scale[0] > 0 and scale[1] > 0
            self.scale = scale
        else:
            assert scale > 0
            self.scale = (scale, scale)
        if (crop_ratio_range[0] > crop_ratio_range[1]) or (
                aspect_ratio_range[0] > aspect_ratio_range[1]):
            raise ValueError(
                'range should be of kind (min, max). '
                f'But received crop_ratio_range {crop_ratio_range} '
                f'and aspect_ratio_range {aspect_ratio_range}.')
        assert isinstance(max_attempts, int) and max_attempts >= 0, \
            'max_attempts mush be int and no less than 0.'
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')

        self.crop_ratio_range = crop_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.max_attempts = max_attempts
        self.interpolation = interpolation
        self.backend = backend

    @cache_randomness
    def rand_crop_params(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to `crop` for a random sized crop.
        """
        h, w = img.shape[:2]
        area = h * w

        for _ in range(self.max_attempts):
            target_area = np.random.uniform(*self.crop_ratio_range) * area
            log_ratio = (math.log(self.aspect_ratio_range[0]),
                         math.log(self.aspect_ratio_range[1]))
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))
            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_w <= w and 0 < target_h <= h:
                offset_h = np.random.randint(0, h - target_h + 1)
                offset_w = np.random.randint(0, w - target_w + 1)

                return offset_h, offset_w, target_h, target_w

        # Fallback to central crop
        in_ratio = float(w) / float(h)
        if in_ratio < min(self.aspect_ratio_range):
            target_w = w
            target_h = int(round(target_w / min(self.aspect_ratio_range)))
        elif in_ratio > max(self.aspect_ratio_range):
            target_h = h
            target_w = int(round(target_h * max(self.aspect_ratio_range)))
        else:  # whole image
            target_w = w
            target_h = h
        offset_h = (h - target_h) // 2
        offset_w = (w - target_w) // 2
        return offset_h, offset_w, target_h, target_w

    def transform(self, results: dict) -> dict:
        """Transform function to randomly resized crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly resized cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        img = results['img']
        offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
        print(offset_h, offset_w, target_h, target_w)
        img = mmcv.imcrop(
            img,
            bboxes=np.array([
                offset_w, offset_h, offset_w + target_w - 1,
                offset_h + target_h - 1
            ]))
        img = mmcv.imresize(
            img,
            tuple(self.scale[::-1]),
            interpolation=self.interpolation,
            backend=self.backend)
        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(scale={self.scale}'
        repr_str += ', crop_ratio_range='
        repr_str += f'{tuple(round(s, 4) for s in self.crop_ratio_range)}'
        repr_str += ', aspect_ratio_range='
        repr_str += f'{tuple(round(r, 4) for r in self.aspect_ratio_range)}'
        repr_str += f', max_attempts={self.max_attempts}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str