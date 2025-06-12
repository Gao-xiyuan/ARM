from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
import os
import pickle
from PIL import Image
import cv2
import io
import torch
from torchvision import transforms
import torchvision.transforms.v2 as transformsv2
from torch.utils.data import Dataset
import numpy as np
import random
import copy
from mmengine import get_file_backend, list_from_file

@DATASETS.register_module()
class AV_KS_Dataset(BaseDataset):

    def __init__(self, data_root: str, split: str = 'train', **kwargs):

        splits = ['train', 'test']
        assert split in splits, \
            f"The split must be one of {splits}, but get '{split}'"
        self.split = split

        self.backend = get_file_backend(data_root, enable_singleton=True)

        test_mode = split == 'test'
        data_prefix = ''


        super(AV_KS_Dataset, self).__init__(
            data_root=data_root,
            test_mode=test_mode,
            data_prefix=data_prefix,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""
        pairs = list_from_file(self.ann_file)
        data_list = []
        if self.split == 'train':
            audio_prefix = '/root/dataset/ks/spec/train'
        else:
            audio_prefix = '/root/dataset/ks/spec/val'


        for pair in pairs:
            img_name, class_name = pair.split(' ')

            audio_path = audio_prefix + '/' + img_name + '.npy'

            img_name = f'{img_name}/frame_00000.jpg'

            img_path = self.backend.join_path(self.img_prefix,
                                              img_name)
            gt_label = int(class_name)

            info = dict(img_path=img_path, gt_label=gt_label, audio_path =audio_path)
            data_list.append(info)

        return data_list

