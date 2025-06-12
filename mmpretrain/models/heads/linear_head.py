# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .cls_head import ClsHead
from ..necks.MI_tool import normalized_conditional_mutual_information as ncmi
from ..necks.MI_tool import normalized_mutual_information as nmi

@MODELS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes


        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.fc_out = nn.Linear(2 * self.in_channels, self.in_channels)

    def pre_logits(self, feats: Tuple[torch.Tensor], data_samples) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``LinearClsHead``, we just obtain the
        feature of the last stage.
        """
        # The LinearClsHead doesn't have other module, just return after
        # unpacking.
        return feats[0]

    def forward(self, feats: Tuple[torch.Tensor], data_samples) -> torch.Tensor:
        """The forward process."""
        cls_scores = []
        cls_scores.append(self.fc(feats[0]))
        cls_scores.append(self.fc(feats[1]))
        cls_scores.append(self.fc(feats[2]))
        prediction = self.train_predict(feats, cls_scores, data_samples)

        cat = feats[0].view(-1)
        img = feats[1].view(-1)
        audio = feats[2].view(-1)
        nmi_a = nmi(audio.cpu().detach(), cat.cpu().detach())
        nmi_v = nmi(img.cpu().detach(), cat.cpu().detach())
        cnmi_a = ncmi(audio.cpu().detach(), cat.cpu().detach(), img.cpu().detach(), bins=2)
        cnmi_v = ncmi(img.cpu().detach(), cat.cpu().detach(), audio.cpu().detach(), bins=2)
        ii_a = nmi_a - cnmi_a
        ii_v = nmi_v - cnmi_v
        pcmi_a = prediction[0] * nmi_a + prediction[1] * ii_a
        pcmi_v = prediction[0] * nmi_v + prediction[2] * ii_v
        # print("(pcmi_a/pcmi_v), (pcmi_v/pcmi_a):", (pcmi_a/pcmi_v), (pcmi_v/pcmi_a))
        add = []
        out = torch.cat(((pcmi_a/pcmi_v)*feats[1], (pcmi_v/pcmi_a) * feats[2]), dim=1)
        out = self.fc_out(out)
        add.append(out)
        outs = tuple(add)
        # print('feats:',len(feats))1
        pre_logits = self.pre_logits(outs, data_samples)
        #pre_logits = (feats[1] + feats[2])/2
        # print('pre_logits:', pre_logits.shape)#[b,512]
        # The final classification head.
        cls_score = self.fc(pre_logits)
        # print('cls_score:', cls_score)#[b,num_class]
        return cls_score
