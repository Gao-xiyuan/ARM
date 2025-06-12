# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .MI_tool import normalized_conditional_mutual_information as ncmi
from .MI_tool import normalized_mutual_information as nmi
@MODELS.register_module()
class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2):
        super(GlobalAveragePooling, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_out = nn.Linear(1024, 512)
    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])


            #print("outs:", len(outs), outs[0].shape, outs[1].shape) #[4,512] [4,512]
            # outs_all = torch.cat((outs[0], outs[1]),dim=1)#Fusion outs[0]=img, outs[1]=audio
            # outs_all = self.fc_out(outs_all)
            # audio = outs[1].view(-1)
            # img = outs[0].view(-1)
            # cat = outs_all.view(-1)
            # #print("outs:", audio.shape, img.shape, cat.shape)
            # nmi_a = nmi(audio.cpu().detach().numpy(), cat.cpu().detach().numpy())
            # nmi_v = nmi(img.cpu().detach().numpy(), cat.cpu().detach().numpy())
            # # print("nmi_a, nmi_v:", nmi_a, nmi_v)
            # # print("(nmi_a/nmi_v), (nmi_v/nmi_a):",(nmi_a/nmi_v),(nmi_v/nmi_a))
            # cnmi_a = ncmi(audio.cpu().detach().numpy(), cat.cpu().detach().numpy(),img.cpu().detach().numpy(),bins=2)
            # cnmi_v = ncmi(img.cpu().detach().numpy(), cat.cpu().detach().numpy(),audio.cpu().detach().numpy(),bins=2)
            # #print("cnmi_a, cnmi_v:", cnmi_a, cnmi_v)
            # ii_a = nmi_a - cnmi_a
            # ii_v = nmi_v - cnmi_v
            # #print("ii_a, ii_v:", ii_a, ii_v)
            out = []
            outs_all = torch.cat((outs[0], outs[1]), dim=1)  # Fusion
            outs_all = self.fc_out(outs_all)
            out.append(outs_all)
            out.append(outs[0])
            out.append(outs[1])
            outs = tuple(out)
            #print("outs:",len(outs), outs[0].shape, outs[1].shape) #[4,512] [4,512]
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
