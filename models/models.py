import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import resnet18


def compute_joint_prob(x, y, z, bins):
    xyz = np.vstack([x, y, z]).T

    joint_counts, _ = np.histogramdd(xyz, bins=[bins, bins, bins])
    joint_prob = joint_counts / joint_counts.sum()
    return joint_prob

def compute_marginal_prob(joint_prob, axis):
    return joint_prob.sum(axis=axis)

def entropy(prob):
    prob = prob[prob > 0]  
    return -np.sum(prob * np.log(prob))

def mutual_information(x, y, bins=30):
    x = x.numpy() if isinstance(x, torch.Tensor) else x
    y = y.numpy() if isinstance(y, torch.Tensor) else y

    joint_prob = compute_joint_prob(x, y, bins)
    x_prob, y_prob = compute_marginal_prob(joint_prob)

    mi = 0.0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (x_prob[i] * y_prob[j]))

    return mi

def normalized_mutual_information(x, y, bins=30):
    mi = mutual_information(x, y, bins)
    joint_prob = compute_joint_prob(x.numpy() if isinstance(x, torch.Tensor) else x, 
                                    y.numpy() if isinstance(y, torch.Tensor) else y, 
                                    bins)
    x_prob, y_prob = compute_marginal_prob(joint_prob)
    
    hx = entropy(x_prob)
    hy = entropy(y_prob)
    
    nmi = 2 * mi / (hx + hy)
    return nmi

def conditional_entropy(joint_prob, z_prob):
   
    z_prob = np.where(z_prob > 0, z_prob, 1e-10)
    conditional_prob = joint_prob / z_prob[:, None, None]
    
    conditional_prob = conditional_prob[conditional_prob > 0]  
    return -np.sum(conditional_prob * np.log(conditional_prob))

def conditional_mutual_information(x, y, z, bins=10):
    x = x.numpy() if isinstance(x, torch.Tensor) else x
    y = y.numpy() if isinstance(y, torch.Tensor) else y
    z = z.numpy() if isinstance(z, torch.Tensor) else z
    
    joint_prob = compute_joint_prob(x, y, z, bins)
    xz_prob = compute_marginal_prob(joint_prob, axis=(1,))
    yz_prob = compute_marginal_prob(joint_prob, axis=(0,))
    z_prob = compute_marginal_prob(joint_prob, axis=(0, 1))

    hx_given_z = conditional_entropy(xz_prob, z_prob)
    hy_given_z = conditional_entropy(yz_prob, z_prob)
    hxy_given_z = conditional_entropy(joint_prob, z_prob)

    cmi = hx_given_z + hy_given_z - hxy_given_z
    print(f'hx_given_z:{hx_given_z}')
    print(f'hy_given_z:{hy_given_z}')
    print(f'hxy_given_z:{hxy_given_z}')
    return cmi, hxy_given_z

def normalized_conditional_mutual_information(x, y, z, bins=10):
    cmi, hxy_given_z = conditional_mutual_information(x, y, z, bins)
    if hxy_given_z == 0:
        return 0
    return cmi, hxy_given_z


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, out):
        # output = torch.cat((x, y), dim=1)
        output = self.fc_out(out)
        return output

class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        n_classes = args.n_classes

        self.fusion_module = ConcatFusion(output_dim=n_classes)

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

    def forward(self, audio, visual, drop = None, drop_arg = None):
        visual = visual.permute(0, 2, 1, 3, 4).contiguous()
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        if drop_arg != None:
            if self.__dict__['training'] and drop_arg.warmup == 0:
                self.p = drop_arg.p
                out, update_flag = self.execute_drop([a, v], self.p)
                self.update = update_flag
                self.update = torch.Tensor(self.update).cuda()
                out = torch.cat((a,v),1)
                out = self.fusion_module(out)
                return a,v,out,self.update

            else:
                out = torch.cat((a,v),1)
                out = self.fusion_module(out)
                # self.update = [1] * B
                return a,v,out
        
        else:
            if drop != None:
                for i in range(len(drop)):
                    if drop[i] == 1:
                        a[i,:] = 0.0
                    elif drop[i] == 2:
                        v[i,:] = 0.0

            out = torch.cat((a,v),1)
            out = self.fusion_module(out)

            return a, v, out
    
    def exec_drop(self, a, v, drop):
        if drop == 'audio':
            ad = torch.zeros_like(a)
            vd = v
        
        else:
            ad = a
            vd = torch.zeros_like(v)
        
        out = torch.cat((ad,vd),1)
        out = self.fusion_module(out)

        return out