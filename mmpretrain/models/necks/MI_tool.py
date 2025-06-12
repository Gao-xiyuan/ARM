import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_joint_prob_cmi(x, y, z, bins):
    xyz = np.vstack([x, y, z]).T

    joint_counts, _ = np.histogramdd(xyz, bins=[bins, bins, bins])
    joint_prob = joint_counts / joint_counts.sum()
    return joint_prob


def compute_marginal_prob_cmi(joint_prob, axis):
    return joint_prob.sum(axis=axis)


def entropy(prob):
    prob = prob[prob > 0]  # 只考虑非零概率
    return -np.sum(prob * np.log(prob))


def compute_joint_prob(x, y, bins):
    joint_counts = np.histogram2d(x, y, bins=bins)[0]
    joint_prob = joint_counts / joint_counts.sum()
    return joint_prob


def compute_marginal_prob(joint_prob):
    x_prob = joint_prob.sum(axis=1)
    y_prob = joint_prob.sum(axis=0)
    return x_prob, y_prob

def conditional_entropy(joint_prob, z_prob):
    # 将所有小于等于0的z_prob值替换为一个非常小的数值，避免除以零
    z_prob = np.where(z_prob > 0, z_prob, 1e-10)
    conditional_prob = joint_prob / z_prob[:, None, None]

    conditional_prob = conditional_prob[conditional_prob > 0]  # 只考虑非零概率
    return -np.sum(conditional_prob * np.log(conditional_prob))

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

def conditional_mutual_information(x, y, z, bins=30):
    x = x.numpy() if isinstance(x, torch.Tensor) else x
    y = y.numpy() if isinstance(y, torch.Tensor) else y
    z = z.numpy() if isinstance(z, torch.Tensor) else z

    joint_prob = compute_joint_prob_cmi(x, y, z, bins)
    xz_prob = compute_marginal_prob_cmi(joint_prob, axis=(1,))
    yz_prob = compute_marginal_prob_cmi(joint_prob, axis=(0,))
    z_prob = compute_marginal_prob_cmi(joint_prob, axis=(0, 1))

    hx_given_z = conditional_entropy(xz_prob, z_prob)
    hy_given_z = conditional_entropy(yz_prob, z_prob)
    hxy_given_z = conditional_entropy(joint_prob, z_prob)

    cmi = hx_given_z + hy_given_z - hxy_given_z
    # print(f'hx_given_z:{hx_given_z}')
    # print(f'hy_given_z:{hy_given_z}')
    #print(f'hxy_given_z:{hxy_given_z}')
    return cmi, hxy_given_z


def normalized_conditional_mutual_information(x, y, z, bins=30):
    cmi, hxy_given_z = conditional_mutual_information(x, y, z, bins)

    # 重新计算这两个熵用于归一化
    joint_prob = compute_joint_prob_cmi(x, y, z, bins)
    xz_prob = compute_marginal_prob_cmi(joint_prob, axis=(1,))
    yz_prob = compute_marginal_prob_cmi(joint_prob, axis=(0,))
    z_prob = compute_marginal_prob_cmi(joint_prob, axis=(0, 1))

    hx_given_z = conditional_entropy(xz_prob, z_prob)
    hy_given_z = conditional_entropy(yz_prob, z_prob)

    denominator = hx_given_z + hy_given_z
    normalized_cmi = (cmi) / denominator if denominator != 0 else 0
    return normalized_cmi
