#!/usr/bin/python2.7
#coding:utf-8

from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# from .utils import random_init_delta, clamp_by_l2_norm, input_diversity, get_clip_range

from .base_attack import Attack

def input_diversity(x, resize_rate=1.10, diversity_prob=0.3):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0

    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    # print(img_size, img_resize, resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

    ret = padded if torch.rand(1) < diversity_prob else x
    return ret

def random_init_delta(x, norm_type, max_norm, mean, std):
    delta = torch.zeros_like(x)

    # mean = torch.Tensor(mean).reshape([3, 1, 1]).cuda()
    # std = torch.Tensor(std).reshape([3, 1, 1]).cuda()
    # mean = torch.Tensor(mean).reshape([3, 1, 1])
    # std = torch.Tensor(std).reshape([3, 1, 1])

    if norm_type == 'l1':
        pass
    elif norm_type == 'l2':
        delta.data.uniform_(-1.0, 1.0)
        delta.data = (delta.data - mean) / std
        delta.data = delta.data - x
        delta.data = clamp_by_l2_norm(delta.data, max_norm)
    elif norm_type == 'linf':
        delta.data.uniform_(-1.0, 1.0)
        delta.data = delta.data * max_norm

    return delta

def clamp_by_l2_norm(delta, max_norm):
    batch_size = delta.shape[0]
    norm = torch.reshape(torch.norm(torch.reshape(delta, shape=[batch_size, -1]), dim=1), shape=[batch_size, 1, 1, 1])
    # if max_norm > norm, mean current l2 norm of delta statisfy the constraint, no need the rescale
    factor = torch.min(max_norm / norm, torch.ones_like(norm))
    delta = delta * factor
    return delta


def iterative_gradient_attack(model, loss_fn, targeted, x, y,
                             norm_type, max_norm, max_norm_per_iter,
                             num_iter, momentum,
                             diversity_resize_rate, diversity_prob, random_init,
                             mean, std, clip_min, clip_max,):
    '''
        Explaining and harnessing adversarial examples. Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. ICLR(Poster) 2015.
        Adversarial examples in the physical world. Kurakin, Alexey, Ian Goodfellow, and Samy Bengio. ICLR(Workshop) 2017.
        Boosting Adversarial Attacks with Momentum. Dong Y , Liao F , Pang T , et al. CVPR 2017.
        Improving Transferability of Adversarial Examples with Input Diversity. Xie, Cihang, et al. CVPR 2019.


        model                   : classification model you want to fool
        loss_fn                 : loss function you want to maxmize or minimize for specific class y(see below)
        targeted                : targeted attack or non-targeted attack
        x                       : input images you want to attack
        y                       : you want to maxmize or minimize loss for specific class y.
                                  For non-targeted attack, maxmize the loss for class y.
                                  For targeted attack, minimize the loss for class y
        norm_type               : l1, l2, linf norm, you can the pertubation satisfy the max norm constratint
        max_norm                : the max norm constarint of the pertubation
        max_norm_per_iter       : the max norm change in each iteration
        num_iter                : number of the iteration attacks
        momentum                : momentum
        diversity_resize_rate   : image rezie rate
        diversity_prob          : probability for using input diversity
        random_init             : random init the init pertubation
        mean                    : mean for dataset
        std                     : std for dataset
        clip_min                : clip min
        ckip_max                : clip max
    '''
    # print('max_norm : ' , max_norm)
    # print('mean : ', mean)
    # print('std : ', std)
    batch_size = x.shape[0]

    if random_init:
        delta = random_init_delta(x, norm_type, max_norm, mean, std)
    else:
        delta = torch.zeros_like(x)

    if y is None:
        logits = model(x)
        y = logits.max(1)[1]

    if targeted:
        scaler = -1
    else:
        scaler = 1

    grad = torch.zeros_like(x)

    for i in range(num_iter):
        delta = delta.detach()
        delta.requires_grad = True

        x_diversity = input_diversity(x + delta, diversity_resize_rate, diversity_prob=diversity_prob)

        logits = model(x_diversity)
        loss = scaler * loss_fn(logits, y)
        loss.backward()

        noise = delta.grad

        if norm_type == 'l1':
            pass
        elif norm_type == 'l2':
            # noise = noise / torch.reshape(torch.mean(torch.abs(torch.reshape(noise, shape=[batch_size, -1])), dim=1), shape=[batch_size, 1, 1, 1])
            # noise = noise / torch.reshape(torch.std(torch.reshape(noise, shape=[batch_size, -1]), dim=1), shape=[batch_size, 1, 1, 1])
            noise = noise / torch.reshape(torch.norm(torch.reshape(noise, shape=[batch_size, -1]), dim=1), shape=[batch_size, 1, 1, 1])
            grad = grad * momentum + noise
            # noise = noise / torch.reshape(torch.std(torch.reshape(noise, shape=[batch_size, -1]), dim=1), shape=[batch_size, 1, 1, 1])
            noise = grad / torch.reshape(torch.norm(torch.reshape(grad, shape=[batch_size, -1]), dim=1), shape=[batch_size, 1, 1, 1])

            # constraint1 : force to satisfy the max norm constaint
            # delta = delta + max_norm_per_iter * noise
            delta = delta + 2 * max_norm_per_iter * noise
            # constarint2 : force ot satisfy the image range constaint
            delta = clamp_by_l2_norm(delta, max_norm)

        elif norm_type == 'linf':

            grad = grad * momentum + noise
            noise = torch.sign(grad)
            # constraint1 : force to satisfy the max norm constaint
            delta = delta.data + max_norm_per_iter * noise
            delta = torch.max(torch.min(delta, max_norm), -max_norm) # if use random init, then need this to enforce satisfy the constraint1
            # constarint2 : force ot satisfy the image range constaint
            delta = torch.max(torch.min(x + delta, clip_max), clip_min) - x

        else:
            raise NotImplementedError('norm_type only can be l1, l2, linf...')

    return delta



# class Attack(object):
#     def __init__(self, model, loss_fn, mean, std, norm_type, max_norm, targeted=False):
#         super(Attack, self).__init__()

#         mean = np.array(mean)
#         std = np.array(std)
#         clip_min = (0 - mean) / std
#         clip_max = (1 - mean) / std

#         channel = mean.shape[0]

#         mean = torch.Tensor(mean).reshape([channel, 1, 1])
#         std = torch.Tensor(std).reshape([channel, 1, 1])
#         clip_min = torch.Tensor(clip_min).reshape([channel, 1, 1])
#         clip_max = torch.Tensor(clip_max).reshape([channel, 1, 1])
#         expand_max_norm = max_norm / std

#         # if model.is_cuda:
#         if True:
#             mean = mean.cuda()
#             std = std.cuda()
#             clip_min = clip_min.cuda()
#             clip_max = clip_max.cuda()
#             expand_max_norm = expand_max_norm.cuda()


#         self.model = model
#         self.loss_fn = loss_fn
#         self.mean = mean
#         self.std = std
#         self.clip_min = clip_min
#         self.clip_max = clip_max
#         self.norm_type = norm_type
#         self.max_norm = max_norm
#         self.expand_max_norm = expand_max_norm
#         self.targeted = targeted


#     def attack(self, x, y=None):

#         error = "Sub-classes must implement perturb."
#         raise NotImplementedError(error)




class FastGradientMethod_L2(Attack):
    def __init__(self, model, loss_fn, mean, std, max_norm=4.0, random_init=False, targeted=False):
        norm_type = 'l2'
        super(FastGradientMethod_L2, self).__init__(model, loss_fn, mean, std, norm_type, max_norm, targeted)
        self.random_init = random_init
        self.norm_per_iter = 2 * self.expand_max_norm

    def attack(self, x, y=None):
        num_iter = 1
        momentum = 0.0
        diversity_resize_rate = 1.10
        diversity_prob = 0.0
        delta = iterative_gradient_attack(self.model, self.loss_fn, self.targeted,
                                          x, y, self.norm_type, self.expand_max_norm,
                                          self.norm_per_iter, num_iter, momentum,
                                          diversity_resize_rate, diversity_prob,
                                          self.random_init, self.mean, self.std, self.clip_min, self.clip_max)

        return x + delta


class FastGradientMethod_LInf(Attack):
    def __init__(self, model, loss_fn, mean, std, max_norm=0.1, random_init=False, targeted=False):
        norm_type = 'linf'
        super(FastGradientMethod_LInf, self).__init__(model, loss_fn, mean, std, norm_type, max_norm, targeted)
        self.random_init = random_init
        self.norm_per_iter = self.expand_max_norm

    def attack(self, x, y=None):
        num_iter = 1
        momentum = 0.0
        diversity_resize_rate = 1.10
        diversity_prob = 0.0
        delta = iterative_gradient_attack(self.model, self.loss_fn, self.targeted,
                                          x, y, self.norm_type, self.expand_max_norm,
                                          self.norm_per_iter, num_iter, momentum,
                                          diversity_resize_rate, diversity_prob,
                                          self.random_init, self.mean, self.std, self.clip_min, self.clip_max)

        return x + delta




class IterativeFastGradientMethod_L2(Attack):
    def __init__(self, model, loss_fn, mean, std, max_norm=4.0, num_iter=10, random_init=False, targeted=False):
        norm_type = 'l2'
        super(IterativeFastGradientMethod_L2, self).__init__(model, loss_fn, mean, std, norm_type, max_norm, targeted)
        self.num_iter = num_iter
        self.random_init = random_init
        self.norm_per_iter = 2 * self.expand_max_norm / self.num_iter

    def attack(self, x, y=None):
        momentum = 0.0
        diversity_resize_rate = 1.10
        diversity_prob = 0.0
        delta = iterative_gradient_attack(self.model, self.loss_fn, self.targeted,
                                          x, y, self.norm_type, self.expand_max_norm,
                                          self.norm_per_iter, self.num_iter, momentum,
                                          diversity_resize_rate, diversity_prob,
                                          self.random_init, self.mean, self.std, self.clip_min, self.clip_max)

        return x + delta

class IterativeFastGradientMethod_LInf(Attack):
    def __init__(self, model, loss_fn, mean, std, max_norm=0.1, num_iter=10, random_init=False, targeted=False):
        norm_type = 'linf'
        super(IterativeFastGradientMethod_LInf, self).__init__(model, loss_fn, mean, std, norm_type, max_norm, targeted)
        self.num_iter = num_iter
        self.random_init = random_init
        self.norm_per_iter = self.expand_max_norm / self.num_iter

    def attack(self, x, y=None):
        momentum = 0.0
        diversity_resize_rate = 1.10
        diversity_prob = 0.0
        delta = iterative_gradient_attack(self.model, self.loss_fn, self.targeted,
                                          x, y, self.norm_type, self.expand_max_norm,
                                          self.norm_per_iter, self.num_iter, momentum,
                                          diversity_resize_rate, diversity_prob,
                                          self.random_init, self.mean, self.std, self.clip_min, self.clip_max)

        return x + delta


class MomentumIterativeFastGradientMethod_L2(Attack):
    def __init__(self, model, loss_fn, mean, std, max_norm=4.0, num_iter=10, momentum=0.9, random_init=False, targeted=False):
        norm_type = 'l2'
        super(MomentumIterativeFastGradientMethod_L2, self).__init__(model, loss_fn, mean, std, norm_type, max_norm, targeted)
        self.num_iter = num_iter
        self.momentum = momentum
        self.random_init = random_init
        self.norm_per_iter = 2 * self.expand_max_norm / self.num_iter

    def attack(self, x, y=None):
        diversity_resize_rate = 1.10
        diversity_prob = 0.0
        delta = iterative_gradient_attack(self.model, self.loss_fn, self.targeted,
                                          x, y, self.norm_type, self.expand_max_norm,
                                          self.norm_per_iter, self.num_iter, self.momentum,
                                          diversity_resize_rate, diversity_prob,
                                          self.random_init, self.mean, self.std, self.clip_min, self.clip_max)

        return x + delta


class MomentumIterativeFastGradientMethod_LInf(Attack):
    def __init__(self, model, loss_fn, mean, std, max_norm=0.1, num_iter=10, momentum=0.9, random_init=False, targeted=False):
        norm_type = 'linf'
        super(MomentumIterativeFastGradientMethod_LInf, self).__init__(model, loss_fn, mean, std, norm_type, max_norm, targeted)
        self.num_iter = num_iter
        self.momentum = momentum
        self.random_init = random_init
        self.norm_per_iter = self.expand_max_norm / self.num_iter

    def attack(self, x, y=None):
        diversity_resize_rate = 1.10
        diversity_prob = 0.0
        delta = iterative_gradient_attack(self.model, self.loss_fn, self.targeted,
                                          x, y, self.norm_type, self.expand_max_norm,
                                          self.norm_per_iter, self.num_iter, self.momentum,
                                          diversity_resize_rate, diversity_prob,
                                          self.random_init, self.mean, self.std, self.clip_min, self.clip_max)

        return x + delta



class MomentumDiversityIterativeFastGradientMethod_L2(Attack):
    def __init__(self, model, loss_fn, mean, std, max_norm=4.0, num_iter=10, momentum=0.9,
            diversity_resize_rate=1.10, diversity_prob=0.3, random_init=False, targeted=False):

        norm_type = 'l2'
        super(MomentumDiversityIterativeFastGradientMethod_L2, self).__init__(model, loss_fn, mean, std, norm_type, max_norm, targeted)
        self.num_iter = num_iter
        self.momentum = momentum
        self.diversity_resize_rate = diversity_resize_rate
        self.diversity_prob = diversity_prob
        self.random_init = random_init
        self.norm_per_iter = 2 * self.expand_max_norm / self.num_iter

    def attack(self, x, y=None):
        delta = iterative_gradient_attack(self.model, self.loss_fn, self.targeted,
                                          x, y, self.norm_type, self.expand_max_norm,
                                          self.norm_per_iter, self.num_iter, self.momentum,
                                          self.diversity_resize_rate, self.diversity_prob,
                                          self.random_init, self.mean, self.std, self.clip_min, self.clip_max)

        return x + delta

class MomentumDiversityIterativeFastGradientMethod_LInf(Attack):
    def __init__(self, model, loss_fn, mean, std, max_norm=4.0, num_iter=10, momentum=0.9,
            diversity_resize_rate=1.10, diversity_prob=0.3, random_init=False, targeted=False):
        norm_type = 'linf'
        super(MomentumDiversityIterativeFastGradientMethod_LInf, self).__init__(model, loss_fn, mean, std, norm_type, max_norm, targeted)
        self.num_iter = num_iter
        self.momentum = momentum
        self.diversity_resize_rate = diversity_resize_rate
        self.diversity_prob = diversity_prob
        self.random_init = random_init
        self.norm_per_iter = self.expand_max_norm / self.num_iter

    def attack(self, x, y=None):
        delta = iterative_gradient_attack(self.model, self.loss_fn, self.targeted,
                                          x, y, self.norm_type, self.expand_max_norm,
                                          self.norm_per_iter, self.num_iter, self.momentum,
                                          self.diversity_resize_rate, self.diversity_prob,
                                          self.random_init, self.mean, self.std, self.clip_min, self.clip_max)

        return x + delta



FGM_L2   = FastGradientMethod_L2
FGM_LInf = FastGradientMethod_LInf

I_FGM_L2   = IterativeFastGradientMethod_L2
I_FGM_LInf = IterativeFastGradientMethod_LInf


MI_FGM_L2   = MomentumIterativeFastGradientMethod_L2
MI_FGM_LInf = MomentumIterativeFastGradientMethod_LInf

M_DI_FGM_L2   = MomentumDiversityIterativeFastGradientMethod_L2
M_DI_FGM_LInf = MomentumDiversityIterativeFastGradientMethod_LInf




