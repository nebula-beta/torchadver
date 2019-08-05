#!/usr/bin/python2.7
#coding:utf-8

import numpy as np
import torch


class Attack(object):
    def __init__(self, model, loss_fn, mean, std, norm_type, max_norm, targeted=False):
        super(Attack, self).__init__()

        mean = np.array(mean)
        std = np.array(std)
        clip_min = (0 - mean) / std
        clip_max = (1 - mean) / std

        channel = mean.shape[0]

        mean = torch.Tensor(mean).reshape([channel, 1, 1])
        std = torch.Tensor(std).reshape([channel, 1, 1])
        clip_min = torch.Tensor(clip_min).reshape([channel, 1, 1])
        clip_max = torch.Tensor(clip_max).reshape([channel, 1, 1])
        expand_max_norm = max_norm / std

        if next(model.parameters()).is_cuda:
            mean = mean.cuda()
            std = std.cuda()
            clip_min = clip_min.cuda()
            clip_max = clip_max.cuda()
            expand_max_norm = expand_max_norm.cuda()


        self.model = model
        self.loss_fn = loss_fn
        self.mean = mean
        self.std = std
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.norm_type = norm_type
        self.max_norm = max_norm
        self.expand_max_norm = expand_max_norm
        self.targeted = targeted


    def attack(self, x, y=None):

        error = "Sub-classes must implement perturb."
        raise NotImplementedError(error)
