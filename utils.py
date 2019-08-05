#!/usr/bin/python2.7
#coding:utf-8

import os
import numpy as np
from PIL import Image

def save_images(images, mean, std, filenames, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # un-normlize to [0, 1]
    images = images * std + mean
    # convert image back to [0, 255]
    images = images * 255
    # [B, C, H, W] to [B, H, W, C]
    images = images.permute(0, 2, 3, 1)
    images = images.detach().cpu().numpy()
    images = images.astype(np.uint8)

    for filename, i in zip(filenames, range(images.shape[0])):
        image = images[i]
        Image.fromarray(image).save(os.path.join(output_dir, filename), format='PNG')





# https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840/3
class Evaluator(object):
    def __init__(self):
        self.top1 = []
        self.top5 = []
        self.batch_sizes = []


    def get_top1(self):
        n = np.sum(self.batch_sizes)
        return np.sum(np.array(self.top1) * np.array(self.batch_sizes) / n)

    def get_top5(self):
        n = np.sum(self.batch_sizes)
        return np.sum(np.array(self.top5) * np.array(self.batch_sizes) / n)


    def add_batch(self, logits, labels):

        batch_size = logits.shape[0]

        top1, top5 = self.topk(logits, labels, topk=(1, 5))

        self.top1.append(top1)
        self.top5.append(top5)
        self.batch_sizes.append(batch_size)

    def reset(self):
        self.top1 = []
        self.top5 = []
        self.batch_sizes = []

    def topk(self, logits, labels, topk=(1,)):
        maxk = max(topk)
        batch_size = logits.shape[0]

        _, preds = logits.topk(maxk, dim=1, largest=True, sorted=True)
        preds = preds.t()

        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            # res.append(correct_k)
            topk_res = correct_k.mul_(100.0 / batch_size)
            # topk_res = topk_res.detach().cpu().numpy()
            res.append(topk_res)
            # res.append(correct_k.mul_(100.0 / batch_size))

        return res
