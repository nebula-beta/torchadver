#!/usr/bin/python2.7
#coding:utf-8


import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm


from models import preact_resnet
from dataloader import cifar10

sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../')))

from torchadver.attacker.iterative_gradient_attack import FGM_L2, FGM_LInf, I_FGM_L2, I_FGM_LInf, MI_FGM_L2, MI_FGM_LInf, M_DI_FGM_L2, M_DI_FGM_LInf
from torchadver.utils import save_images, Evaluator, calc_max_norm, calc_average_norm

if __name__ == '__main__':

    # step1 : define model
    ckpt_path = os.path.join(sys.path[0], './preact_resnet18.ckpt')
    if not os.path.exists(ckpt_path):
        print("%s doesn't exists..." % ckpt_path)
        sys.exit()

    if torch.cuda.is_available:
        device = torch.device('cuda:%d' % 0)
    else:
        device = torch.device('cpu')

    model = preact_resnet.PreActResNet18()
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # step2 : define dataloader
    batch_size = 4
    num_workers = 4
    train_loader = cifar10.get_train_loader(batch_size, num_workers, shuffle=False)
    val_loader = cifar10.get_val_loader(batch_size, num_workers)

    # step3 : define evaluator
    ori_evaluator = Evaluator() # evaluate the accuracy before the attack
    adv_evaluator = Evaluator() # evaluate the accuracy after the attack


    # step4 : define attacker
    # mean and std used to normalize the images by loader
    # the attack method need these parameter to detemine the range of image pixels
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    # adversary = FGM_L2(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=4.0, random_init=True)
    adversary = FGM_LInf(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=0.1, random_init=True)
    # adversary = I_FGM_L2(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=4.0, random_init=True)
    # adversary = I_FGM_LInf(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=0.1, random_init=True)
    # adversary = MI_FGM_L2(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=4.0, random_init=True)
    # adversary = MI_FGM_LInf(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=0.1, random_init=True)
    # adversary = M_DI_FGM_L2(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=4.0, random_init=True)
    # adversary = M_DI_FGM_LInf(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=0.1, random_init=True)


    # tbar = tqdm(train_loader)
    tbar = tqdm(val_loader)
    tot = 0
    model.eval()
    for sample in tbar:

        images = sample[0].to(device)
        labels = sample[1].to(device)

        # step6 : evaluate the accuracy on the origin images
        logits = model(images)
        ori_evaluator.add_batch(logits, labels)


        # step7 : attack
        adv_images = adversary.attack(images, labels)
        # adv_images = adversary.attack(images)

        # step8 : evaluate the accuracy on the adverasial images
        logits = model(adv_images)
        adv_evaluator.add_batch(logits, labels)



        # step9 : save the adversarial images
        filenames = [ '%05d.png' % (tot + i) for i in range(batch_size) ]
        save_images(adv_images, mean, std, filenames, './adv_images')
        batch_size = images.shape[0]
        tot += batch_size


        # (optimal step) : print the attack result by per-batch
        # ori_top1 = ori_evaluator.get_top1()
        # ori_top5 = ori_evaluator.get_top5()
        # adv_top1 = adv_evaluator.get_top1()
        # adv_top5 = adv_evaluator.get_top5()
        # print('Before attack | top1 : %.4f, top5 : %.4f' % (ori_top1, ori_top5))
        # print('After attack  | top1 : %.4f, top5 : %.4f' % (adv_top1, adv_top5))


        # (optimal step) : check dose the norm satisfy the max norm constraint
        # max_l2_norm = calc_max_norm(images, adv_images, mean, std, 'l2')
        # max_linf_norm = calc_max_norm(images, adv_images, mean, std, 'linf')
        # print('max l2 norm   : ', max_l2_norm.item())
        # print('max linf norm : ', max_linf_norm.item())

        # (optimal step) : print the ave change between images and adv_images
        # ave_l2_norm = calc_average_norm(images, adv_images, mean, std, 'l2')
        # ave_linf_norm = calc_average_norm(images, adv_images, mean, std, 'linf')
        # print('ave l2 norm   : ', max_l2_norm.item())
        # print('ave linf norm : ', max_linf_norm.item())



    # step10 : print the comparison result
    ori_top1 = ori_evaluator.get_top1()
    ori_top5 = ori_evaluator.get_top5()
    adv_top1 = adv_evaluator.get_top1()
    adv_top5 = adv_evaluator.get_top5()
    print('Before attack | top1 : %.4f, top5 : %.4f' % (ori_top1, ori_top5))
    print('After attack  | top1 : %.4f, top5 : %.4f' % (adv_top1, adv_top5))

