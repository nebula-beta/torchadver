#!/usr/bin/python2.7
#coding:utf-8


'''
experiment1:
    use random init or not
experiment2:
    explore the effect of different max_normA
experiment3:
    explore the effect of the different of num_iter
experiment4:
    explore the effect of the different of momentum
experiment5:
    explore the effect of the different of diversity_prob(and diversity_resize_rate)


LInf norm bound
FGM         : with random init or not
I_FGM       : with random init or not, different num_iter
MI_FGM      : with random init or not, different num_iter, different momentum
M_DI_FGM    : with random init or not, different num_iter, diffrernt momentum, different diversity_prob

同一个方法内部的比较，　不同方法之间的比较
'''

import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm
import numpy as np


from models import preact_resnet, densenet, inceptionresnetv2
from dataloader import cifar10

sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../')))

from torchadver.attacker.iterative_gradient_attack import FGM_L2, FGM_LInf, I_FGM_L2, I_FGM_LInf, MI_FGM_L2, MI_FGM_LInf, M_DI_FGM_L2, M_DI_FGM_LInf
from torchadver.utils import save_images, Evaluator, calc_max_norm, calc_average_norm


model_class_map = {
        'preact_resnet18' : (preact_resnet.PreActResNet18, './preact_resnet18.ckpt'),
        'densenet'        : (densenet.densenet121, './densenet121.ckpt'),
        'inceptionresnetv2': (inceptionresnetv2.inceptionresnetv2, './incresv2.ckpt'),
}


def get_attack_result(model, device, adversary, mean, std):
    # step1 : define model
    # ckpt_path = os.path.join(sys.path[0], './preact_resnet18.ckpt')
    # if not os.path.exists(ckpt_path):
    #     print("%s doesn't exists..." % ckpt_path)
    #     sys.exit()

    # if torch.cuda.is_available:
    #     device = torch.device('cuda:%d' % 0)
    # else:
    #     device = torch.device('cpu')

    # model = preact_resnet.PreActResNet18()
    # checkpoint = torch.load(ckpt_path, map_location=device)
    # model.load_state_dict(checkpoint)
    # model = model.to(device)

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
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]

    # adversary = FGM_L2(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=4.0, random_init=True)
    # adversary = FGM_LInf(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=0.1, random_init=True)
    # adversary = I_FGM_L2(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=4.0, random_init=True)
    # adversary = I_FGM_LInf(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=0.1, random_init=True)
    # adversary = MI_FGM_L2(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=4.0, random_init=True)
    # adversary = MI_FGM_LInf(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=0.1, random_init=True)
    # adversary = M_DI_FGM_L2(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=4.0, random_init=True)
    # adversary = M_DI_FGM_LInf(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, max_norm=0.1, random_init=True)


    # tbar = tqdm(train_loader)
    tbar = tqdm(val_loader)
    tot = 0
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

    return ori_top1, ori_top5, adv_top1, adv_top5





def _experiment_random_init(model_name):
    Model = model_class_map[model_name][0]
    ckpt_path = model_class_map[model_name][1]
    '''
    Dose random init lead to higher attack success rates?
    '''
    # step1 : define model
    # ckpt_path = os.path.join(sys.path[0], './preact_resnet18.ckpt')
    ckpt_path = os.path.join(sys.path[0], ckpt_path)
    if not os.path.exists(ckpt_path):
        print("%s doesn't exists..." % ckpt_path)
        sys.exit()

    if torch.cuda.is_available:
        device = torch.device('cuda:%d' % 0)
    else:
        device = torch.device('cpu')

    # model = preact_resnet.PreActResNet18()
    model = Model()
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # step4 : define attacker
    # mean and std used to normalize the images by loader
    # the attack method need these parameter to detemine the range of image pixels
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]


    experiment_setup_for_l2_bound = [
            ('FGM_L2 with random init',         FGM_L2,      {'max_norm' : 4.0, 'random_init' : True}),
            ('FGM_L2 without random_init',      FGM_L2,      {'max_norm' : 4.0, 'random_init' : False}),
            ('I_FGM_L2 with radnom init',       I_FGM_L2,    {'max_norm' : 4.0, 'random_init' : True}),
            ('I_FGM_L2 without radnom init',    I_FGM_L2,    {'max_norm' : 4.0, 'random_init' : False}),
            ('MI_FGM_L2 with radnom init',      MI_FGM_L2,   {'max_norm' : 4.0, 'random_init' : True}),
            ('MI_FGM_L2 without radnom init',   MI_FGM_L2,   {'max_norm' : 4.0, 'random_init' : False}),
            ('M_DI_FGM_L2 with radnom init',    M_DI_FGM_L2, {'max_norm' : 4.0, 'random_init' : True}),
            ('M_DI_FGM_L2 without radnom init', M_DI_FGM_L2, {'max_norm' : 4.0, 'random_init' : False}),
    ]


    experiment_setup_for_linf_bound = [
            ('FGM_LInf with random init',         FGM_LInf,      {'max_norm' : 0.1, 'random_init' : True}),
            ('FGM_LInf without random_init',      FGM_LInf,      {'max_norm' : 0.1, 'random_init' : False}),
            ('I_FGM_LInf with radnom init',       I_FGM_LInf,    {'max_norm' : 0.1, 'random_init' : True}),
            ('I_FGM_LInf without radnom init',    I_FGM_LInf,    {'max_norm' : 0.1, 'random_init' : False}),
            ('MI_FGM_LInf with radnom init',      MI_FGM_LInf,   {'max_norm' : 0.1, 'random_init' : True}),
            ('MI_FGM_LInf without radnom init',   MI_FGM_LInf,   {'max_norm' : 0.1, 'random_init' : False}),
            ('M_DI_FGM_LInf with radnom init',    M_DI_FGM_LInf, {'max_norm' : 0.1, 'random_init' : True}),
            ('M_DI_FGM_LInf without radnom init', M_DI_FGM_LInf, {'max_norm' : 0.1, 'random_init' : False}),
    ]

    experiment_results_for_l2_bound = []

    for experiment_name, Attacker, params in experiment_setup_for_l2_bound:
        print(experiment_name+'...')
        adversary = Attacker(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, **params)
        ori_top1, ori_top5, adv_top1, adv_top5 = \
        get_attack_result(model, device, adversary, mean, std)

        result = (experiment_name, ori_top1, ori_top5, adv_top1, adv_top5)
        experiment_results_for_l2_bound.append(result)


    experiment_results_for_linf_bound = []

    for experiment_name, Attacker, params in experiment_setup_for_linf_bound:
        print(experiment_name+'...')
        adversary = Attacker(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, **params)
        ori_top1, ori_top5, adv_top1, adv_top5 = \
        get_attack_result(model, device, adversary, mean, std)

        result = (experiment_name, ori_top1, ori_top5, adv_top1, adv_top5)
        experiment_results_for_linf_bound.append(result)

    return experiment_results_for_l2_bound, experiment_results_for_linf_bound


def experiment_random_init():
    for model_name in model_class_map.keys():
        print('Target model : ', model_name)
        experiment_results = _experiment_random_init(model_name)
        print(experiment_results)


def _experiment_different_max_norm(model_name):
    Model = model_class_map[model_name][0]
    ckpt_path = model_class_map[model_name][1]
    '''
    Dose random init lead to higher attack success rates?
    '''
    # step1 : define model
    # ckpt_path = os.path.join(sys.path[0], './preact_resnet18.ckpt')
    ckpt_path = os.path.join(sys.path[0], ckpt_path)
    if not os.path.exists(ckpt_path):
        print("%s doesn't exists..." % ckpt_path)
        sys.exit()

    if torch.cuda.is_available:
        device = torch.device('cuda:%d' % 0)
    else:
        device = torch.device('cpu')

    # model = preact_resnet.PreActResNet18()
    model = Model()
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # step4 : define attacker
    # mean and std used to normalize the images by loader
    # the attack method need these parameter to detemine the range of image pixels
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]



    experiment_setup_for_l2_bound = [
            ('FGM_L2',     FGM_L2,      [{'max_norm' : max_norm, 'random_init' : True} for max_norm in np.linspace(start=0.01, stop=0.1, num=19)]),
            ('I_FGM_L2',   I_FGM_L2,    [{'max_norm' : max_norm, 'random_init' : True} for max_norm in np.linspace(start=0.01, stop=0.1, num=19)]),
            ('MI_FGM_L2',  MI_FGM_L2,   [{'max_norm' : max_norm, 'random_init' : True} for max_norm in np.linspace(start=0.01, stop=0.1, num=19)]),
            ('M_DIFGM_L2', M_DI_FGM_L2, [{'max_norm' : max_norm, 'random_init' : True} for max_norm in np.linspace(start=0.01, stop=0.1, num=19)]),
    ]
    experiment_setup_for_linf_bound = [
            ('FGM_LInf',     FGM_LInf,      [{'max_norm' : max_norm, 'random_init' : True} for max_norm in np.linspace(start=0.1, stop=5.0, num=19)]),
            ('I_FGM_LInf',   I_FGM_LInf,    [{'max_norm' : max_norm, 'random_init' : True} for max_norm in np.linspace(start=0.1, stop=5.0, num=19)]),
            ('MI_FGM_LInf',  MI_FGM_LInf,   [{'max_norm' : max_norm, 'random_init' : True} for max_norm in np.linspace(start=0.1, stop=5.0, num=19)]),
            ('M_DIFGM_LInf', M_DI_FGM_LInf, [{'max_norm' : max_norm, 'random_init' : True} for max_norm in np.linspace(start=0.1, stop=5.0, num=19)]),
    ]

    experiment_results_for_l2_bound = []

    for experiment_name, Attacker, multi_params in experiment_setup_for_l2_bound:
        for params in multi_params:
            print(experiment_name + ' with max_l2_norm = %.3f' % (params['max_norm']))

            adversary = Attacker(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, **params)
            ori_top1, ori_top5, adv_top1, adv_top5 = \
            get_attack_result(model, device, adversary, mean, std)

            result = (experiment_name, params['max_norm'], ori_top1, ori_top5, adv_top1, adv_top5)
            experiment_results_for_l2_bound.append(result)

    experiment_results_for_linf_bound = []
    for experiment_name, Attacker, multi_params in experiment_setup_for_linf_bound:
        for params in multi_params:
            print(experiment_name + ' with max_linf_norm = %.3f' % (params['max_norm']))

            adversary = Attacker(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, **params)
            ori_top1, ori_top5, adv_top1, adv_top5 = \
            get_attack_result(model, device, adversary, mean, std)

            result = (experiment_name, params['max_norm'], ori_top1, ori_top5, adv_top1, adv_top5)
            experiment_results_for_linf_bound.append(result)

    return experiment_results_for_l2_bound, experiment_results_for_linf_bound



def experiment_different_max_norm():
    for model_name in model_class_map.keys():
        print('Target model : ', model_name)
        experiment_results = _experiment_different_max_norm(model_name)
        print(experiment_results)


def _experiment_different_num_iter(model_name):
    Model = model_class_map[model_name][0]
    ckpt_path = model_class_map[model_name][1]
    '''
    Dose random init lead to higher attack success rates?
    '''
    # step1 : define model
    # ckpt_path = os.path.join(sys.path[0], './preact_resnet18.ckpt')
    ckpt_path = os.path.join(sys.path[0], ckpt_path)
    if not os.path.exists(ckpt_path):
        print("%s doesn't exists..." % ckpt_path)
        sys.exit()

    if torch.cuda.is_available:
        device = torch.device('cuda:%d' % 0)
    else:
        device = torch.device('cpu')

    # model = preact_resnet.PreActResNet18()
    model = Model()
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # step4 : define attacker
    # mean and std used to normalize the images by loader
    # the attack method need these parameter to detemine the range of image pixels
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]



    experiment_setup_for_l2_bound = [
            ('I_FGM_L2',   I_FGM_L2,    [{'max_norm' : 4.0, 'num_iter' : num_iter, 'random_init' : True} for num_iter in range(1, 16)]),
    ]
    experiment_setup_for_linf_bound = [
            ('I_FGM_LInf',   I_FGM_LInf,    [{'max_norm' : 0.1, 'num_iter' : num_iter, 'random_init' : True} for num_iter in range(1, 16)]),
    ]

    experiment_results_for_l2_bound = []

    for experiment_name, Attacker, multi_params in experiment_setup_for_l2_bound:
        for params in multi_params:
            print(experiment_name + ' with num_iter = %d' % (params['num_iter']))

            adversary = Attacker(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, **params)
            ori_top1, ori_top5, adv_top1, adv_top5 = \
            get_attack_result(model, device, adversary, mean, std)

            result = (experiment_name, params['num_iter'], ori_top1, ori_top5, adv_top1, adv_top5)
            experiment_results_for_l2_bound.append(result)

    experiment_results_for_linf_bound = []
    for experiment_name, Attacker, multi_params in experiment_setup_for_linf_bound:
        for params in multi_params:
            print(experiment_name + ' with num_iter = %d' % (params['num_iter']))

            adversary = Attacker(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, **params)
            ori_top1, ori_top5, adv_top1, adv_top5 = \
            get_attack_result(model, device, adversary, mean, std)

            result = (experiment_name, params['num_iter'], ori_top1, ori_top5, adv_top1, adv_top5)
            experiment_results_for_linf_bound.append(result)

    return experiment_results_for_l2_bound, experiment_results_for_linf_bound

def experiment_different_num_iter():
    for model_name in model_class_map.keys():
        print('Target model : ', model_name)
        experiment_results = _experiment_different_num_iter(model_name)
        print(experiment_results)

def _experiment_different_momentum(model_name):
    Model = model_class_map[model_name][0]
    ckpt_path = model_class_map[model_name][1]
    '''
    Dose random init lead to higher attack success rates?
    '''
    # step1 : define model
    # ckpt_path = os.path.join(sys.path[0], './preact_resnet18.ckpt')
    ckpt_path = os.path.join(sys.path[0], ckpt_path)
    if not os.path.exists(ckpt_path):
        print("%s doesn't exists..." % ckpt_path)
        sys.exit()

    if torch.cuda.is_available:
        device = torch.device('cuda:%d' % 0)
    else:
        device = torch.device('cpu')

    # model = preact_resnet.PreActResNet18()
    model = Model()
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # step4 : define attacker
    # mean and std used to normalize the images by loader
    # the attack method need these parameter to detemine the range of image pixels
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]



    experiment_setup_for_l2_bound = [
            ('MI_FGM_L2',   MI_FGM_L2,    [{'max_norm' : 4.0, 'num_iter' : 10, 'momentum' : momentum, 'random_init' : True}
                                          for momentum in np.linspace(start=0, stop=1.0, num=11)]),
    ]
    experiment_setup_for_linf_bound = [
            ('MI_FGM_LInf',   MI_FGM_LInf,    [{'max_norm' : 0.1, 'num_iter' : 10, 'momentum' : momentum, 'random_init' : True} 
                                          for momentum in np.linspace(start=0, stop=1.0, num=11)]),
    ]

    experiment_results_for_l2_bound = []

    for experiment_name, Attacker, multi_params in experiment_setup_for_l2_bound:
        for params in multi_params:
            print(experiment_name + ' with momentum = %.2f' % (params['momentum']))

            adversary = Attacker(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, **params)
            ori_top1, ori_top5, adv_top1, adv_top5 = \
            get_attack_result(model, device, adversary, mean, std)

            result = (experiment_name, params['momentum'], ori_top1, ori_top5, adv_top1, adv_top5)
            experiment_results_for_l2_bound.append(result)

    experiment_results_for_linf_bound = []
    for experiment_name, Attacker, multi_params in experiment_setup_for_linf_bound:
        for params in multi_params:
            print(experiment_name + ' with momentum = %.2f' % (params['momentum']))

            adversary = Attacker(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, **params)
            ori_top1, ori_top5, adv_top1, adv_top5 = \
            get_attack_result(model, device, adversary, mean, std)

            result = (experiment_name, params['momentum'], ori_top1, ori_top5, adv_top1, adv_top5)
            experiment_results_for_linf_bound.append(result)

    return experiment_results_for_l2_bound, experiment_results_for_linf_bound

def experiment_different_momentum():
    for model_name in model_class_map.keys():
        print('Target model : ', model_name)
        experiment_results = _experiment_different_momentum(model_name)
        print(experiment_results)



def _experiment_different_diversity_prob(model_name):
    Model = model_class_map[model_name][0]
    ckpt_path = model_class_map[model_name][1]
    '''
    Dose random init lead to higher attack success rates?
    '''
    # step1 : define model
    # ckpt_path = os.path.join(sys.path[0], './preact_resnet18.ckpt')
    ckpt_path = os.path.join(sys.path[0], ckpt_path)
    if not os.path.exists(ckpt_path):
        print("%s doesn't exists..." % ckpt_path)
        sys.exit()

    if torch.cuda.is_available:
        device = torch.device('cuda:%d' % 0)
    else:
        device = torch.device('cpu')

    # model = preact_resnet.PreActResNet18()
    model = Model()
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # step4 : define attacker
    # mean and std used to normalize the images by loader
    # the attack method need these parameter to detemine the range of image pixels
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]



    experiment_setup_for_l2_bound = [
            ('M_DI_FGM_L2',   M_DI_FGM_L2,    [{'max_norm' : 4.0, 'num_iter' : 10, 'momentum' : 0.9,
                                                'diversity_prob' : diversity_prob, 'random_init' : True}
                                                for diversity_prob in np.linspace(start=0, stop=1.0, num=11)]),
    ]
    experiment_setup_for_linf_bound = [
            ('M_DI_FGM_LInf',   M_DI_FGM_LInf,    [{'max_norm' : 0.1, 'num_iter' : 10, 'momentum' : 0.9,
                                                   'diversity_prob' : diversity_prob, 'random_init' : True}
                                                    for diversity_prob in np.linspace(start=0, stop=1.0, num=11)]),
    ]

    experiment_results_for_l2_bound = []

    for experiment_name, Attacker, multi_params in experiment_setup_for_l2_bound:
        for params in multi_params:
            print(experiment_name + ' with diversity_prob = %.2f' % (params['diversity_prob']))

            adversary = Attacker(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, **params)
            ori_top1, ori_top5, adv_top1, adv_top5 = \
            get_attack_result(model, device, adversary, mean, std)

            result = (experiment_name, params['diversity_prob'], ori_top1, ori_top5, adv_top1, adv_top5)
            experiment_results_for_l2_bound.append(result)

    experiment_results_for_linf_bound = []
    for experiment_name, Attacker, multi_params in experiment_setup_for_linf_bound:
        for params in multi_params:
            print(experiment_name + ' with diversity_prob = %.2f' % (params['diversity_prob']))

            adversary = Attacker(model, loss_fn=nn.CrossEntropyLoss(), mean=mean, std=std, **params)
            ori_top1, ori_top5, adv_top1, adv_top5 = \
            get_attack_result(model, device, adversary, mean, std)

            result = (experiment_name, params['diversity_prob'], ori_top1, ori_top5, adv_top1, adv_top5)
            experiment_results_for_linf_bound.append(result)

    return experiment_results_for_l2_bound, experiment_results_for_linf_bound

def experiment_different_diversity_prob():
    for model_name in model_class_map.keys():
        print('Target model : ', model_name)
        experiment_results = _experiment_different_diversity_prob(model_name)
        print(experiment_results)
if __name__ == '__main__':
    '''
    experiment1:
        use random init or not
    experiment2:
        explore the effect of different max_normA
    experiment3:
        explore the effect of the different of num_iter
    experiment4:
        explore the effect of the different of momentum
    experiment5:
    explore the effect of the different of diversity_prob(and diversity_resize_rate)
    '''

    # experiment1
    # experiment_random_init()
    # experiment2
    # experiment_different_max_norm()
    # experiment3
    # experiment_different_num_iter()
    # experiment4
    # experiment_different_momentum()
    # experiment5
    experiment_different_diversity_prob()
