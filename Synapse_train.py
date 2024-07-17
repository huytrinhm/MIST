#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import time
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from trainer import trainer_synapse
from lib.networks import MIST_CAM


import gc
gc.collect()
torch.cuda.empty_cache()


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/synapse/train_npz_new', help='root dir for data')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate') #0.001
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input') #224
parser.add_argument('--snapshot_path', type=str,
                    default='./model/experiment/', help='snapshot_path')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='z_spacing')
parser.add_argument('--is_pretrain', type=bool,
                    default=True, help='is_pretrain')
parser.add_argument('--pretrained_path', type=str,
                    default='maxxvit_rmlp_small_rw_256_sw-37e217ff.pth', help='pretrained_path')
args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)
    
    # img_size_s2 has no effect :)
    net = MIST_CAM(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear', args.pretrained_path).cuda()

    
    print('Model %s created, param count: %d' %
                     ('MIST_CAM: ', sum([m.numel() for m in net.parameters()])))

    net = net.cuda()
   
    trainer_synapse(args, net, args.snapshot_path)
