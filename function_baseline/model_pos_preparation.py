from __future__ import print_function, absolute_import, division

import glob

import torch

from models_baseline.mlp.linear_model import LinearModel, init_weights
from models_baseline.videopose.model_VideoPose3D import TemporalModelOptimized1f,TemporalModel


def model_pos_preparation(args, dataset, device):
    """
    return a posenet Model: with Bx16x2 --> posenet --> Bx16x3
    """
    # Create model
    num_joints = dataset.skeleton().num_joints()   # num_joints = 16 fix
    print('create model: {}'.format(args.posenet_name))


    if args.posenet_name == 'videopose':
   
        if args.pad==0:
            filter_widths=[1,1,1,1]
        elif args.pad==1:
            filter_widths=[1,1,1,3]
        elif args.pad==13:
            filter_widths=[3,3,3]
        elif args.pad==40:
            filter_widths=[3,3,3,3]

        model_pos = TemporalModelOptimized1f(16, 2, 15, filter_widths=filter_widths, causal=False,
                                             dropout=0.25, channels=1024)
                                   
    else:
        assert False, 'posenet_name invalid'

    model_pos = model_pos.to(device)
    print("==> Total parameters for model {}: {:.2f}M"
          .format(args.posenet_name, sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    if args.pretrain:

        tmp_ckpt = torch.load(args.pretrain_path)
        model_pos.load_state_dict(tmp_ckpt['state_dict'])
        print('==> Pretrained posenet loaded')
    else:
        model_pos.apply(init_weights)

    return model_pos
