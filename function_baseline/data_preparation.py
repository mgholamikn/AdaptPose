from __future__ import print_function, absolute_import, division

import os.path as path

import numpy as np
from torch.utils.data import DataLoader
from common.viz import *
from common.data_loader import PoseDataSet, PoseBuffer
from utils.data_utils import *



def data_preparation(args):
    ############################################
    # load dataset
    ############################################
    dataset_path = path.join('data', 'data_3d_' + args.dataset + '.npz')
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset, TEST_SUBJECTS
        dataset = Human36mDataset(dataset_path)
        if args.s1only:
            subjects_train = ['S1']
        else:
            subjects_train = ['S1', 'S5', 'S6', 'S7', 'S8']
        subjects_test = TEST_SUBJECTS
    else:
        raise KeyError('Invalid dataset')

    print('==> Preparing data...')
    dataset = read_3d_data(dataset)

    print('==> Loading 2D detections...')
    keypoints = create_2d_data(path.join('data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset,tag=args.dataset_target)

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
        print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample

    ############################################
    # general 2D-3D pair dataset
    ############################################
    poses_train, poses_train_2d, actions_train, cams_train = fetch(subjects_train, dataset, keypoints, action_filter,
                                                                   stride)
    poses_valid, poses_valid_2d, actions_valid, cams_valid = fetch(subjects_test, dataset, keypoints, action_filter,
                                                                   stride)

    poses_train, poses_train_2d,cams_train=np.concatenate(poses_train), np.concatenate(poses_train_2d),np.concatenate(cams_train)
    poses_valid, poses_valid_2d,cams_valid=np.concatenate(poses_valid), np.concatenate(poses_valid_2d),np.concatenate(cams_valid)


    train_loader = DataLoader(PoseDataSet(poses_train, poses_train_2d, actions_train, cams_train, pad=args.pad),
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(PoseDataSet(poses_valid, poses_valid_2d, actions_valid, cams_valid, pad=args.pad),
                              batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    ############################################
    # prepare cross dataset validation
    ############################################
    if args.dataset_target == '3dhp' or args.dataset_target =='h36m':
        keypoints_test=create_2d_data_3dhp_test('data/data_3dhp_gt_test.npz')
        mpi3dhp_3d,mpi3dhp_2d=fetch_3dhp('data/data_3dhp_gt_test.npz',keypoints_test)
        mpi3dhp_3d,mpi3dhp_2d=np.concatenate(mpi3dhp_3d),np.concatenate(mpi3dhp_2d)
        print('test_shape',mpi3dhp_3d.shape)
        test_loader = DataLoader(PoseBuffer(mpi3dhp_3d, mpi3dhp_2d,pad=args.pad),
                                batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)

    elif args.dataset_target == '3dpw':
        keypoints_test=np.load(path.join('data', 'data_'+args.dataset_target + '_' + args.keypoints_target + '_test.npz'))
        _3dpw_3d, _3dpw_2d = keypoints_test['positions_3d'],keypoints_test['positions_2d']
        _3dpw_3d=_3dpw_3d-_3dpw_3d[:,:1]
        joints=[0,4,5,6,1,2,3,7,8,9,10,11,12,13,14,15]
        _3dpw_3d, _3dpw_2d= _3dpw_3d[:,joints], _3dpw_2d[:,joints]
        print('test_shape',_3dpw_2d.shape)
        test_loader = DataLoader(PoseBuffer(_3dpw_3d, _3dpw_2d,pad=args.pad),
                                batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)


    elif args.dataset_target == 'skii':
        keypoints_test=np.load(path.join('data', 'data_'+args.dataset_target + '_' + args.keypoints_target + '_test.npz'))
        skii_3d, skii_2d = keypoints_test['positions_3d'],keypoints_test['positions_2d']
        skii_2d-=skii_2d[:,:1]
        skii_2d/=np.linalg.norm(skii_2d,axis=(-1,-2),keepdims=True)
        skii_3d=skii_3d-skii_3d[:,:1]
        # joints=[0,4,5,6,1,2,3,7,8,9,10,11,12,13,14,15]
        # skii_3d, _3dpw_2d= _3dpw_3d[:,joints], _3dpw_2d[:,joints]
        print('test_shape',skii_2d.shape)
        test_loader = DataLoader(PoseBuffer(skii_3d, skii_2d,pad=args.pad),
                                batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # plot_16j(np.concatenate((skii_3d[200:201],poses_train[200:201]),axis=0))
    # for ii in range(4):
    #     plot_16j_2d(np.concatenate((skii_2d[ii*100:ii*100+1],poses_train_2d[ii*100:ii*100+1]),axis=0))

    return {
        'dataset': dataset,
        'train_loader': train_loader,
        'H36M_test': valid_loader,
        '3DHP_test': test_loader,
        'action_filter': action_filter,
        'subjects_test': subjects_test,
        'keypoints': keypoints
    }
