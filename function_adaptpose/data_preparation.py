from __future__ import print_function, absolute_import, division

import os.path as path
import copy
import numpy as np
from torch.utils.data import DataLoader
from common.viz import *
from common.data_loader import PoseDataSet, PoseBuffer, PoseTarget, PoseTarget_temp
from utils.data_utils import create_2d_data_3dhp_test, fetch, fetch_3dhp, fetch_3dhp_train, procrustes_torch,read_3d_data, create_2d_data, fetch_target, create_2d_data_target,align_poses



def data_preparation(args,remove_static_joints=True):

    dataset_path = path.join('data', 'data_3d_' + args.dataset + '.npz')
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset, TEST_SUBJECTS
        dataset = Human36mDataset(dataset_path,remove_static_joints=remove_static_joints)
        if args.s1only:
            subjects_train = ['S1']
        else:
            subjects_train = ['S1', 'S5', 'S6', 'S7', 'S8']
        subjects_test = TEST_SUBJECTS
    else:
        raise KeyError('Invalid dataset')



    print('==> Loading 3D data...')
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


    # prepare train loader for GT 2D - 3D, which will update by using projection.
    train_gt2d3d_loader = DataLoader(PoseDataSet(poses_train, poses_train_2d, actions_train, cams_train,args.pad),
                                     batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, pin_memory=True)

    valid_loader = DataLoader(PoseDataSet(poses_valid, poses_valid_2d, actions_valid, cams_valid,args.pad),
                              batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    ############################################
    # data loader for GAN training
    ############################################

    if args.dataset_target == '3dhp':
        keypoints_target = create_2d_data_target(path.join('data', 'data_'+args.dataset_target + '_' + args.keypoints_target + '_train.npz'), dataset)
        subjects_target=['S1','S2','S3','S4','S5','S6','S7']
        poses_target_2d = fetch_target(subjects_target,keypoints_target)
        poses_target_2d=np.concatenate(poses_target_2d)
        if args.keypoints_target=='pt':
            poses_target_2d = np.repeat(poses_target_2d,4,axis=0)
        poses_target_3d = fetch_3dhp_train(path.join('data', 'data_'+args.dataset_target + '_gt_train_3d.npz'))


    if args.dataset_target == 'h36m':
        poses_target_2d=poses_train_2d
    if args.dataset_target == '3dpw':
        keypoints_target = np.load(path.join('data', 'data_'+args.dataset_target + '_' + args.keypoints_target + '_train.npz'), allow_pickle=True)
        poses_target_2d = keypoints_target['positions_2d']
        poses_target_2d = np.repeat(poses_target_2d,50,axis=0)
        poses_target_3d = np.load(path.join('data', 'data_'+args.dataset_target + '_' + args.keypoints_target + '_train.npz'), allow_pickle=True)
        poses_target_3d = keypoints_target['positions_3d']   
        poses_target_3d = np.repeat(poses_target_3d,50,axis=0)
        print('3dpw_shape',poses_target_3d.shape)
    if args.dataset_target == 'skii':
        keypoints_target = np.load(path.join('data', 'data_'+args.dataset_target + '_' + args.keypoints_target + '_train.npz'), allow_pickle=True)
        poses_target_2d = keypoints_target['positions_2d']
        poses_target_2d-=poses_target_2d[:,:1]
        poses_target_2d/=np.linalg.norm(poses_target_2d,axis=(-1,-2),keepdims=True)
        # poses_target_2d = np.repeat(poses_target_2d,190,axis=0)
        # poses_target_2d=np.concatenate((poses_train_2d,poses_target_2d),axis=0)
        # np.random.shuffle(poses_target_2d)
        poses_target_2d=poses_train_2d
    print('target_shape',poses_target_2d.shape)

    target_2d_loader = DataLoader(PoseTarget(poses_target_2d),
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)

    target_3d_loader = DataLoader(PoseTarget(poses_train),
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    ## just for visualization of camera distribution 
    target_3d_loader2 = DataLoader(PoseTarget(poses_target_3d),
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)

    ############################################
    # prepare cross dataset validation
    ############################################

    if args.dataset_target == '3dhp' or args.dataset_target =='h36m':

        keypoints_test=create_2d_data_3dhp_test(path.join('data', 'data_3dhp_' + args.keypoints_target + '_test.npz'))
        mpi3dhp_3d,mpi3dhp_2d=fetch_3dhp(path.join('data', 'data_3dhp_' + args.keypoints_target + '_test.npz'),keypoints_test)
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

    
    # for ii in range(6):
    #     plot_16j(np.concatenate((mpi3dhp_3d[ii*500:ii*500+1],poses_train[ii*500:ii*500+1]),axis=0),frame_legend=['3dhp','h36m'])
    #     plot_16j_2d(np.concatenate((mpi3dhp_2d[ii*500:ii*500+1],poses_train_2d[ii*500:ii*500+1]),axis=0))

    return {
        'dataset': dataset,
        'train_gt2d3d_loader': train_gt2d3d_loader,
        'target_2d_loader': target_2d_loader,
        'target_3d_loader': target_3d_loader,
        'target_3d_loader2': target_3d_loader2,
        'H36M_test': valid_loader,
        'mpi3d_loader': test_loader,
        'action_filter': action_filter,
        'subjects_test': subjects_test,
        'keypoints': keypoints,
    }
