from __future__ import print_function, absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset
from functools import reduce

from common.viz import plot_16j


#####################################
# data loader with four output
#####################################
class PoseDataSet(Dataset):
    def __init__(self, poses_3d, poses_2d, actions, cams , pad=13):
        assert poses_3d is not None

        self._poses_3d = poses_3d
        self._poses_2d = poses_2d
        self._actions = reduce(lambda x, y: x + y, actions)
        self._cams = cams
        self._pad=pad

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0] and self._poses_3d.shape[0] == len(self._actions)
        assert self._poses_3d.shape[0] == self._cams.shape[0]
        print('Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, index):


        if self._pad>0:
            stride=np.random.randint(1,3)

            pad_idx=np.arange(2*self._pad+1)*stride-(self._pad*stride)+index

            if pad_idx[0]>=0 and pad_idx[-1]<len(self._poses_3d):
   
                out_pose_3d = np.reshape(self._poses_3d[pad_idx],(1,2*self._pad+1,self._poses_3d.shape[1],self._poses_3d.shape[2]))
                out_pose_2d = np.reshape(self._poses_2d[pad_idx],(1,2*self._pad+1,self._poses_2d.shape[1],self._poses_2d.shape[2]))
            else:
                out_pose_3d=np.zeros((1,2*self._pad+1,self._poses_3d.shape[1],self._poses_3d.shape[2]))
                out_pose_2d=np.zeros((1,2*self._pad+1,self._poses_2d.shape[1],self._poses_2d.shape[2]))
        else:
            out_pose_3d = np.reshape(self._poses_3d[index],(1,2*self._pad+1,self._poses_3d.shape[-2],self._poses_3d.shape[-1]))
            out_pose_2d = np.reshape(self._poses_2d[index],(1,2*self._pad+1,self._poses_2d.shape[-2],self._poses_2d.shape[-1]))

        out_action = self._actions[index]
        out_cam = self._cams[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d, out_action, out_cam

    def __len__(self):
        return len(self._actions)


#####################################
# data loader with four output
#####################################
class PoseDataSet2(Dataset):
    def __init__(self, poses_3d, poses_2d, actions, cams , pad=13):
        assert poses_3d is not None

        self._poses_3d = poses_3d
        self._poses_2d = poses_2d
        self._actions = reduce(lambda x, y: x + y, actions)
        self._cams = cams
        self._pad=pad

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0] and self._poses_3d.shape[0] == len(self._actions)
        assert self._poses_3d.shape[0] == self._cams.shape[0]
        print('Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, index):

        out_pose_3d = self._poses_3d[index:index+1]
        out_pose_2d = self._poses_2d[index:index+1]

        out_action = self._actions[index]
        out_cam = self._cams[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d, out_action, out_cam

    def __len__(self):
        return len(self._actions)

#####################################
# data loader with two output
#####################################
class PoseBuffer(Dataset):
    def __init__(self, poses_3d, poses_2d, pad=121, score=None ):
        assert poses_3d is not None

        self._poses_3d = poses_3d
        self._poses_2d = poses_2d
        self._pad = pad

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0]
        print('Generating {} poses...'.format(self._poses_3d.shape[0]))

    def __getitem__(self, index):
        if self._pad>0:

            if index-self._pad>=0 and index+self._pad<len(self._poses_3d):
                out_pose_3d = np.reshape(self._poses_3d[index-self._pad:index+self._pad+1],(1,2*self._pad+1,self._poses_3d.shape[1],self._poses_3d.shape[2]))
                out_pose_2d = np.reshape(self._poses_2d[index-self._pad:index+self._pad+1],(1,2*self._pad+1,self._poses_2d.shape[1],self._poses_2d.shape[2]))
            else:
                out_pose_3d=np.zeros((1,2*self._pad+1,self._poses_3d.shape[1],self._poses_3d.shape[2]))
                out_pose_2d=np.zeros((1,2*self._pad+1,self._poses_2d.shape[1],self._poses_2d.shape[2]))
        else:
            out_pose_3d = np.reshape(self._poses_3d[index],(1,2*self._pad+1,self._poses_3d.shape[-2],self._poses_3d.shape[-1]))
            out_pose_2d = np.reshape(self._poses_2d[index],(1,2*self._pad+1,self._poses_2d.shape[-2],self._poses_2d.shape[-1]))

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d

    def __len__(self):
        return len(self._poses_2d)


#############################################################
# data loader for GAN
#############################################################
class PoseTarget(Dataset):
    def __init__(self, poses):
        assert poses is not None
        # self._poses = np.concatenate(poses)
        self._poses = poses
        print('Generating {} poses...'.format(self._poses.shape[0]))

    def __getitem__(self, index):
        out_pose = self._poses[index]
        out_pose = torch.from_numpy(out_pose).float()
        return out_pose

    def __len__(self):
        return len(self._poses)

class PoseTarget_temp(Dataset):
    def __init__(self, poses, pad=121, score=None ):
        assert poses is not None
        self._poses = np.concatenate(poses)
        self._pad = pad
        print('Generating {} poses...'.format(self._poses.shape[0]))

    def __getitem__(self, index):
        if self._pad>0:
            if index-self._pad>0 and index+self._pad<len(self._poses):
                out_pose = np.reshape(self._poses[index-self._pad:index+self._pad+1],(1,2*self._pad+1,self._poses.shape[1],self._poses.shape[2]))
            else:
                out_pose=np.zeros((1,2*self._pad+1,self._poses.shape[1],self._poses.shape[2]))
        else:
            out_pose = self._poses[index]

        out_pose = torch.from_numpy(out_pose).float()
        return out_pose
        
    def __len__(self):
        return len(self._poses)


class PoseTarget3D(Dataset):
    def __init__(self, poses_3d):
        assert poses_3d is not None
        self._poses_3d = np.concatenate(poses_3d)
        print('Generating {} poses...'.format(self._poses_3d.shape[0]))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        return out_pose_3d

    def __len__(self):
        return len(self._poses_3d)


class PoseTarget2D(Dataset):
    def __init__(self, poses_2d):
        assert poses_2d is not None
        poses_2d = np.concatenate(poses_2d)
        tmp_mask = np.ones((poses_2d.shape[0], poses_2d.shape[1], 1), dtype='float32')
        self._poses_2d = np.concatenate((poses_2d, tmp_mask), axis=2)
        print('Generating {} poses...'.format(self._poses_2d.shape[0]))

    def __getitem__(self, index):
        out_pose_2d = self._poses_2d[index]
        out_pose_2d = torch.from_numpy(out_pose_2d).float()
        return out_pose_2d[:, :-1], out_pose_2d[:, -1:]

    def __len__(self):
        return len(self._poses_2d)

