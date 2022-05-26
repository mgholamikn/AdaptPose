from __future__ import absolute_import, division

import numpy as np
import torch
import random
from common.camera import world_to_camera, normalize_screen_coordinates
from common.viz import plot_16j


def create_2d_data(data_path, dataset,tag):
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                if tag=='skii':
                    kps-=kps[:,:1]
                    kps/=np.linalg.norm(kps,axis=(-1,-2),keepdims=True)
                else:
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    return keypoints

def create_2d_data_target(data_path, dataset):
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()

        
    for subject in keypoints.keys():
        for seq in keypoints[subject].keys():
            for cam_idx in keypoints[subject][seq].keys():
                for ii in range(len(keypoints[subject][seq][cam_idx])): 
                    # Normalize camera frame
                    joints=[0,4,5,6,1,2,3,7,8,9,13,14,15,10,11,12]
                    keypoints[subject][seq][cam_idx][ii,:,:] = normalize_screen_coordinates(keypoints[subject][seq][cam_idx][ii,joints,:2], w=2048, h=2048)

    return keypoints

def create_2d_data_3dhp_test(data_path):
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()
    image_shape={'TS1':[2048,2048],'TS2':[2048,2048],'TS3':[2048,2048],'TS4':[2048,2048],'TS5':[1080,1920],'TS6':[1080,1920]}
     
    for subject in keypoints.keys():
        print(subject)
        for ii in range(len(keypoints[subject])): 
            # Normalize camera frame

            keypoints[subject][ii,:,:] = normalize_screen_coordinates(keypoints[subject][ii,:,:2], w=image_shape[subject][1], h=image_shape[subject][0])

    return keypoints

def fetch_3dhp(data_path,keypoints):
    data3d = np.load(data_path, allow_pickle=True)
    data3d = data3d['positions_3d'].item()
    data2d=keypoints

    data_3d=[]
    data_2d=[] 
    subjects=['TS1','TS2','TS3','TS4','TS5','TS6']  
    for subject in subjects:
        data3d[subject]-=data3d[subject][:,:1]
        # row=np.sum(np.abs(data2d[subject])>1,axis=(-1,-2))==0
        # data_3d.append(data3d[subject][row]/1000)
        # data_2d.append(data2d[subject][row])   
        if subject in ['TS3','TS4']:
            data_3d.append(data3d[subject][100:]/1000)
            data_2d.append(data2d[subject][100:])
        else:
            data_3d.append(data3d[subject]/1000)
            data_2d.append(data2d[subject])       
        #      
        
    return data_3d,data_2d

def fetch_3dhp_train(data_path):
    data3d = np.load(data_path, allow_pickle=True)
    data3d = data3d['positions_3d'].item()

    data_3d=[]

    for subject in data3d.keys():
        for seq in data3d[subject].keys():
            for cam in range(len(data3d[subject][seq])):
                data_3d.append(data3d[subject][seq][cam]/1000)
     
    data_3d=np.concatenate(data_3d)    
    return data_3d

def read_3d_data(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

    return dataset


def fetch(subjects, dataset, keypoints, action_filter=None, stride=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []
    out_cam = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.split(' ')[0] == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action.split(' ')[0]] * poses_2d[i].shape[0])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
                    cam = dataset[subject][action]['cameras'][i]['intrinsic']
                    out_cam.append([cam] * poses_3d[i].shape[0])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            out_cam[i]=out_cam[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    
    return out_poses_3d, out_poses_2d, out_actions, out_cam



def fetch_target(subjects, keypoints):
    out_poses_2d = []
    
    for subject in subjects:
        for seq in keypoints[subject].keys():
            for cam_idx in keypoints[subject][seq].keys():
                
                poses_2d=keypoints[subject][seq][cam_idx]

                out_poses_2d.append(poses_2d)

    return out_poses_2d


def procrustes_torch(X, Y, reflection=False):
    
    """
    Reimplementation of MATLAB's `procrustes` function to Numpy.
    """
    X1=X[:,[1,4,11,14,9,10]]
    Y1=Y[:,[1,4,11,14,9,10]]
    batch,n, m = X1.shape
    batch, ny, my = Y1.shape

    muX = torch.mean(X1,dim=1,keepdim=True)
    muY = torch.mean(Y1,dim=1,keepdim=True)

    X0 = X1 - muX
    Y0 = Y1 - muY

    # optimum rotation matrix of Y
    A = torch.matmul(torch.transpose(X0,-1,-2), Y0)
    U,s,V = torch.svd(A,some=False)
    T = torch.matmul(V, torch.transpose(U,-1,-2))

    # does the current solution use a reflection?
    have_reflection = np.linalg.det(T) < 0

    # if that's not what was specified, force another reflection
    # if reflection != have_reflection:
    V[have_reflection,:,-1] *= -1
    s[have_reflection,-1] *= -1
    T[have_reflection] = torch.matmul(V[have_reflection], torch.transpose(U[have_reflection],-1,-2))

    X1=X
    Y1=Y
    muX = torch.mean(X1,dim=1,keepdim=True)
    muY = torch.mean(Y1,dim=1,keepdim=True)

    X0 = X1 - muX
    Y0 = Y1 - muY

    Z = torch.matmul(Y0, T) + muY


    return np.array(Z)

def align_poses(dataset):
    subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
    for subject in subjects:
            for action in dataset[subject].keys():
                ref_pose=torch.from_numpy(dataset['S1']['Directions']['positions'][0:1].astype('float32'))
                data=dataset[subject][action]['positions']
                data=torch.from_numpy(data.astype('float32'))
                ref_pose.repeat(len(data),1,1)
                dataset[subject][action]['positions']=procrustes_torch(ref_pose,data)
    return dataset               

def random_loader(dataset,pad):
    pad=2*pad+1
    out_poses_3d = []
    subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
    rare_actions=['Sitti', 'Photo']
    
    for ii in range(1024):
        subject=subjects[torch.randint(0,5,(1,))]
        actions=list(dataset[subject].keys())
        
        act1=random.randint(0,len(actions)-1)
        act2=random.randint(0,len(actions)-1)
        act3=random.randint(0,len(actions)-1)
        act4=random.randint(0,len(actions)-1)
        act5=random.randint(0,len(actions)-1)


        if actions[act1][:5] in rare_actions:
                action=actions[act1]
                stride=np.random.randint(2,25)
                data=dataset[subject][action]['positions']
                data=data[::stride]
                rand1=random.randint(0,len(data)-pad-1) 
                rand2=random.randint(0,len(data)-pad-1)
                rand3=random.randint(0,len(data)-pad-1)
                rand4=random.randint(0,len(data)-pad-1)
                rand5=random.randint(0,len(data)-pad-1)
                P1=data[rand1:rand1+pad,:,:]
                P2=data[rand2:rand2+pad,:,:]
                P3=data[rand3:rand3+pad,:,:]
                P4=data[rand4:rand4+pad,:,:]
                P5=data[rand5:rand5+pad,:,:]
                new_data=np.array(list(P1))
        
                new_data[:,13:16,:]=P2[:,13:16,:]-P2[:,13:14,:]+P1[:,13:14,:]
                new_data[:,10:13,:]=P3[:,10:13,:]-P3[:,10:11,:]+P1[:,10:11,:]
                new_data[:,1:4,:]=P4[:,1:4,:]-P4[:,1:2,:]+P1[:,1:2,:]
                new_data[:,4:7,:]=P4[:,4:7,:]-P4[:,4:5,:]+P1[:,4:5,:]

        else:
                stride=np.random.randint(2,25)
                data1=dataset[subject][actions[act1]]['positions']
                data2=dataset[subject][actions[act2]]['positions']
                data3=dataset[subject][actions[act3]]['positions']
                data4=dataset[subject][actions[act4]]['positions']
                data5=dataset[subject][actions[act5]]['positions']
                data1=data1[::stride]
                data2=data2[::stride]
                data3=data3[::stride]
                data4=data4[::stride]
                data5=data5[::stride]

                rand1=random.randint(0,len(data1)-pad-1) #actions[act1]
                rand2=random.randint(0,len(data2)-pad-1)
                rand3=random.randint(0,len(data3)-pad-1)
                rand4=random.randint(0,len(data4)-pad-1)
                rand5=random.randint(0,len(data5)-pad-1)
                P1=data1[rand1:rand1+pad,:,:]
                P2=data2[rand2:rand2+pad,:,:]
                P3=data3[rand3:rand3+pad,:,:]
                P4=data4[rand4:rand4+pad,:,:]
                P5=data5[rand5:rand5+pad,:,:]
                new_data=np.array(list(P1))
        
                new_data[:,13:16,:]=P2[:,13:16,:]-P2[:,13:14,:]+P1[:,13:14,:]
                new_data[:,10:13,:]=P3[:,10:13,:]-P3[:,10:11,:]+P1[:,10:11,:]
                new_data[:,1:4,:]=P4[:,1:4,:]-P4[:,1:2,:]+P1[:,1:2,:]
                new_data[:,4:7,:]=P5[:,4:7,:]-P5[:,4:5,:]+P1[:,4:5,:]


        # jj=0
        # plot_16j(np.concatenate((P1[jj:jj+1],P2[jj:jj+1],P22[jj:jj+1],P3[jj:jj+1],P33[jj:jj+1],P4[jj:jj+1],P44[jj:jj+1],P5[jj:jj+1],P55[jj:jj+1],new_data[jj:jj+1]),axis=0))
        # plot_16j(np.concatenate((P1[jj:jj+1],P2[jj:jj+1],P3[jj:jj+1],P4[jj:jj+1],P5[jj:jj+1],new_data[jj:jj+1]),axis=0))

        out_poses_3d.append(new_data)

    out_poses_3d=np.array(out_poses_3d)
    
    return torch.from_numpy(out_poses_3d.astype('float32'))

