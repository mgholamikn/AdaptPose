from __future__ import print_function, absolute_import, division

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data_loader import PoseDataSet
from progress.bar import Bar
from utils.data_utils import fetch
from utils.loss import mpjpe, p_mpjpe, compute_PCK, compute_AUC
from utils.utils import AverageMeter
import matplotlib.pyplot as plt

#################################################################
def angle(v1, v2):
        return np.arccos((dotproduct(v1, v2) / (length(v1) * length(v2)+0.000001)))
def dotproduct(v1, v2):
        return sum((a*b) for a, b in zip(v1, v2))

def length(v):
        return np.sqrt(dotproduct(v, v))

def align(X,ref_vec):
        Z=np.zeros_like(X)
        v1=X[:,4]-X[:,1]
        v1[:,2]=0
        v2=[0,1,0]
        # v2=ref_vec
        def dotproduct(v1, v2):
                return sum((a*b) for a, b in zip(v1, v2))

        def length(v):
                return np.sqrt(dotproduct(v, v))

        def angle(v1, v2):
                return np.arccos((dotproduct(v1, v2) / (length(v1) * length(v2)+0.000001)))
        ii=0
        count=0
        while ii<X.shape[0]:
                theta=angle(v2,v1[ii])
                R=[[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]
                
                X[ii]=np.matmul(R,X[ii].T).T
                v1[ii]=X[ii,4]-X[ii,1]
                v1[:,2]=0
                res=angle(v2,v1[ii])

                if res<0.01:
                    ii+=1 
                    count=0
                else:
                    count+=1

                if count>500:
                    count=0
                    ii+=1  
        return X

####################################################################
# ### evaluate p1 p2 pck auc dataset with test-flip-augmentation
####################################################################
def evaluate(data_loader, model_pos_eval, device, summary=None, writer=None, key='', tag='', flipaug=''):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_p1 = AverageMeter()
    epoch_p2 = AverageMeter()
    epoch_auc = AverageMeter()
    epoch_pck = AverageMeter()

    # Switch to evaluate mode
    model_pos_eval.eval()
    end = time.time()

    bar = Bar('Eval posenet on {}'.format(key), max=len(data_loader))
    for i, temp in enumerate(data_loader):
        targets_3d, inputs_2d = temp[0], temp[1]
       

        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        inputs_2d = inputs_2d.to(device)

        with torch.no_grad():
            if flipaug:  # flip the 2D pose Left <-> Right
                joints_left = [4, 5, 6, 10, 11, 12]
                joints_right = [1, 2, 3, 13, 14, 15]
                out_left = [4, 5, 6, 10, 11, 12]
                out_right = [1, 2, 3, 13, 14, 15]
                inputs_2d_flip = inputs_2d.detach().clone()
                inputs_2d_flip[:, :, 0] *= -1
                inputs_2d_flip[:,  joints_left + joints_right, :] = inputs_2d_flip[:,  joints_right + joints_left, :]
                outputs_3d_flip = model_pos_eval(inputs_2d_flip.view(num_poses, -1)).view(num_poses, -1, 3).cpu()
                outputs_3d_flip = model_pos_eval(inputs_2d_flip).view(num_poses, -1, 3).cpu()
                outputs_3d_flip[:, :, 0] *= -1
                outputs_3d_flip[:, out_left + out_right, :] = outputs_3d_flip[:, out_right + out_left, :]

                outputs_3d = model_pos_eval(inputs_2d.view(num_poses, -1)).view(num_poses, -1, 3).cpu()
                outputs_3d = model_pos_eval(inputs_2d).view(num_poses, -1, 3).cpu()
                outputs_3d = (outputs_3d + outputs_3d_flip) / 2.0

            else:
                outputs_3d = model_pos_eval(inputs_2d.view(num_poses, -1)).view(num_poses, -1, 3).cpu()

        # caculate the relative position.

        targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint
        outputs_3d = outputs_3d[:, :, :] - outputs_3d[:, :1, :]

        p1score = mpjpe(outputs_3d, targets_3d).item() * 1000.0
        epoch_p1.update(p1score, num_poses)
        p2score = mpjpe(outputs_3d, targets_3d).item() * 1000.0 #p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0 #
        epoch_p2.update(p2score, num_poses)


        targets_3d=np.asarray(targets_3d.cpu())
        outputs_3d=np.asarray(outputs_3d.cpu())

        ref_vec=[0,1,0]
        A=align(outputs_3d[:],ref_vec)
        B=align(targets_3d[:],ref_vec)
        

        # plot17j(A)
        Knee_Angle_t=[]
        Hip_Angle_t=[]
        Knee_Angle=[]
        Hip_Angle=[]
        for jj in range(len(A)):
                v1=A[jj,2]-A[jj,1]
                v2=A[jj,3]-A[jj,2]
                v3=A[jj,0]-A[jj,7]

                v4=B[jj,2]-B[jj,1]
                v5=B[jj,3]-B[jj,2]
                v6=B[jj,0]-B[jj,7]
                
                v1[1]=0
                v2[1]=0
                v3[1]=0

                v4[1]=0
                v5[1]=0
                v6[1]=0

                v3=np.flip(v3)
                v3[2]*=-1

                v6=np.flip(v6)
                v6[2]*=-1

                if jj==0:
                        Knee0=angle(v1,v2)
                        Hip0=angle(v1,v3)
                        Knee0_t=angle(v4,v5)
                        Hip0_t=angle(v4,v6)
                Knee_Angle.append((angle(v1,v2)-Knee0)/np.pi*180)
                Hip_Angle.append(-(angle(v1,v3)-Hip0)/np.pi*180)
                Knee_Angle_t.append((angle(v4,v5)-Knee0_t)/np.pi*180)
                Hip_Angle_t.append(-(angle(v4,v6)-Hip0_t)/np.pi*180)        
        plt.plot(Knee_Angle,label='Knee Angle Pred')
        plt.plot(Knee_Angle_t,label='Knee Angle Target')
        plt.legend()
        plt.show()
        plt.plot(Hip_Angle,label='Hip Angle Pred')
        plt.plot(Hip_Angle_t,label='Hip Angle Target')
        plt.legend()
        plt.show()

            
       
       


        # compute AUC and PCK
        # pck = compute_PCK(targets_3d.numpy(), outputs_3d.numpy())
        # epoch_pck.update(pck, num_poses)
        # auc = compute_AUC(targets_3d.numpy(), outputs_3d.numpy())
        # epoch_auc.update(auc, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_p1.avg,
                    e2=epoch_p2.avg)
        bar.next()

    if writer:
        writer.add_scalar('posenet_{}'.format(key) + flipaug + '/p1score' + tag, epoch_p1.avg, summary.epoch)
        writer.add_scalar('posenet_{}'.format(key) + flipaug + '/p2score' + tag, epoch_p2.avg, summary.epoch)
        # writer.add_scalar('posenet_{}'.format(key) + flipaug + '/_pck' + tag, epoch_pck.avg, summary.epoch)
        # writer.add_scalar('posenet_{}'.format(key) + flipaug + '/_auc' + tag, epoch_auc.avg, summary.epoch)

    bar.finish()
    return epoch_p1.avg, epoch_p2.avg


#########################################
# overall evaluation function
#########################################
def evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device, summary, writer, tag):
    """
    evaluate H36M and 3DHP
    test-augment-flip only used for 3DHP as it does not help on H36M.
    """
    with torch.no_grad():
        model_pos_eval.load_state_dict(model_pos.state_dict())
        h36m_p1, h36m_p2 = evaluate(data_dict['H36M_test'], model_pos_eval, device, summary, writer,
                                             key='H36M_test', tag=tag, flipaug='')  # no flip aug for h36m
        dhp_p1, dhp_p2 = evaluate(data_dict['mpi3d_loader'], model_pos_eval, device, summary, writer,
                                           key='mpi3d_loader', tag=tag, flipaug='_flip')
    return h36m_p1, h36m_p2, dhp_p1, dhp_p2

