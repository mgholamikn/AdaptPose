import torch
import torch.nn as nn
import pytorch3d.transforms as torch3d
from common.viz import plot_16j
import numpy as np
class KCSpath(nn.Module):
    def __init__(self, num_joints=16, channel=1000, channel_mid=100):
        super(KCSpath, self).__init__()
        # KCS path
        self.kcs_layer_1 = nn.Linear(225, channel)
        self.kcs_layer_2 = nn.Linear(channel, channel)
        self.kcs_layer_3 = nn.Linear(channel, channel)

        self.layer_last = nn.Linear(channel, channel_mid)
        self.layer_pred = nn.Linear(channel_mid, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # KCS path
        psi_vec = self.relu(self.kcs_layer_1(x))
        d1_psi = self.relu(self.kcs_layer_2(psi_vec))
        d2_psi = self.kcs_layer_3(d1_psi) + psi_vec
        y = self.relu(self.layer_last(d2_psi))
        y = self.layer_pred(y)
        return y

class KCSpath2d(nn.Module):
    def __init__(self, num_joints=16, channel=1000, channel_mid=100):
        super(KCSpath2d, self).__init__()
        # KCS path
        self.kcs_layer_1 = nn.Linear(225, channel)
        self.kcs_layer_2 = nn.Linear(channel, channel)
        self.kcs_layer_3 = nn.Linear(channel, channel)

        self.layer_last = nn.Linear(channel, channel_mid)
        self.layer_pred = nn.Linear(channel_mid, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # KCS path
        psi_vec = self.relu(self.kcs_layer_1(x))
        d1_psi = self.relu(self.kcs_layer_2(psi_vec))
        d2_psi = self.kcs_layer_3(d1_psi) + psi_vec
        y = self.relu(self.layer_last(d2_psi))

        return y

from utils.gan_utils import get_bone_unit_vecbypose3d, get_pose3dbyBoneVec, get_BoneVecbypose3d


class Pos3dDiscriminator(nn.Module):
    def __init__(self, num_joints=16, kcs_channel=256, channel_mid=100):
        super(Pos3dDiscriminator, self).__init__()
        # only check on bone angle, not bone vector.

        # KCS path
        self.kcs_path_1 = KCSpath(channel=kcs_channel, channel_mid=channel_mid)
        self.kcs_path_2 = KCSpath(channel=kcs_channel, channel_mid=channel_mid)
        self.kcs_path_3 = KCSpath(channel=kcs_channel, channel_mid=channel_mid)
        self.kcs_path_4 = KCSpath(channel=kcs_channel, channel_mid=channel_mid)
        self.kcs_path_5 = KCSpath(channel=kcs_channel, channel_mid=channel_mid)

        self.relu = nn.LeakyReLU()

    def forward(self, inputs_3d):
        # convert 3d pose to root relative
        inputs_3d=inputs_3d.squeeze()
        x = inputs_3d - inputs_3d[:, :1, :]  # x: root relative
        bv_unit = get_bone_unit_vecbypose3d(x)
        x = get_pose3dbyBoneVec(bv_unit)

        ## Perturbation on:
        # x_axis=torch.rand(x.shape).cuda()
        # x_axis = x_axis/torch.linalg.norm(x_axis,dim=-1,keepdim=True)
        # x_axis = x_axis*10/180*3.14
        # x_rM = torch3d.axis_angle_to_matrix(x_axis.view(-1,3))[..., :3, :3]  # Nx4x4->Nx3x3 rotation matrix
        # x_rM=x_rM.view(x.shape[0],x.shape[1],3,3)

        # modifyed_x=torch.matmul(x_rM,x.unsqueeze(-1)).squeeze()
        
        ## Perturbation off:
        modifyed_x=x
        # plot_16j(np.concatenate((modifyed_x[:1].cpu().detach().numpy(),x[:1].cpu().detach().numpy()),axis=0),frame_legend=['mod','gt'])
        # KCS path
        
        psi_vec_lh = kcs_layer_lh(modifyed_x).view((x.size(0), -1))
        k_lh = self.kcs_path_1(psi_vec_lh)

        psi_vec_rh = kcs_layer_rh(modifyed_x).view((x.size(0), -1))
        k_rh = self.kcs_path_2(psi_vec_rh)

        psi_vec_ll = kcs_layer_ll(modifyed_x).view((x.size(0), -1))
        k_ll = self.kcs_path_3(psi_vec_ll)

        psi_vec_rl = kcs_layer_rl(modifyed_x).view((x.size(0), -1))
        k_rl = self.kcs_path_4(psi_vec_rl)

        psi_vec_hb = kcs_layer_hb(x).view((x.size(0), -1))
        k_hb = self.kcs_path_5(psi_vec_hb)

        out = torch.cat([k_lh, k_rh, k_ll, k_rl, k_hb], dim=1)
        return out


def kcs_layer_hb(x, num_joints=16):
    """
    torso part
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [0, 3, 6, 7, 8, 9, 12]
    mask[:, hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0, 2, 1).contiguous())
    return Psi


def kcs_layer_rl(x, num_joints=16):
    """
    right leg
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [0, 3, 4, 5, 6]
    mask[:, hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0, 2, 1).contiguous())
    return Psi


def kcs_layer_ll(x, num_joints=16):
    """
    left leg
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [0, 1, 2, 3, 6]
    mask[:, hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0, 2, 1).contiguous())
    return Psi


def kcs_layer_lh(x, num_joints=16):
    """
    left hand
    """
    
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [7, 9, 10, 11]
    mask[:, hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0, 2, 1).contiguous())
    return Psi


def kcs_layer_rh(x, num_joints=16):
    """
    right hand
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [7, 12, 13, 14]
    mask[:, hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0, 2, 1).contiguous())
    return Psi




class Pos3dDiscriminator_temp(nn.Module):
    def __init__(self, num_joints=16, kcs_channel=256, channel_mid=100):
        super(Pos3dDiscriminator_temp, self).__init__()
        # only check on bone angle, not bone vector.

        # KCS path
        self.kcs_path_1 = KCSpath(channel=kcs_channel, channel_mid=channel_mid)
        self.kcs_path_2 = KCSpath(channel=kcs_channel, channel_mid=channel_mid)
        self.kcs_path_3 = KCSpath(channel=kcs_channel, channel_mid=channel_mid)
        self.kcs_path_4 = KCSpath(channel=kcs_channel, channel_mid=channel_mid)
        self.kcs_path_5 = KCSpath(channel=kcs_channel, channel_mid=channel_mid)

        self.relu = nn.LeakyReLU()

    def forward(self, inputs_3d):
        # convert 3d pose to root relative
        inputs_3d=inputs_3d.squeeze()
        x = inputs_3d - inputs_3d[:, :, :1, :]  # x: root relative
        bv_unit = get_bone_unit_vecbypose3d(x)
        x = get_pose3dbyBoneVec(bv_unit)
        x = x[:,1:]-x[:,:-1]
        # KCS path
        psi_vec_lh = kcs_layer_lh_temp(x).view((x.size(0),-1))
        k_lh = self.kcs_path_1(psi_vec_lh)

        psi_vec_rh = kcs_layer_rh_temp(x).view((x.size(0), -1))
        k_rh = self.kcs_path_2(psi_vec_rh)

        psi_vec_ll = kcs_layer_ll_temp(x).view((x.size(0), -1))
        k_ll = self.kcs_path_3(psi_vec_ll)

        psi_vec_rl = kcs_layer_rl_temp(x).view((x.size(0), -1))
        k_rl = self.kcs_path_4(psi_vec_rl)

        psi_vec_hb = kcs_layer_hb_temp(x).view((x.size(0), -1))
        k_hb = self.kcs_path_5(psi_vec_hb)

        out = torch.cat([k_lh, k_rh, k_ll, k_rl, k_hb], dim=1)
        return out


def kcs_layer_hb_temp(x, num_joints=16):
    """
    torso part
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [0, 3, 6, 7, 8, 9, 12]
    mask[:,:, hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0,1, 3, 2).contiguous())
    return Psi


def kcs_layer_rl_temp(x, num_joints=16):
    """
    right leg
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [0, 3, 4, 5, 6]
    mask[:, :,hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0,1, 3, 2).contiguous())
    return Psi


def kcs_layer_ll_temp(x, num_joints=16):
    """
    left leg
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [0, 1, 2, 3, 6]
    mask[:, :,hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0, 1,3, 2).contiguous())
    return Psi


def kcs_layer_lh_temp(x, num_joints=16):
    """
    left hand
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [7, 9, 10, 11]
    mask[:, :, hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0, 1, 3, 2).contiguous())
    return Psi


def kcs_layer_rh_temp(x, num_joints=16):
    """
    right hand
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [7, 12, 13, 14]
    mask[:, :,hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0, 1,3, 2).contiguous())
    return Psi


class Pos2dDiscriminator(nn.Module):
    def __init__(self, num_joints=16):
        super(Pos2dDiscriminator, self).__init__()

        # Pose path
        self.pose_layer_1 = nn.Linear(num_joints * 2, 100)
        self.pose_layer_2 = nn.Linear(100, 100)
        self.pose_layer_3 = nn.Linear(100, 100)
        self.pose_layer_4 = nn.Linear(100, 100)

        self.layer_last = nn.Linear(100, 100)
        self.layer_pred = nn.Linear(100, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # Pose path
        x = x.contiguous().view(x.size(0), -1)
        d1 = self.relu(self.pose_layer_1(x))
        d2 = self.relu(self.pose_layer_2(d1))
        d3 = self.relu(self.pose_layer_3(d2) + d1)
        d4 = self.pose_layer_4(d3)

        d_last = self.relu(self.layer_last(d4))
        d_out = self.layer_pred(d_last)

        return d_out


class Pos2dDiscriminator_temp(nn.Module):
    def __init__(self, num_joints=16,win_length=243):
        super(Pos2dDiscriminator_temp, self).__init__()
        
        # Pose path
        self.pose_layer_1 = nn.Linear((win_length-1)*num_joints * 2, 100)
        self.pose_layer_2 = nn.Linear(100, 100)
        self.pose_layer_3 = nn.Linear(100, 100)
        self.pose_layer_4 = nn.Linear(100, 100)

        self.layer_last = nn.Linear(100, 100)
        self.layer_pred = nn.Linear(100, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # Pose path
        x=x.squeeze()
        x=x[:,1:]-x[:,:-1]
        x = x.contiguous().view(x.size(0), -1)
        d1 = self.relu(self.pose_layer_1(x))
        d2 = self.relu(self.pose_layer_2(d1))
        d3 = self.relu(self.pose_layer_3(d2) + d1)
        d4 = self.pose_layer_4(d3)

        d_last = self.relu(self.layer_last(d4))
        d_out = self.layer_pred(d_last)

        return d_out

########################################
########################################
#########  2D KCS ######################


class Pos2dDiscriminator_kcs(nn.Module):
    def __init__(self, num_joints=16, kcs_channel=256, channel_mid=100):
        super(Pos2dDiscriminator_kcs, self).__init__()
        # only check on bone angle, not bone vector.

        # KCS path
        # self.kcs_path_1 = KCSpath(channel=kcs_channel, channel_mid=channel_mid)
        # self.kcs_path_2 = KCSpath(channel=kcs_channel, channel_mid=channel_mid)
        # self.kcs_path_3 = KCSpath(channel=kcs_channel, channel_mid=channel_mid)
        # self.kcs_path_4 = KCSpath(channel=kcs_channel, channel_mid=channel_mid)
        self.kcs_path_5 = KCSpath2d(channel=kcs_channel, channel_mid=channel_mid)
        # Pose path
        self.pose_layer_1 = nn.Linear(num_joints * 2, 100)
        self.pose_layer_2 = nn.Linear(100, 100)
        self.pose_layer_3 = nn.Linear(100, 100)
        self.pose_layer_4 = nn.Linear(100, 100)

        self.layer_last = nn.Linear(100, 100)
        self.layer_pred1 = nn.Linear(200, 100)
        self.layer_pred2 = nn.Linear(100, 1)

        self.relu = nn.LeakyReLU()
        self.relu = nn.LeakyReLU()

    def forward(self, inputs_2d):
        # convert 3d pose to root relative
        inputs_2d=inputs_2d.squeeze()
        x = inputs_2d

        psi_vec_hb = kcs_layer_hb_2d(x).view((x.size(0), -1))
        k_hb = self.kcs_path_5(psi_vec_hb)

        x = inputs_2d.contiguous().view(inputs_2d.size(0), -1)
        d1 = self.relu(self.pose_layer_1(x))
        d2 = self.relu(self.pose_layer_2(d1))
        d3 = self.relu(self.pose_layer_3(d2) + d1)
        d4 = self.pose_layer_4(d3)

        d_last = self.relu(self.layer_last(d4))
        d_last=torch.cat((k_hb,d_last),dim=-1)
        d_out = self.relu(self.layer_pred1(d_last))
        d_out = self.layer_pred2(d_out)
        return d_out


def kcs_layer_hb_2d(x, num_joints=16):
    """
    torso part
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [0, 1,2,3,4,5, 6, 7, 8, 9,10,11, 12,13,14]
    mask[:, hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0, 2, 1).contiguous())
    return Psi


def kcs_layer_rl_2d(x, num_joints=16):
    """
    right leg
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [0, 3, 4, 5, 6]
    mask[:, hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0, 2, 1).contiguous())
    return Psi


def kcs_layer_ll_2d(x, num_joints=16):
    """
    left leg
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [0, 1, 2, 3, 6]
    mask[:, hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0, 2, 1).contiguous())
    return Psi


def kcs_layer_lh_2d(x, num_joints=16):
    """
    left hand
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [7, 9, 10, 11]
    mask[:, hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0, 2, 1).contiguous())
    return Psi


def kcs_layer_rh_2d(x, num_joints=16):
    """
    right hand
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [7, 12, 13, 14]
    mask[:, hb_idx, :] = 1
    bv = bv * mask
    Psi = torch.matmul(bv, bv.permute(0, 2, 1).contiguous())
    return Psi
