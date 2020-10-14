import torch
import torch.nn as nn
from torch.autograd import Variable


class ChamferLoss_Brute(nn.Module):
    def __init__(self, opt):
        super(ChamferLoss_Brute, self).__init__()
        self.opt = opt
        self.dimension = 3

    def forward(self, pc_src_input, pc_dst_input, sigma_src=None, sigma_dst=None):
        '''
        :param pc_src_input: Bx3xM Tensor in GPU
        :param pc_dst_input: Bx3xN Tensor in GPU
        :param sigma_src: BxM Tensor in GPU
        :param sigma_dst: BxN Tensor in GPU
        :return:
        '''

        B, M = pc_src_input.size()[0], pc_src_input.size()[2]
        N = pc_dst_input.size()[2]

        pc_src_input_expanded = pc_src_input.unsqueeze(3).expand(B, 3, M, N)
        pc_dst_input_expanded = pc_dst_input.unsqueeze(2).expand(B, 3, M, N)

        # the gradient of norm is set to 0 at zero-input. There is no need to use custom norm anymore.
        diff = torch.norm(pc_src_input_expanded - pc_dst_input_expanded, dim=1, keepdim=False)  # BxMxN

        # pc_src vs selected pc_dst, M
        src_dst_min_dist, _ = torch.min(diff, dim=2, keepdim=False)  # BxM
        forward_loss = src_dst_min_dist.mean()

        # pc_dst vs selected pc_src, N
        dst_src_min_dist, _ = torch.min(diff, dim=1, keepdim=False)  # BxN
        backward_loss = dst_src_min_dist.mean()

        chamfer_pure = forward_loss + backward_loss
        chamfer_weighted = chamfer_pure

        return forward_loss + backward_loss, chamfer_pure, chamfer_weighted


class CoverageLoss(nn.Module):
    def __init__(self, opt):
        super(CoverageLoss, self).__init__()
        self.opt = opt
        self.cov_criteria = nn.SmoothL1Loss() # reduction='none'

    def forward(self, kp, pc):
        # singular values - not efficient
        '''U, Spc, V = torch.svd(pc) 
        U, Skp, V = torch.svd(kp) 
        Spc = torch.div(Spc,torch.sum(Spc[:,:3], dim= 1).unsqueeze(1))
        Skp = torch.div(Skp,torch.sum(Skp[:,:3], dim= 1).unsqueeze(1))
        cov_loss = self.cov_criteria(Skp, Spc)''' 

        # volume
        val_max_pc, _ = torch.max(pc,2)
        val_min_pc, _ = torch.min(pc,2)
        dim_pc = val_max_pc - val_min_pc
        val_max_kp, _ = torch.max(kp,2)
        val_min_kp, _ = torch.min(kp,2)
        dim_kp = val_max_kp - val_min_kp
        cov_loss = self.cov_criteria(dim_kp, dim_pc)

        return cov_loss


class InclusivityLoss(nn.Module):
    def __init__(self, opt):
        super(InclusivityLoss, self).__init__()
        self.opt = opt

        self.single_side_chamfer = SingleSideChamferLoss_Brute(opt)

    def forward(self, keypoint, pc):
        loss = self.single_side_chamfer(keypoint, pc)
        
        return torch.mean(loss)


class SingleSideChamferLoss_Brute(nn.Module):
    def __init__(self, opt):
        super(SingleSideChamferLoss_Brute, self).__init__()
        self.opt = opt
        self.dimension = 3

    def forward(self, pc_src_input, pc_dst_input):
        '''
        :param pc_src_input: Bx3xM Variable in GPU
        :param pc_dst_input: Bx3xN Variable in GPU
        :return:
        '''

        B, M = pc_src_input.size()[0], pc_src_input.size()[2]
        N = pc_dst_input.size()[2]

        pc_src_input_expanded = pc_src_input.unsqueeze(3).expand(B, 3, M, N)
        pc_dst_input_expanded = pc_dst_input.unsqueeze(2).expand(B, 3, M, N)

        diff = torch.norm(pc_src_input_expanded - pc_dst_input_expanded, dim=1, keepdim=False)  # BxMxN

        # pc_src vs selected pc_dst, M
        src_dst_min_dist, _ = torch.min(diff, dim=2, keepdim=False)  # BxM

        return src_dst_min_dist


class SeparationLoss(nn.Module):
    def __init__(self, opt):
        super(SeparationLoss, self).__init__()
        self.opt = opt

    def forward(self, kp, delta):
        # We need input 3D keypoints and delta, a separation threshold. Incur 0 cost if the distance >= delta.
        num_kp = kp.shape[2] # b,3,M
        t1 = kp.repeat(1,num_kp,1) # b,3,M**2
        t2 = torch.reshape(kp.repeat(1,1,num_kp),t1.shape) # b,3*M,M -> b,3,M**2
        lensqr = torch.norm(t1 - t2, dim=2, keepdim=False)**2 # -> [b, M**2]
        sep_loss = torch.sum(torch.max(-lensqr + delta, torch.tensor([0.0]).cuda()))/(num_kp * self.opt.batch_size) 
        #sep_loss = torch.sum(lensqr/(num_kp*self.opt.batch_size))
        return sep_loss
