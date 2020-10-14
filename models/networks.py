import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,'../util'))
import som
from models.layers import PointNet, GeneralKNNFusionModule, EquivariantLayer, InstanceBranch
import index_max


class KP_Detector(nn.Module): 
    def __init__(self, opt):
        super(KP_Detector, self).__init__()
        self.opt = opt

        # ---- Nodes branch definition ----

        self.C1 = 128
        self.C2 = 512
        input_channels = self.C1 + self.C2
        output_channels = 4 # 3 coordinates + sigma
        assert self.opt.node_knn_k_1 >= 2

        self.first_pointnet = PointNet(3+self.opt.surface_normal_len,
                                       [int(self.C1/2), int(self.C1/2), int(self.C1/2)],
                                       activation=self.opt.activation,
                                       normalization=self.opt.normalization,
                                       momentum=opt.bn_momentum,
                                       bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                       bn_momentum_decay=opt.bn_momentum_decay)

        self.second_pointnet = PointNet(self.C1, [self.C1, self.C1], activation=self.opt.activation,
                                        normalization=self.opt.normalization,
                                        momentum=opt.bn_momentum,
                                        bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                        bn_momentum_decay=opt.bn_momentum_decay)

        self.knnlayer_1 = GeneralKNNFusionModule(3 + self.C1, (int(self.C2/2), int(self.C2/2), int(self.C2/2)), (self.C2, self.C2),
                                                 activation=self.opt.activation,
                                                 normalization=self.opt.normalization,
                                                 momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                                 bn_momentum_decay=opt.bn_momentum_decay)

        self.mlp1 = EquivariantLayer(input_channels, 512,
                                     activation=self.opt.activation, normalization=self.opt.normalization,
                                     momentum=opt.bn_momentum,
                                     bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                     bn_momentum_decay=opt.bn_momentum_decay)
       
        self.mlp2 = EquivariantLayer(512, 256,
                                     activation=self.opt.activation, normalization=self.opt.normalization,
                                     momentum=opt.bn_momentum,
                                     bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                     bn_momentum_decay=opt.bn_momentum_decay)
        
        self.mlp3 = EquivariantLayer(256, output_channels, activation=None, normalization=None)

        self.mlp3.conv.weight.data.normal_(0, 1e-4)
        self.mlp3.conv.bias.data.zero_()
        self.softplus = torch.nn.Softplus()

        # ---- Pose and coefficients branch definition ----

        self.third_pointnet2 = InstanceBranch(3+self.opt.surface_normal_len, 
                                        [int(self.C1/2), self.C1, self.opt.basis_num+1], 
                                        self.opt.basis_num,
                                        activation=self.opt.activation,
                                        normalization=self.opt.normalization,
                                        momentum=opt.bn_momentum,
                                        bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                        bn_momentum_decay=opt.bn_momentum_decay)

        # ---- Additional learnable parameters ----

        #self.basis = torch.nn.Parameter((torch.rand(1, 3, self.opt.node_num, self.opt.basis_num) - 0.5).cuda()) # 1x3xMxK
        self.basis = torch.nn.Parameter((torch.rand(1, 3, self.opt.node_num//2, self.opt.basis_num)-0.5).cuda()) # 1x3xM/2xK
        self.n_pl = torch.nn.Parameter(torch.rand(1,2).cuda())
        # self.R_n_pl = torch.nn.Parameter((torch.rand(1)-0.5).cuda())


    def forward(self, x, sn, node, is_train=False, epoch=None):
        '''
        :param x: Bx3xN Tensor
        :param sn: Bx3xN Tensor
        :param node: Bx3xM FloatTensor
        :param is_train: determine whether to add noise in KNNModule
        :return:
        '''
        # modify the x according to the nodes, minus the center
        mask, mask_row_max, min_idx = som.query_topk(node, x, node.size()[2], k=self.opt.k)  # BxkNxnode_num, Bxnode_num, BxkN
        mask_row_sum = torch.sum(mask, dim=1)  # Bxnode_num
        mask = mask.unsqueeze(1)  # Bx1xkNxnode_num

        # if necessary, stack the x
        x_stack = x.repeat(1, 1, self.opt.k)
        sn_stack = sn.repeat(1, 1, self.opt.k)

        x_stack_data_unsqueeze = x_stack.unsqueeze(3)  # BxCxkNx1
        x_stack_data_masked = x_stack_data_unsqueeze * mask.float()  # BxCxkNxnode_num
        cluster_mean = torch.sum(x_stack_data_masked, dim=2) / (mask_row_sum.unsqueeze(1).float() + 1e-5).detach()  # BxCxnode_num
        som_node_cluster_mean = cluster_mean

        B, N, kN, M = x.size()[0], x.size()[2], x_stack.size()[2], som_node_cluster_mean.size()[2]

        # assign each point with a center
        node_expanded = som_node_cluster_mean.unsqueeze(2)  # BxCx1xnode_num, som.node is BxCxnode_num
        centers = torch.sum(mask.float() * node_expanded, dim=3).detach()  # BxCxkN

        x_decentered = (x_stack - centers).detach()  # Bx3xkN
        x_augmented = torch.cat((x_decentered, sn_stack), dim=1)  # Bx6xkN

        # ---- Nodes branch ----

        # First PointNet
        if self.opt.surface_normal_len >= 1:
            first_pn_out = self.first_pointnet(x_augmented, epoch)
        else:
            first_pn_out = self.first_pointnet(x_decentered, epoch)

        with torch.cuda.device(first_pn_out.get_device()):
            first_gather_index = index_max.forward_cuda_shared_mem(first_pn_out.detach(), min_idx.int(), M).detach().long()
        first_pn_out_masked_max = first_pn_out.gather(dim=2,index=first_gather_index) * mask_row_max.unsqueeze(1).float()  # BxCxM

        # scatter the masked_max back to the kN points
        scattered_first_masked_max = torch.gather(first_pn_out_masked_max,
                                                  dim=2,
                                                  index=min_idx.unsqueeze(1).expand(B, first_pn_out.size()[1], kN))  # BxCxkN
        first_pn_out_fusion = torch.cat((first_pn_out, scattered_first_masked_max), dim=1)  # Bx2CxkN

        # Second PointNet
        second_pn_out = self.second_pointnet(first_pn_out_fusion, epoch)

        with torch.cuda.device(second_pn_out.get_device()):
            second_gather_index = index_max.forward_cuda_shared_mem(second_pn_out, min_idx.int(), M).detach().long()
        second_pn_out_masked_max = second_pn_out.gather(dim=2,index=second_gather_index) * mask_row_max.unsqueeze(1).float()  # BxCxM
     
        # knn search on nodes
        knn_feature_1 = self.knnlayer_1(query=som_node_cluster_mean,
                                        database=som_node_cluster_mean,
                                        x=second_pn_out_masked_max,
                                        K=self.opt.node_knn_k_1,
                                        epoch=epoch)
        node_feature_aggregated = torch.cat((second_pn_out_masked_max, knn_feature_1), dim=1)  # Bx(C1+C2)xM
        
        # mlp to calculate the per-node keypoint
        y = self.mlp1(node_feature_aggregated)
        point_descriptor = self.mlp2(y)
        keypoint_sigma = self.mlp3(point_descriptor)  # Bx(3+1)xkN
        nodes = keypoint_sigma[:, 0:3, :] + som_node_cluster_mean  # Bx3xM
        

        # -- Pose and coefficients branch --
        x_init_augmented = torch.cat((x_stack, sn_stack), dim=1)
        coefsR = self.third_pointnet2(x_init_augmented, epoch)
        coefs = coefsR[:,:-1]
        rot = coefsR[:,-1:]
        
        return nodes, coefs, rot 

    
class KP_Detector_reduced(nn.Module):
    def __init__(self, opt):
        super(KP_Detector_reduced, self).__init__()
        self.opt = opt

        # -- Pose and coefficients branch definition --
        self.C1 = 128
        self.instance_branch = InstanceBranch(
            3 + self.opt.surface_normal_len,
            [int(self.C1 / 4), int(self.C1 / 2), int(self.C1 / 2), self.C1],
            self.opt.basis_num * 2,
            1,
            self.opt.node_num,
            activation=self.opt.activation,
            normalization=self.opt.normalization,
            momentum=opt.bn_momentum,
            bn_momentum_decay_step=opt.bn_momentum_decay_step,
            bn_momentum_decay=opt.bn_momentum_decay,
        )

        # -- Additional learnable parameters --
        # self.basis = torch.nn.Parameter((torch.rand(1, 3, self.opt.node_num, self.opt.basis_num) - 0.5).cuda()) # 1x3xNxK - no symmetry
        self.basis = torch.nn.Parameter(
            (torch.rand(1, 3, self.opt.node_num // 2, self.opt.basis_num) - 0.5).cuda()
        )  # 1x3xN/2xK - symmetry
        self.n_pl = torch.nn.Parameter(torch.rand(1, 2).cuda())
        self.R_n_pl = torch.nn.Parameter((torch.rand(1) - 0.5).cuda())


    def forward(self, x, sn, node, is_train=False, epoch=None):

        # -- Prepare input --
        # if necessary, stack the x
        x_stack = x.repeat(1, 1, self.opt.k)
        sn_stack = sn.repeat(1, 1, self.opt.k)

        # -- Pose and coefficients branch --
        x_init_augmented = torch.cat((x_stack, sn_stack), dim=1)
        coeffs, rot = self.instance_branch(x_init_augmented, epoch) 

        return None, coeffs, rot 


class KP_Detector_(nn.Module):
    def __init__(self, opt):
        super(KP_Detector, self).__init__()
        self.opt = opt

        # -- Nodes branch definition --
        self.C1 = 128
        self.first_pointnet = PointNet(
            3 + self.opt.surface_normal_len,
            [int(self.C1 / 2), int(self.C1 / 2), int(self.C1 / 2)],
            activation=self.opt.activation,
            normalization=self.opt.normalization,
            momentum=opt.bn_momentum,
            bn_momentum_decay_step=opt.bn_momentum_decay_step,
            bn_momentum_decay=opt.bn_momentum_decay,
        )
        self.second_pointnet = PointNet(
            self.C1,
            [self.C1, self.C1],
            activation=self.opt.activation,
            normalization=self.opt.normalization,
            momentum=opt.bn_momentum,
            bn_momentum_decay_step=opt.bn_momentum_decay_step,
            bn_momentum_decay=opt.bn_momentum_decay,
        )
        assert self.opt.node_knn_k_1 >= 2
        self.C2 = 512
        self.knnlayer_1 = GeneralKNNFusionModule(
            3 + self.C1,
            (int(self.C2 / 2), int(self.C2 / 2), int(self.C2 / 2)),
            (self.C2, self.C2),
            activation=self.opt.activation,
            normalization=self.opt.normalization,
            momentum=opt.bn_momentum,
            bn_momentum_decay_step=opt.bn_momentum_decay_step,
            bn_momentum_decay=opt.bn_momentum_decay,
        )
        input_channels = self.C1 + self.C2
        output_channels = 4  # 3 coordinates + sigma
        self.mlp1 = EquivariantLayer(
            input_channels,
            512,
            activation=self.opt.activation,
            normalization=self.opt.normalization,
            momentum=opt.bn_momentum,
            bn_momentum_decay_step=opt.bn_momentum_decay_step,
            bn_momentum_decay=opt.bn_momentum_decay,
        )
        self.mlp2 = EquivariantLayer(
            512,
            256,
            activation=self.opt.activation,
            normalization=self.opt.normalization,
            momentum=opt.bn_momentum,
            bn_momentum_decay_step=opt.bn_momentum_decay_step,
            bn_momentum_decay=opt.bn_momentum_decay,
        )
        self.mlp3 = EquivariantLayer(
            256, output_channels, activation=None, normalization=None
        )
        self.mlp3.conv.weight.data.normal_(0, 1e-4)
        self.mlp3.conv.bias.data.zero_()
        self.softplus = torch.nn.Softplus()

        # -- Pose and coefficients branch definition --
        self.instance_branch = InstanceBranch2(
            3 + self.opt.surface_normal_len,
            [int(self.C1 / 2), self.C1],
            self.opt.basis_num,
            1,
            self.opt.node_num,
            activation=self.opt.activation,
            normalization=self.opt.normalization,
            momentum=opt.bn_momentum,
            bn_momentum_decay_step=opt.bn_momentum_decay_step,
            bn_momentum_decay=opt.bn_momentum_decay,
        )

        # -- Additional learnable parameters --
        # self.basis = torch.nn.Parameter((torch.rand(1, 3, self.opt.node_num, self.opt.basis_num) - 0.5).cuda()) # 1x3xMxK - no symmetry
        self.basis = torch.nn.Parameter(
            (torch.rand(1, 3, self.opt.node_num // 2, self.opt.basis_num) - 0.5).cuda()
        )  # 1x3xM/2xK - symmetry
        self.n_pl = torch.nn.Parameter(torch.rand(1, 2).cuda())
        self.R_n_pl = torch.nn.Parameter((torch.rand(1) - 0.5).cuda())


    def forward(self, x, sn, node, is_train=False, epoch=None):
        """
        :param x: Bx3xN Tensor
        :param sn: Bx3xN Tensor
        :param node: Bx3xM FloatTensor
        :param is_train: determine whether to add noise in KNNModule
        :return:
        """
        # -- Prepare input --
        # modify the x according to the nodes, minus the center
        mask, mask_row_max, min_idx = som.query_topk(
            node, x, node.size()[2], k=self.opt.k
        )  # BxkNxnode_num, Bxnode_num, BxkN
        mask_row_sum = torch.sum(mask, dim=1)  # Bxnode_num
        mask = mask.unsqueeze(1)  # Bx1xkNxnode_num

        # if necessary, stack the x
        x_stack = x.repeat(1, 1, self.opt.k)
        sn_stack = sn.repeat(1, 1, self.opt.k)

        x_stack_data_unsqueeze = x_stack.unsqueeze(3)  # BxCxkNx1
        x_stack_data_masked = x_stack_data_unsqueeze * mask.float()  # BxCxkNxnode_num
        cluster_mean = (
            torch.sum(x_stack_data_masked, dim=2)
            / (mask_row_sum.unsqueeze(1).float() + 1e-5).detach()
        )  # BxCxnode_num
        # cluster_mean = node
        som_node_cluster_mean = cluster_mean

        B, N, kN, M = (
            x.size()[0],
            x.size()[2],
            x_stack.size()[2],
            som_node_cluster_mean.size()[2],
        )

        # assign each point with a center
        node_expanded = som_node_cluster_mean.unsqueeze(
            2
        )  # BxCx1xnode_num, som.node is BxCxnode_num
        centers = torch.sum(mask.float() * node_expanded, dim=3).detach()  # BxCxkN

        x_decentered = (x_stack - centers).detach()  # Bx3xkN
        x_augmented = torch.cat((x_decentered, sn_stack), dim=1)  # Bx6xkN

        # -- Nodes branch --
        # First PointNet
        if self.opt.surface_normal_len >= 1:
            first_pn_out = self.first_pointnet(x_augmented, epoch)
        else:
            first_pn_out = self.first_pointnet(x_decentered, epoch)

        with torch.cuda.device(first_pn_out.get_device()):
            first_gather_index = (
                index_max.forward_cuda_shared_mem(
                    first_pn_out.detach(), min_idx.int(), M
                )
                .detach()
                .long()
            )
        first_pn_out_masked_max = (
            first_pn_out.gather(dim=2, index=first_gather_index)
            * mask_row_max.unsqueeze(1).float()
        )  # BxCxM
        # scatter the masked_max back to the kN points
        scattered_first_masked_max = torch.gather(
            first_pn_out_masked_max,
            dim=2,
            index=min_idx.unsqueeze(1).expand(B, first_pn_out.size()[1], kN),
        )  # BxCxkN
        first_pn_out_fusion = torch.cat(
            (first_pn_out, scattered_first_masked_max), dim=1
        )  # Bx2CxkN
        # Second PointNet
        second_pn_out = self.second_pointnet(first_pn_out_fusion, epoch)
        with torch.cuda.device(second_pn_out.get_device()):
            second_gather_index = (
                index_max.forward_cuda_shared_mem(second_pn_out, min_idx.int(), M)
                .detach()
                .long()
            )
        second_pn_out_masked_max = (
            second_pn_out.gather(dim=2, index=second_gather_index)
            * mask_row_max.unsqueeze(1).float()
        )  # BxCxM
        # knn search on nodes
        knn_feature_1 = self.knnlayer_1(
            query=som_node_cluster_mean,
            database=som_node_cluster_mean,
            x=second_pn_out_masked_max,
            K=self.opt.node_knn_k_1,
            epoch=epoch,
        )
        node_feature_aggregated = torch.cat(
            (second_pn_out_masked_max, knn_feature_1), dim=1
        )  # Bx(C1+C2)xM

        # mlp to calculate the per-node keypoint
        y = self.mlp1(node_feature_aggregated)
        point_descriptor = self.mlp2(y)
        keypoint_sigma = self.mlp3(point_descriptor)  # Bx(3+1)xkN
        nodes = keypoint_sigma[:, 0:3, :] + som_node_cluster_mean  # Bx3xM

        # -- Pose and coefficients branch --
        x_init_augmented = torch.cat((x_stack, sn_stack), dim=1)
        coeffs, rot = self.instance_branch(x_init_augmented, epoch)

        return nodes, coeffs, rot



    