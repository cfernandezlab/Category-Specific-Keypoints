import torch
import torch.nn as nn
import numpy as np
import math
from collections import OrderedDict
import os
import random
from models import networks, losses


def euler2mat(angle):
    """Creates a rotation matrix from a given angle in the y axis. 
    We assume point clouds are usually aligned to the gravity direction."""

    y = angle
    zeros = y.detach() * 0
    ones = zeros.detach() + 1
    cosy = torch.cos(y.detach())
    siny = torch.sin(y.detach())

    ymat = torch.stack(
        [cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1
    ).reshape(angle.size(0), 3, 3)  # Bx3x3

    return ymat


def get_reflection_operator(n_pl):
    """ The reflection operator is parametrized by the normal vector 
    of the plane of symmetry passing through the origin. """
    norm_npl = torch.norm(n_pl, 2)
    n_x = n_pl[0, 0] / norm_npl # torch.tensor(1.0).cuda()
    n_y = torch.tensor(0.0).cuda()
    n_z = n_pl[0, 1] / norm_npl
    refl_mat = torch.stack(
        [
            1 - 2 * n_x * n_x,
            -2 * n_x * n_y,
            -2 * n_x * n_z,
            -2 * n_x * n_y,
            1 - 2 * n_y * n_y,
            -2 * n_y * n_z,
            -2 * n_x * n_z,
            -2 * n_y * n_z,
            1 - 2 * n_z * n_z,
        ],
        dim=0,
    ).reshape(1, 3, 3)  

    return refl_mat


def get_category_specific_keypoints(c, basis, n_pl, angles, angle_n_pl, symtype="shape", misalign = False):
    """The category-specific symmetric 3D keypoints are computed with the deformation function.

    Arguments:
        c {torch.Tensor} -- predicted def coefficients - BxK
        basis {torch.Tensor} -- basis shapes, free optimizable variable - 1x3xNxK
        n_pl {torch.Tensor} -- normal vector of the plane of symmetry passing through the origin
        angles {torch.Tensor} -- estimated rotation wrt the plane of symmetry
        angle_n_pl {torch.Tensor} -- estimated rotation of the plane of symmetry
        
    Keyword Arguments:
        symtype {str} -- defines the symmetric deformation space "shape" or "basis" (default: {"shape"})

    Returns:
        torch.Tensor -- shape: category-specific symmetric 3D keypoints
    """
    refl_mat = get_reflection_operator(n_pl)
    basis_half = basis
    c = c.unsqueeze_(1).unsqueeze_(1)  # Bx1x1xK

    if symtype == "shape":
        refl_batch = refl_mat.repeat(c.shape[0], 1, 1)
        kpts_half = torch.sum(c * basis_half, 3)  # Bx3xM
        kpts_half_reflected = torch.matmul(refl_batch, kpts_half)

    elif symtype == "basis":
        refl_batch = refl_mat.unsqueeze(0).repeat(basis_half.shape[0], 1, 1, 1)
        basis_half_reflected = torch.matmul(
            refl_batch, torch.transpose(torch.transpose(basis_half, 1, 3), 2, 3),
        )
        basis_half_reflected = torch.transpose(torch.transpose(basis_half_reflected, 1, 3), 1, 2)
        c1 = c[:, :, :, 0 : c.shape[3] // 2]
        c2 = c[:, :, :, c.shape[3] // 2 :]
        kpts_half = torch.sum(c1 * basis_half, 3)
        kpts_half_reflected = torch.sum(c2 * basis_half_reflected, 3)
    
    kpts = torch.cat((kpts_half, kpts_half_reflected), 2)

    if misalign == True:
        R_n_pl = euler2mat(angle_n_pl)
        R = euler2mat(angles)
        kpts = torch.matmul(R_n_pl, kpts)
        kpts = torch.matmul(R, kpts)

    return kpts 


def loss_category_specific_kpts(self, kpts, nodes, pc):
    chf_loss, _, _ = self.chamfer_criteria(kpts, pc)
    cov_loss = self.coverage_criteria(nodes, pc)
    inc_loss = self.inclusivity_criteria(nodes, pc)

    loss = 2 * inc_loss + cov_loss + 1 * chf_loss 

    return loss

            
class ModelDetector():
    def __init__(self, opt):
        self.opt = opt

        self.detector = networks.KP_Detector(opt).to(self.opt.device) 

        self.chamfer_criteria = losses.ChamferLoss_Brute(opt).to(self.opt.device)
        self.inclusivity_criteria = losses.InclusivityLoss(opt).to(self.opt.device)
        self.coverage_criteria = losses.CoverageLoss(opt).to(self.opt.device)

        # learning rate_control
        self.old_lr_detector = self.opt.lr
        self.optimizer_detector = torch.optim.Adam(self.detector.parameters(),
                                                   lr=self.old_lr_detector,
                                                   betas=(0.9, 0.999),
                                                   weight_decay=0)


        # place holder for GPU tensors
        self.pc = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_().to(self.opt.device)
        self.sn = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_().to(self.opt.device)
        self.label = torch.LongTensor(self.opt.batch_size).fill_(1).to(self.opt.device)
        self.node = torch.FloatTensor(self.opt.batch_size, 3, self.opt.node_num).to(self.opt.device)

        # record the test loss and accuracy
        self.test_chamfer_average = torch.tensor([0], dtype=torch.float32, requires_grad=False).to(self.opt.device)
        self.test_loss_average = torch.tensor([0], dtype=torch.float32, requires_grad=False).to(self.opt.device)
        self.test_inclusivity_average = torch.tensor([0], dtype=torch.float32, requires_grad=False).to(self.opt.device)
        self.test_coverage_average = torch.tensor([0], dtype=torch.float32, requires_grad=False).to(self.opt.device) 


    def set_input(self, pc, sn, node, R, angles):

        self.pc = pc.float().to(self.opt.device).detach()
        self.sn = sn.float().to(self.opt.device).detach()
        self.node = node.float().to(self.opt.device).detach()
        self.R = R.float().to(self.opt.device)#.detach()
        self.angles = angles.float().to(self.opt.device).detach()
        
        torch.cuda.synchronize()


    def forward(self, pc, sn, node, is_train=False, epoch=None):
        with torch.cuda.device(pc.get_device()):
            nodes, coefs, rot = self.detector(pc, sn, node, is_train, epoch)  # Bx1024
            return nodes, coefs, rot


    def optimize(self, epoch=None):
        with torch.cuda.device(self.pc.get_device()):

            if self.opt.random_pc_dropout_lower_limit < 0.99:
                dropout_keep_ratio = random.uniform(self.opt.random_pc_dropout_lower_limit, 1.0)
                resulting_pc_num = round(dropout_keep_ratio*self.opt.input_pc_num)
                chosen_indices = np.random.choice(self.opt.input_pc_num, resulting_pc_num, replace=False)
                chosen_indices_tensor = torch.from_numpy(chosen_indices).to(self.opt.device)

                self.pc = torch.index_select(self.pc, dim=2, index=chosen_indices_tensor)
                self.sn = torch.index_select(self.sn, dim=2, index=chosen_indices_tensor)

            self.detector.train()

            self.nodes, self.coefs, self.rot = self.forward(self.pc, self.sn, self.node, is_train=True, epoch=epoch)

            self.detector.zero_grad()

            if 'R_n_pl' not in [name for name, params in self.detector.state_dict().items()]:
                self.detector.R_n_pl = torch.nn.Parameter((torch.rand(1)-0.5).cuda())

            self.kpts = get_category_specific_keypoints(
                self.coefs,
                self.detector.basis,
                self.detector.n_pl,
                self.rot,
                self.detector.R_n_pl,
                "shape",
                self.opt.misalign,
            )

            self.loss = loss_category_specific_kpts(self, self.kpts, self.nodes, self.pc)

            self.loss.backward()

            self.optimizer_detector.step()


    def test_model(self):
        self.detector.eval()
        
        self.nodes, self.coefs, self.rot = self.forward(self.pc, self.sn, self.node, is_train=False)

        self.detector.zero_grad()

        if 'R_n_pl' not in [name for name, params in self.detector.state_dict().items()]:
            self.detector.R_n_pl = torch.nn.Parameter((torch.rand(1)-0.5).cuda())

        self.kpts = get_category_specific_keypoints(
            self.coefs,
            self.detector.basis,
            self.detector.n_pl,
            self.rot,
            self.detector.R_n_pl,
            "shape",
            self.opt.misalign,
        )

        self.loss = loss_category_specific_kpts(self, self.kpts, self.nodes, self.pc)

  
    def freeze_model(self):
        for p in self.detector.parameters():
            p.requires_grad = False


    def run_model(self, pc, sn, node):
        self.detector.eval()
        with torch.no_grad():
            nodes, coefs, rot = self.forward(pc, sn, node, is_train=False, epoch=None)
        return nodes, coefs, rot


    def get_current_errors(self):
        return OrderedDict([
            ('loss', self.loss.item()),
            ('av_loss', self.test_loss_average.item()),
            ('incl_loss', self.test_inclusivity_average.item()),
            ('cov_loss', self.test_coverage_average.item()),
            ('chf_loss', self.test_chamfer_average.item()),
        ])


    def save_network(self, network, network_label, epoch_label, gpu_id):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        

    def update_learning_rate(self, ratio):
        lr_clip = 0.00001
        lr_detector = self.old_lr_detector * ratio
        if lr_detector < lr_clip:
            lr_detector = lr_clip
        for param_group in self.optimizer_detector.param_groups:
            param_group['lr'] = lr_detector
        print('update detector learning rate: %f -> %f' % (self.old_lr_detector, lr_detector))
        self.old_lr_detector = lr_detector
