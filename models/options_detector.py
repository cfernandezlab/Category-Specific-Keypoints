import argparse
import os
from util import util 
import torch
import GPUtil
import numpy as np


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='auto or gpu_ids seperated by comma.')

        self.parser.add_argument('--data_dir', type=str, default='/home/cajad/scratch/Datasets/', help='folder where all data is stored')
        self.parser.add_argument('--dataset', type=str, default='modelnet', help='ModelNet10, ShapeNet, dfaust, faces, sunrgbd')
        self.parser.add_argument('--category', type=str, default='chair', help='name of the category we are training with')
        
        self.parser.add_argument('--ckpt_model', type=str, default=None, help='name of the trained model') 

        self.parser.add_argument('--batch_size', type=int, default=4, help='input batch size') 
        self.parser.add_argument('--input_pc_num', type=int, default=3000, help='# of input points') 
        self.parser.add_argument('--surface_normal_len', type=int, default=3, help='3 - surface normal, 0 - only xyz coordinates')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data') 

        self.parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
        self.parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')

        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--node_num', type=int, default=16, help='max number of keypoints to predict') 
        self.parser.add_argument('--k', type=int, default=1, help='k nearest neighbor')
        self.parser.add_argument('--node_knn_k_1', type=int, default=3, help='k nearest neighbor of nodes searching on nodes') 
        self.parser.add_argument('--basis_num', type=int, default=10, help='num of basis shapes')

        self.parser.add_argument('--random_pc_dropout_lower_limit', type=float, default=1, help='keep ratio lower limit')
        self.parser.add_argument('--bn_momentum', type=float, default=0.1, help='normalization momentum, typically 0.1. Equal to (1-m) in TF')
        self.parser.add_argument('--bn_momentum_decay_step', type=int, default=None, help='BN momentum decay step. e.g, 0.5->0.01.')
        self.parser.add_argument('--bn_momentum_decay', type=float, default=0.6, help='BN momentum decay step. e.g, 0.5->0.01.')

        self.parser.add_argument("--misalign", type=bool, default=False, help="misaligned input data")

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # === processing options === begin ===
        self.opt.gpu_ids = list(map(int, self.opt.gpu_ids.split(',')))
        self.opt.device = torch.device("cuda:%d" % self.opt.gpu_ids[0] if (torch.cuda.is_available() and len(self.opt.gpu_ids) >= 1) else "cpu")

        # === processing options === end ===

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        self.opt.checkpoints_dir = self.opt.data_dir + self.opt.dataset + '/checkpoints/' + self.opt.ckpt_model 
        expr_dir =  os.path.join(self.opt.checkpoints_dir, 'train')
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
