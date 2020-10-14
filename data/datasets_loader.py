import torch.utils.data as data
import random
import os
import numpy as np
import torch
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,'../data'))
from helper import transform_pc_pytorch
from pathlib import Path
import glob


def make_dataset(root, mode, opt):
    folder_name = root + opt.dataset + '/' + mode + '_data_npy/' + opt.category
    list_el = glob.glob(os.path.join(folder_name, '*.npy'))

    dataset = []
    for i, name in enumerate(list_el):
        idx = Path(name).stem 
        item = (name, idx)
        dataset.append(item)

    return dataset


class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts


class KeypointsDataset(data.Dataset):
    def __init__(self, root, mode, opt):
        super(KeypointsDataset, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode

        if self.opt.dataset not in ['ModelNet10', 'ShapeNet', 'dfaust', 'faces', 'sunrgbd']:
            raise Exception('dataset error.') 
        
        self.dataset = make_dataset(self.root, mode, opt)

        self.fathest_sampler = FarthestSampler()

    def __len__(self,):
        return len(self.dataset)

    def get_instance_unaugmented_np(self, index):
        pc_np_file, idx = self.dataset[index] 

        data = np.load(pc_np_file)
        data = data[np.random.choice(data.shape[0], self.opt.input_pc_num, replace=False), :]

        pc_np = data[:, 0:3]  # Nx3 point coordinates
        sn_np = data[:, 3:6]  # Nx3 surface normal

        node_np = self.fathest_sampler.sample(
            pc_np[np.random.choice(pc_np.shape[0], int(self.opt.input_pc_num / 4), replace=False)],
            self.opt.node_num,
        )

        return pc_np, sn_np, node_np, idx


    def __getitem__(self, index):
        pc_np, sn_np, node_np, idx = self.get_instance_unaugmented_np(index)

        pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
        sn = torch.from_numpy(sn_np.transpose().astype(np.float32))  # 3xN
        node = torch.from_numpy(node_np.transpose().astype(np.float32))  # 3xM

        if self.opt.misalign:
            pc, sn, node, R, scale, shift, angles = transform_pc_pytorch(pc, sn, node, rot_type="2d", scale_thre=0, shift_thre=0)
        else:
            R = np.empty([3, 3])
            angles = np.empty([1, 3])

        return pc, sn, node, R, angles, idx