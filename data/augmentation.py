import random
import numpy as np
import torch
from pyntcloud import PyntCloud
import trimesh


def read_data(name, dataset):
    if dataset == 'ShapeNet':
        # read .pts
        cloud = PyntCloud.from_file(name, sep=" ", header=0, names=["x","y","z"])
        return np.asarray(cloud.points)
    elif dataset == 'ModelNet10':
        # read .off, .obj, .ply
        mesh = trimesh.load(name)
        return np.asarray(mesh.vertices)


def normalize_data(pc):
    """ center models and normalize [-1,1].""" 
    pc_shift = np.sum(pc, axis = 0)/len(pc)
    pc = pc - pc_shift
    dimX = np.max(pc[:,0]) - np.min(pc[:,0])
    dimY = np.max(pc[:,1]) - np.min(pc[:,1])
    dimZ = np.max(pc[:,2]) - np.min(pc[:,2])
    scale = 2/np.max([dimX, dimY, dimZ])
    pc = pc*scale
    return pc


def angles2rotation_matrix(angles):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def R2angles(R):
    x_angle = 0
    y_angle = np.arctan2(R[0][2], R[0][0])
    z_angle = 0
    angles = np.array([x_angle, y_angle, z_angle])
    return angles


def transform_pc_pytorch(pc, sn, node, rot_type='2d', scale_thre=0.2, shift_thre=0.2, rot_perturbation=False):
    '''

    :param pc: 3xN tensor
    :param sn: 5xN tensor / 4xN tensor
    :param node: 3xM tensor
    :return: pc, sn, node of the same shape, detach
    '''
    device = pc.device

    # 1. rotation (2d: up axis - y, or 3d)
    if rot_type == '2d':
        x_angle, z_angle = 0, 0
        y_angle = (np.random.uniform()-0.5) * np.pi/4 # /12-15 /6-30  /4-45 /3-60
    elif rot_type == '3d':
        x_angle = np.random.uniform() * 2 * np.pi
        y_angle = np.random.uniform() * 2 * np.pi
        z_angle = np.random.uniform() * 2 * np.pi
    elif rot_type is None:
        x_angle, y_angle, z_angle = 0, 0, 0
    else:
        raise Exception('Invalid rot_type.')

    if rot_perturbation == True:
        angle_sigma = 0.06
        angle_clip = 3 * angle_sigma
        x_angle += np.clip(angle_sigma * np.random.randn(), -angle_clip, angle_clip)
        y_angle += np.clip(angle_sigma * np.random.randn(), -angle_clip, angle_clip)
        z_angle += np.clip(angle_sigma * np.random.randn(), -angle_clip, angle_clip)

    angles = [x_angle, y_angle, z_angle]
    R = torch.from_numpy(angles2rotation_matrix(angles).astype(np.float32)).to(device)  # 3x3
    angles_Ry = torch.tensor(R2angles(R), dtype=torch.float)
    pc = torch.matmul(R, pc)  # 3x3 * 3xN -> 3xN
    if sn.size()[0] >= 3:
        sn[0:3, :] = torch.matmul(R, sn[0:3, :])  # 3x3 * 3xN -> 3xN
    node = torch.matmul(R, node)  # 3x3 * 3xN -> 3xN

    # 2. scale 
    scale = np.random.uniform(low=1-scale_thre, high=1+scale_thre)
    pc = pc * scale
    node = node * scale

    # 3. translation
    shift = torch.from_numpy(np.random.uniform(-1*shift_thre, shift_thre, (3, 1)).astype(np.float32)).to(device)  # 3x1
    pc = pc + shift
    node = node + shift
    
    return pc.detach(), sn.detach(), node.detach(), \
           R, scale, shift, angles_Ry