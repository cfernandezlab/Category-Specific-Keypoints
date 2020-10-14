import os
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
from models.basis_keypoint_detector import ModelDetector
from data.datasets_loader import KeypointsDataset
from models import options_detector

opt_detector = options_detector.Options().parse()

if opt_detector.dataset not in ['ModelNet10', 'ShapeNet', 'dfaust', 'faces', 'sunrgbd']:
    raise Exception('dataset error.') 
if opt_detector.ckpt_model is None:
    raise Exception('model checkpoint not specified.') 

# set paths according to dataset
root = opt_detector.data_dir 
detector_model_path = root + opt_detector.dataset + '/checkpoints/' + opt_detector.ckpt_model + '/checkpoint.tar' 
output_folder = root + opt_detector.dataset + '/results/' + opt_detector.ckpt_model + '/' 

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)


if __name__ == '__main__':

    testset = KeypointsDataset(root, 'test', opt_detector)

    testloader = torch.utils.data.DataLoader(testset, batch_size=opt_detector.batch_size,
                                             shuffle=False, num_workers=opt_detector.nThreads, pin_memory=False)

    # load model
    model_detector = ModelDetector(opt_detector)
    checkpoint = torch.load(detector_model_path)
    model_detector.detector.load_state_dict(checkpoint['model_state_dict'])
    model_detector.optimizer_detector.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(detector_model_path, epoch))
    model_detector.freeze_model()

    keypoint_num_list = []
    for i, data in enumerate(testloader):
        print(i)
        anc_pc, anc_sn, anc_node, R, angles, anc_idx = data 

        # add noise on anc_pc
        noise_sigma = 0
        anc_pc = anc_pc + torch.randn(anc_pc.size()) * noise_sigma

        anc_pc_cuda = anc_pc.to(opt_detector.device)
        anc_sn_cuda = anc_sn.to(opt_detector.device)
        anc_node_cuda = anc_node.to(opt_detector.device)

        # run detection
        anc_nodes, anc_coefs, anc_rot = model_detector.run_model(anc_pc_cuda, anc_sn_cuda, anc_node_cuda) 
        anc_nodes_np = anc_nodes.detach().permute(0, 2, 1).contiguous().cpu().numpy()   
        anc_coefs_np = anc_coefs.detach().cpu().numpy()
        anc_rot_np = anc_rot.detach().cpu().numpy()
        sh_basis = model_detector.detector.basis.detach().cpu().numpy()
        basis_shapes = []
        for k in range(sh_basis.shape[3]):
            sh_base_k = sh_basis[0, :, :, k]
            basis_shapes.append(sh_base_k)
        sh_npl = model_detector.detector.n_pl.detach().cpu().numpy() 
        #sh_Rnpl = model_detector.detector.R_n_pl.detach().cpu().numpy() # =1

        batch_size_dyn = anc_pc.size(0)
        for b in range(batch_size_dyn): 
            frame_pc_np = np.transpose(anc_pc[b].detach().numpy()) 
            frame_nodes_np = anc_nodes_np[b]
            frame_coef_np = anc_coefs_np[b]
            frame_rot_np = anc_rot_np[b]

            # write results to file
            output_file = os.path.join(output_folder, '%s.mat' % anc_idx[b]) 
            sio.savemat(output_file, {'defCoefs':frame_coef_np, 'BasisShapes': basis_shapes, 'n_pl': sh_npl, 'nodes':frame_nodes_np, 'rot':frame_rot_np}) # 'Rn_pl': sh_Rnpl, 

    print("output folder is: %s" % output_folder)

