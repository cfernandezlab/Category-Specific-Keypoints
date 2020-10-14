import time
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
from models.basis_keypoint_detector import ModelDetector
from data.datasets_loader import KeypointsDataset
from models import options_detector

opt = options_detector.Options().parse()

chkpt_path = os.path.join(opt.checkpoints_dir, 'checkpoint.tar')

log_fout = open(os.path.join(opt.checkpoints_dir, 'log_train.txt'), 'a')
log_fout.write(str(opt)+'\n')
def log_string(out_str):
    log_fout.write(out_str+'\n')
    log_fout.flush()
    print(out_str)


if __name__=='__main__':

    root = opt.data_dir 
    trainset = KeypointsDataset(root, 'train', opt)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads, drop_last=True)
    print('#training point clouds = %d' % len(trainset))
    
    testset = KeypointsDataset(root, 'test', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    print('#test point clouds = %d' % len(testset))

    # create model, optionally load pre-trained model
    model = ModelDetector(opt)

    start_epoch = 0   
    if chkpt_path is not None and os.path.isfile(chkpt_path):
        checkpoint = torch.load(chkpt_path)
        model.detector.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer_detector.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        log_string("-> loaded checkpoint %s (epoch: %d)"%(chkpt_path, start_epoch))

    
    best_loss = 1e6
    for epoch in range(start_epoch, 280):

        epoch_iter = 0
        for i, data in enumerate(trainloader):
            iter_start_time = time.time()
            epoch_iter += opt.batch_size

            ob1_pc, ob1_sn, ob1_node, R, angles, _ = data

            model.set_input(ob1_pc, ob1_sn, ob1_node, R, angles)

            model.optimize(epoch=epoch)
            
            batch_interval = 10
            if i % batch_interval == 0 and i > 0:
                t = (time.time() - iter_start_time) / opt.batch_size
                errors = model.get_current_errors()

        
        # test network: print some params
        if epoch >= 0 and epoch%1==0:
            batch_amount = 0
            model.test_loss_average.zero_()
            model.test_chamfer_average.zero_()
            model.test_inclusivity_average.zero_()
            model.test_coverage_average.zero_()

            for i, data in enumerate(testloader):

                ob1_pc, ob1_sn, ob1_node, R, angles, _ = data

                model.set_input(ob1_pc, ob1_sn, ob1_node, R, angles)

                model.test_model()

                batch_amount += ob1_pc.size()[0]

                # accumulate loss
                model.test_loss_average += model.loss.detach() * ob1_pc.size()[0] 


            # update best loss
            model.test_loss_average /= batch_amount
            
            if model.test_loss_average.item() <= best_loss:
                best_loss = model.test_loss_average.item()
            log_string('Tested network. So far best loss: %f' % best_loss)
            log_string('Tested network. loss: %f' % model.test_loss_average.item())


            """loss_incl = 2*model.test_inclusivity_average.item()
            loss_cov = 100*model.test_coverage_average.item()
            loss_chf = model.test_chf_average.item()

            log_string('Tested network. incl loss: %f' % loss_incl)
            log_string('Tested network. cov loss: %f' % loss_cov)
            log_string('Tested network. chf loss: %f' % loss_chf)"""

            # save models
            # if (model.test_loss_average.item() <= best_loss + 1e-5) and (model.test_chamfer_average.item() < 0.1) and (epoch>40):
            if True:
                log_string("Saving network... (epoch: %d) "%(epoch))
                # Saving & Loading Model for Inference
                # model.save_network(model.detector, 'detector', 'gpu%d_%d_%d_%f' % (opt.gpu_ids[0], epoch, best_loss, model.test_loss_average.item()), opt.gpu_ids[0])
                # Saving & Loading a General Checkpoint for Inference and/or Resuming Training
                save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                             'model_state_dict' : model.detector.state_dict(),
                             'optimizer_state_dict': model.optimizer_detector.state_dict(),
                             'loss': model.test_loss_average.item()}
                torch.save(save_dict, os.path.join(opt.checkpoints_dir, 'checkpoint.tar'))

        # learning rate decay
        lr_decay_step = 40
        if epoch%lr_decay_step==0 and epoch > 0:
            model.update_learning_rate(0.5)
        # batch normalization momentum decay:
        next_epoch = epoch + 1
        if (opt.bn_momentum_decay_step is not None) and (next_epoch >= 1) and (
                next_epoch % opt.bn_momentum_decay_step == 0):
            current_bn_momentum = opt.bn_momentum * (
            opt.bn_momentum_decay ** (next_epoch // opt.bn_momentum_decay_step))
            log_string('BN momentum updated to: %f' % current_bn_momentum)



