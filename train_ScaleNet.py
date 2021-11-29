import numpy as np
import argparse
import time
import random
import os
from os import path as osp
import pickle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from assets.training_utils.training_dataset.scalenet_dataset import ScaleNet_dataset as scale_dataset
from assets.training_utils.tools.training_utils import train_epoch, validate_epoch
from ScaleNet.scalenet import scalenet_network as scalenet
from assets.training_utils.tools.utils import save_to_file, load_ckp

def train_scalenet():

    """
        Training script to train ScaleNet scale estimator network
    """

    # Argument parsing
    parser = argparse.ArgumentParser(description='Scale-Net train script')

    # Paths
    parser.add_argument('--image_data_path', type=str, default='path-to-megadepth-d2net',
                        help='path to dataset images')
    parser.add_argument('--pairs_path', type=str, default='assets/data/',
                        help='path to the txt files containing image pairs')
    parser.add_argument('--root_precomputed_files', type=str, default='assets/data/tmp_data/',
                        help='path to store precomputed image pairs. It stores the center-cropped images instead '
                             'of generating them every epoch, which is faster during training')
    parser.add_argument('--save_processed_im', type=bool, default=True,
                        help='path to store precomputed image pairs. It stores the center-cropped images instead '
                             'of generating them every epoch, which is faster during training')
    parser.add_argument('--model_name', type=str, default='scaleNet_default', help='Model to use')
    parser.add_argument('--snapshots', type=str, default='./snapshots/')
    parser.add_argument('--logs', type=str, default='./logs')

    # Optimization parameters
    parser.add_argument('--lr', type=float, default=10e-5, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum constant')
    parser.add_argument('--start_epoch', type=int, default=-1, help='start epoch')
    parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='training batch size')
    parser.add_argument('--n_threads', type=int, default=6, help='number of parallel threads for dataloaders')

    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay constant')
    parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')
    parser.add_argument('--resume_training', type=bool, default=False, help='resume_training')
    parser.add_argument('--checkpoint_fpath', type=str, default='.', help='Checkpoint Path')
    parser.add_argument('--cuda_device', type=str, default='0', help='Indicates which GPU should be used')
    parser.add_argument('--is_debug', type=bool, default=False, help='Indicates if debugging')

    # Network configuration
    parser.add_argument('--extractor', type=str, default='VGG', help='Indicates the feature extractor')
    parser.add_argument('--scale_levels', type=int, default=13, help='Indicates the number of bins in the scale '
                                                                     'distribution output')
    # model & loss configuration
    parser.add_argument('--add_corrA', type=bool, default=True, help='Indicates if corrA should be added to model')
    parser.add_argument('--add_corrB', type=bool, default=True, help='Indicates if corrB should be added to model')
    parser.add_argument('--multi_scale', type=bool, default=True, help='Indicates if use multiscale features')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device


    # Create directories for logs and weights
    if not os.path.isdir(args.snapshots):
        os.mkdir(args.snapshots)
    args.snapshots += args.model_name
    if not os.path.isdir(args.snapshots):
        os.mkdir(args.snapshots)

    cur_snapshot = time.strftime('%Y_%m_%d_%H_%M')

    save_path = osp.join(args.snapshots, cur_snapshot)

    if not osp.isdir(save_path):
        os.mkdir(save_path)

    with open(osp.join(save_path, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    log_writer = os.path.join(save_path, 'log.txt')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define model
    model = scalenet(extractor=args.extractor, multi_scale=args.multi_scale,
                     is_test=False, device=device, scale_levels=args.scale_levels,
                     add_corr_a=args.add_corrA, add_corr_b=args.add_corrB)

    print('Scale-Net created.')
    print('Save in: ' + save_path)

    # Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15, 30], gamma=0.1)

    if args.resume_training:
        model, optimizer, epoch_start = load_ckp(args.checkpoint_fpath, model, optimizer, device, strict=False)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15, 30], gamma=0.5)

    # Load training and validation datasets
    train_dataset = scale_dataset(image_path=args.image_data_path, pairs_path=args.pairs_path + 'train_pairs.txt',
                                  is_debug=args.is_debug, is_training=True, scale_levels=args.scale_levels,
                                  root_np=args.root_precomputed_files)

    val_dataset = scale_dataset(image_path=args.image_data_path, pairs_path=args.pairs_path + 'val_pairs.txt',
                                is_debug=args.is_debug, is_training=False, scale_levels=args.scale_levels,
                                root_np=args.root_precomputed_files)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)

    # Init training variables
    model = model.to(device)
    patience = 10
    count = 0.
    prev_model = None
    train_started = time.time()
    epoch_start = 0

    # Start training loop
    for epoch in range(epoch_start, args.n_epoch):

        # Training one epoch
        train_loss, diff_scale_train = train_epoch(model, optimizer, train_dataloader, train_dataset.scale_distr, device)

        # Validation
        val_loss, diff_scale_val = validate_epoch(model, val_dataloader, train_dataset.scale_distr, device)

        scheduler.step()

        # Tensorboard and log save
        train_writer.add_scalar('train_loss', train_loss, epoch)
        train_writer.add_scalar('val_loss', val_loss, epoch)
        train_writer.add_scalar('train_diff_scale', diff_scale_train, epoch)
        train_writer.add_scalar('val_diff_scale', diff_scale_val, epoch)

        info_log = "\nEpoch: {}. Loss: {:.3f}. Val loss: {:.3f}. Diff scale: {:.3f}. Val diff scale: {:.3f}\n".\
            format(epoch+1, train_loss, val_loss, diff_scale_train, diff_scale_val)
        save_to_file(info_log, log_writer)

        if epoch > args.start_epoch:
            '''
            We will be saving only the snapshot which
            has lowest loss value on the validation set
            '''
            cur_name = osp.join(args.snapshots, cur_snapshot, 'epoch_{}.pth'.format(epoch + 1))

            if prev_model is None:
                torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, cur_name)
                prev_model = cur_name
                best_ratio_val = 10e6
            else:
                if diff_scale_val < best_ratio_val:
                    count = 0.
                    best_ratio_val = diff_scale_val
                    os.remove(prev_model)
                    save_to_file('Saved snapshot: {}\n'.format(cur_name), log_writer)
                    torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, cur_name)
                    prev_model = cur_name
                else:
                    count += 1

        # Early stop check
        if count >= patience:
            info_log = '\nPatience reached ({}). Best model: {}. Stop Training.'.format(count, prev_model)
            save_to_file(info_log, log_writer)
            break

    print(args.seed, 'Training took:', time.time()-train_started, 'seconds')


if __name__ == '__main__':

    train_scalenet()