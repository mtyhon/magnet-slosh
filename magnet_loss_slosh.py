import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
import argparse
from math import ceil
from tqdm import tqdm


import torch.nn as nn
from torch import optim
from sklearn.model_selection import train_test_split
import torch.utils.data as utils
import os, re
from magnet_loss.magnet_tools import *
from magnet_loss.magnet_loss import MagnetLoss
from magnet_loss.utils import plot_embedding, plot_smooth
from utils.average_meter import AverageMeter
from visualizer.visualizer import VisdomLinePlotter
from utils.slosh_utils import *
from utils.sampler import SubsetSequentialSampler

def parse_settings():
    # Training settings
    parser = argparse.ArgumentParser(description='SLOSH with Magnet Loss')
    print(parser)
    package_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--name', default='MagnetLoss', type=str,
						help='name of experiment')
    parser.add_argument('--lr', default=0.0005, type=float,
						help='Optimizer initial learning rate')
    parser.add_argument('--root_folder', default=package_dir +'/data/starclass_image_StandardScaled/', type=str,
						help='Folder with SLOSH input data')
    parser.add_argument('--model_folder', default=package_dir +'/saved_models/', type=str,
						help='Folder to keep trained models')
    return parser.parse_args()

def run_magnet_loss(args):

    m = 6
    d = 6
    k = 6
    alpha = 1.0
    batch_size = m * d

    root_folder = args.root_folder
    print('ROOT FOLDER: ', root_folder)
    folder_filenames = []
    file_kic = []
    labels = []

    print('Parsing files in folder... ')
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for i, filex in enumerate(
                filenames):
            if filex.endswith('.npz'):
                folder_filenames.append(os.path.join(dirpath, filex))
                kicx = int(re.search(r'\d+', filex).group())
                file_kic.append(kicx)
                labels.append(np.load(os.path.join(dirpath, filex))['label'])

    file_kic = np.array(file_kic)
    folder_filenames = np.array(folder_filenames)
    labels = np.array(labels)
    print('folder filenames: ')
    train_ids, val_ids, train_labels, val_labels = train_test_split(folder_filenames, labels, stratify=labels,
                                                                    test_size=0.15, random_state=137)

    train_labels = labels[np.in1d(folder_filenames, train_ids)]
    val_labels = labels[np.in1d(folder_filenames, val_ids)]
    train_filenames = folder_filenames[np.in1d(folder_filenames, train_ids)]
    val_filenames = folder_filenames[np.in1d(folder_filenames, val_ids)]


    print('Total Files: ', len(file_kic))

    print('Train Unique IDs: ', len(train_ids))
    print('Setting up generators... ')

    train_gen = NPZ_Dataset(filenames=train_filenames, labels=train_labels)
    train_sampler = SubsetSequentialSampler(range(len(train_gen)), range(m*d))
    train_dataloader = utils.DataLoader(train_gen,num_workers=1, batch_size = m*d,shuffle=False,
								 sampler=train_sampler)

    val_gen = NPZ_Dataset(filenames=val_filenames, labels=val_labels)
    val_dataloader = utils.DataLoader(val_gen, num_workers=4)

    trainloader, testloader, trainset, testset = train_dataloader, val_dataloader, train_gen, val_gen

    emb_dim = 8
    epoch_steps = len(trainloader)
    n_steps = epoch_steps * 50 * 2
    cluster_refresh_interval = epoch_steps
    model =  SLOSH_Embedding(embed_size=emb_dim)
    model.cuda()
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    minibatch_magnet_loss = MagnetLoss()

    # Get initial embedding
    initial_reps = compute_reps(model, train_gen, 400)

    # Create batcher
    batch_builder = ClusterBatchBuilder(train_labels, k, m, d)
    batch_builder.update_clusters(initial_reps)

    batch_losses = []

    batch_example_inds, batch_class_inds = batch_builder.gen_batch()
    trainloader.sampler.batch_indices = batch_example_inds

    model.train()

    losses = AverageMeter()
  
    for i in tqdm(range(n_steps)):
        for batch_idx, (img, target) in enumerate(trainloader):
            img = img.cuda().float()
            optimizer.zero_grad()
            output, _ = model(img)
            batch_loss, batch_example_losses = minibatch_magnet_loss(output,
                                                                    batch_class_inds,
                                                                    m,
                                                                    d,
                                                                    alpha)
            batch_loss.backward()
            optimizer.step()

        # Update loss index
        batch_builder.update_losses(batch_example_inds,
                                    batch_example_losses)

        batch_losses.append(batch_loss.item()) 

        if not i % 1000:
            print ('Epoch %d, Loss: %.4f' %(i, batch_loss))

        if not i % cluster_refresh_interval:
            print("Refreshing clusters")
            reps = compute_reps(model, trainset, 400)
            batch_builder.update_clusters(reps)

        if not i % 2000:
            n_plot = 10000
            plot_embedding(X=compute_reps(model, trainset, 400)[:n_plot],
                           y=train_labels[:n_plot], name=str(i) +'_train_embed%d' %emb_dim,
                           save_embed=True, filename=train_filenames, batch_builder=batch_builder)
            plot_embedding(X=compute_reps(model, testset, 400)[:n_plot],
                           y=val_labels[:n_plot], name=str(i) +'__val_embed%d' %emb_dim,
                           save_embed=False, filename=val_filenames)
            torch.save(model.state_dict(), args.model_folder + str(i) + "_train_embed%d.torchmodel" %emb_dim)


        batch_example_inds, batch_class_inds = batch_builder.gen_batch()
        trainloader.sampler.batch_indices = batch_example_inds

        losses.update(batch_loss, 1)



def main():
    args = parse_settings()
    run_magnet_loss(args)

if __name__ == '__main__':
    main()

