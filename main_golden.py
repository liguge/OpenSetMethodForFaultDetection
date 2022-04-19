import os
import csv
import itertools
import numpy as np
from tqdm import tqdm, trange

import torch

import argparse

from models.utils import load_CIFAR10_datasets, load_from_dict
from models.resnet import resnet20

from BitFlipFI import BitFlipFI


dataset = 'test'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)
print(f'Running on device {device}')

train_loader, _, test_loader = load_CIFAR10_datasets(train_batch_size=10, test_batch_size=10)
if dataset == 'test':
    loader = test_loader
else:
    loader = train_loader

resnet = resnet20()
resnet.to(device)
load_from_dict(network=resnet,
               device=device,
               path='models/pretrained_models/resnet20-trained.th')

resnet_layers = [m for m in resnet.modules() if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear)]
resnet_layers_shape = [layer.weight.shape for layer in resnet_layers]

with torch.set_grad_enabled(False):
    resnet.eval()

    filename_post = f'fault_injection/resnet20/golden_{dataset}_post_softmax.csv'
    filename_pre = f'fault_injection/resnet20/golden_{dataset}_pre_softmax.csv'

    with open(filename_post, 'w', newline='') as f_inj_post,\
            open(filename_pre, 'w', newline='') as f_inj_pre:
        writer_inj_post = csv.writer(f_inj_post)
        writer_inj_pre = csv.writer(f_inj_pre)
        title_row = [
            'ImageIndex',
            'Class_0_Score',
            'Class_1_Score',
            'Class_2_Score',
            'Class_3_Score',
            'Class_4_Score',
            'Class_5_Score',
            'Class_6_Score',
            'Class_7_Score',
            'Class_8_Score',
            'Class_9_Score']
        writer_inj_post.writerow(title_row)
        writer_inj_pre.writerow(title_row)
        f_inj_post.flush()
        f_inj_pre.flush()


        for image_index, data in enumerate(loader):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            y_pred = resnet(x)

            output_list = []
            for index in range(0, len(y_pred)):
                output_list.append(np.concatenate([
                    [image_index * loader.batch_size + index],
                    y_pred[index].cpu().numpy()
                ]))
            writer_inj_pre.writerows(output_list)

            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            output_list = []
            for index in range(0, len(y_pred)):
                output_list.append(np.concatenate([
                    [image_index * loader.batch_size + index],
                    y_pred[index].cpu().numpy(),
                ]))
            writer_inj_post.writerows(output_list)

            f_inj_post.flush()
            f_inj_pre.flush()

