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


def compute_date_n(N, p, e, t):
    return N / (1 + e**2 * (N-1)/(t**2 * p * (1-p)))


def main(layer_start=0, layer_end=-1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)
    print(f'Running on device {device}')

    _, _, test_loader = load_CIFAR10_datasets(test_batch_size=10)

    resnet = resnet20()
    resnet.to(device)
    load_from_dict(network=resnet,
                   device=device,
                   path='models/pretrained_models/resnet20-trained.th')

    resnet_layers = [m for m in resnet.modules() if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear)]
    if layer_end == -1:
        resnet_layers = resnet_layers[layer_start:]
    else:
        resnet_layers = resnet_layers[layer_start:layer_end]
    resnet_layers_shape = [layer.weight.shape for layer in resnet_layers]

    fault_injection(net=resnet,
                    net_name='resnet20',
                    net_layer_shape=resnet_layers_shape,
                    loader=test_loader,
                    device=device)


def fault_injection(net,
                    net_name,
                    net_layer_shape,
                    loader,
                    device,
                    seed=51195,
                    p=0.5,
                    e=0.01,
                    t=2.58):

    cwd = os.getcwd()
    os.makedirs(f'{cwd}/fault_injection/{net_name}', exist_ok=True)

    exhaustive_fault_list = []
    for layer, layer_shape in enumerate(tqdm(net_layer_shape, desc='Generating fault list')):

        if len(layer_shape) == 4:
            k = np.arange(layer_shape[0])
            dim1 = np.arange(layer_shape[1])
            dim2 = np.arange(layer_shape[2])
            dim3 = np.arange(layer_shape[3])
            bits = np.arange(0, 32)

            exhaustive_fault_list = exhaustive_fault_list + list(itertools.product(*[[layer], k, dim1, dim2, dim3, bits]))
        else:
            k = np.arange(layer_shape[0])
            dim1 = np.arange(layer_shape[1])
            bits = np.arange(0, 32)

            exhaustive_fault_list = exhaustive_fault_list + list(itertools.product(*[[layer], k, dim1, [], [], bits]))

    random_generator = np.random.default_rng(seed=seed)
    n = compute_date_n(N=len(exhaustive_fault_list),
                       p=p,
                       e=e,
                       t=t)
    fault_list = random_generator.choice(exhaustive_fault_list, int(n))
    del exhaustive_fault_list

    fault_list_filename = f'{cwd}/fault_injection/fault_list/{net_name}'
    os.makedirs(fault_list_filename, exist_ok=True)
    with open(f'{fault_list_filename}/{seed}_fault_list.csv', 'w', newline='') as f_list:
        writer_fault = csv.writer(f_list)
        writer_fault.writerow(['Injection',
                               'Layer',
                               'k',
                               'dim1',
                               'dim2',
                               'dim3',
                               'bit'])
        for index, fault in enumerate(fault_list):
            if len(fault) < 6:
                writer_fault.writerow(np.concatenate([[index], [int(fault[0])], [int(fault[1])], [int(fault[2])], [np.nan], [np.nan], [int(fault[3])]]))
            else:
                writer_fault.writerow(np.concatenate([[index], fault]))

    with torch.set_grad_enabled(False):
        net.eval()

        filename_post = f'{cwd}/fault_injection/{net_name}/{seed}_post_softmax.csv'
        filename_pre = f'{cwd}/fault_injection/{net_name}/{seed}_pre_softmax.csv'

        with open(filename_post, 'w', newline='') as f_inj_post,\
                open(filename_pre, 'w', newline='') as f_inj_pre:
            writer_inj_post = csv.writer(f_inj_post)
            writer_inj_pre = csv.writer(f_inj_pre)
            title_row = ['Injection',
                         'Layer',
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
                         'Class_9_Score',
                         'Golden',
                         'Bit',
                         'NoChange']
            writer_inj_post.writerow(title_row)
            writer_inj_pre.writerow(title_row)
            f_inj_post.flush()
            f_inj_pre.flush()

            pbar = tqdm(fault_list, desc='Fault injection campaign')
            for injection_index, fault in enumerate(pbar):
                layer = fault[0]
                bit = fault[-1]

                layer_index = layer

                pfi_model = BitFlipFI(net,
                                      fault_location=fault,
                                      batch_size=1,
                                      input_shape=[3, 32, 32],
                                      layer_types=["all"],
                                      use_cuda=(device == 'cuda'))

                corrupt_net = pfi_model.declare_weight_bit_flip()

                for image_index, data in enumerate(loader):
                    x, y_true = data
                    x, y_true = x.to(device), y_true.to(device)

                    y_pred = corrupt_net(x)

                    output_list = []
                    for index in range(0, len(y_pred)):
                        output_list.append(np.concatenate([[injection_index],
                                                           [layer_index],
                                                           [image_index * loader.batch_size + index],
                                                           y_pred[index].cpu().numpy(),
                                                           [y_true[index].cpu()],
                                                           [bit],
                                                           [False]]))
                    writer_inj_pre.writerows(output_list)

                    softmax = torch.nn.Softmax(dim=1)
                    y_pred = softmax(y_pred)
                    output_list = []
                    for index in range(0, len(y_pred)):
                        output_list.append(np.concatenate([[injection_index],
                                                           [layer_index],
                                                           [image_index * loader.batch_size + index],
                                                           y_pred[index].cpu().numpy(),
                                                           [y_true[index].cpu()],
                                                           [bit],
                                                           [False]]))
                    writer_inj_post.writerows(output_list)

                    f_inj_post.flush()
                    f_inj_pre.flush()


if __name__ == '__main__':

    main()
