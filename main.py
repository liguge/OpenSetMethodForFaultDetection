import os
import csv
import itertools
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.autograd import Variable

import argparse

from models.utils import load_CIFAR10_datasets, load_from_dict
from models.resnet import resnet20

from BitFlipFI import BitFlipFI


def odin_inference(corrupt_net, loader, device, injection_index, layer_index, bit, writer_inj_post, f_inj_post, temperature=1000, epsilon=0.0014):
    with torch.set_grad_enabled(True):
        for image_index, data in enumerate(loader):

            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            x = Variable(x.to(device), requires_grad=True)

            y_pred = corrupt_net(x)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)

            # Temperature scaling
            y_pred = y_pred / temperature

            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(y_pred, y_true)
            loss.backward()

            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(x.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # Normalizing the gradient to the same space of image
            gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
            gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
            gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)

            x_noisy = torch.add(x.data, gradient, alpha=-epsilon)
            y_pred_noisy = corrupt_net(Variable(x_noisy))
            y_pred_noisy = y_pred_noisy / temperature
            y_pred_noisy = softmax(y_pred_noisy)

            output_list = []
            for index in range(0, len(y_pred_noisy)):
                output_list.append(np.concatenate([[injection_index],
                                                   [layer_index],
                                                   [image_index * loader.batch_size + index],
                                                   y_pred_noisy[index].cpu().detach().numpy(),
                                                   [y_true[index].cpu()],
                                                   [bit],
                                                   [False]]))
            writer_inj_post.writerows(output_list)

            f_inj_post.flush()


def compute_date_n(N, p, e, t):
    return N / (1 + e**2 * (N-1)/(t**2 * p * (1-p)))


def main(fine_tuning, layer_start=0, layer_end=-1, test_batch_size=128, test_image_per_class=None, force_n=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)
    print(f'Running on device {device}')

    _, _, test_loader = load_CIFAR10_datasets(test_batch_size=test_batch_size, test_image_per_class=test_image_per_class)

    resnet = resnet20()
    resnet.to(device)

    weights_path = 'models/pretrained_models/resnet20-trained.th' if fine_tuning == 'clean' else 'fine_tuning/tuned_models/energy_ft_faulty_tuning.pth'
    load_from_dict(network=resnet,
                   device=device,
                   path=weights_path)

    resnet_layers = [m for m in resnet.modules() if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear)]
    if layer_end == -1:
        resnet_layers = resnet_layers[layer_start:]
    else:
        resnet_layers = resnet_layers[layer_start:layer_end]
    resnet_layers_shape = [layer.weight.shape for layer in resnet_layers]

    print(f'Starting fault injection on {fine_tuning} resnet20')
    fault_injection(net=resnet,
                    net_name='resnet20',
                    fine_tuning=fine_tuning,
                    net_layer_shape=resnet_layers_shape,
                    loader=test_loader,
                    device=device,
                    force_n=force_n)


def fault_injection(net,
                    net_name,
                    fine_tuning,
                    net_layer_shape,
                    loader,
                    device,
                    seed=51195,
                    p=0.5,
                    e=0.01,
                    t=2.58,
                    force_n=None):

    cwd = os.getcwd()
    prefix_folder = f'{cwd}/fault_injection/{net_name}/' if fine_tuning == 'clean' else f'{cwd}/fault_injection/{net_name}/tuning/{fine_tuning}'

    os.makedirs(prefix_folder, exist_ok=True)
    filename_post = f'{prefix_folder}/{seed}_post_softmax.csv'
    filename_pre = f'{prefix_folder}/{seed}_pre_softmax.csv'

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
    n = force_n if force_n is not None else compute_date_n(N=len(exhaustive_fault_list),
                                                           p=p,
                                                           e=e,
                                                           t=t)

    fault_list = random_generator.choice(exhaustive_fault_list, int(n))
    del exhaustive_fault_list

    fault_list_filename = f'{cwd}/fault_injection/fault_list/{net_name}/' if fine_tuning == 'clean' else f'{cwd}/fault_injection/fault_list/{net_name}/tuning/{fine_tuning}'
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

                if fine_tuning == 'odin':
                    odin_inference(corrupt_net=corrupt_net,
                                   loader=loader,
                                   device=device,
                                   injection_index=injection_index,
                                   layer_index=layer_index,
                                   bit=bit,
                                   writer_inj_post=writer_inj_post,
                                   f_inj_post=f_inj_post)
                else:
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

    parser = argparse.ArgumentParser(description='Run a fault injection campaign',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fine_tuning', type=str, choices=['clean', 'energy', 'energy_faulty', 'odin'],
                        help='clean: no fine-tuning, energy: energy-based fine tuning')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for the inference')
    parser.add_argument('--force-n', type=int, default=None,
                        help='Force a number of fault injections')
    args = parser.parse_args()

    main(fine_tuning=args.fine_tuning, test_batch_size=args.batch_size, force_n =args.force_n)
