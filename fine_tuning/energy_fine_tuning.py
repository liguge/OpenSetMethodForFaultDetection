'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from models.resnet import resnet20
# from models.utils import load_from_dict
# from utils import progress_bar
import torchvision.transforms as trn

from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    torch.multiprocessing.freeze_support()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./weights', train=True, download=True, transform=transform_train)

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    ood_data = torchvision.datasets.SVHN(
        root=f'weights',
        transform=trn.Compose(
            [trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]),
        split='train',
        download=True
    )

    train_loader_in = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0)

    train_loader_out = torch.utils.data.DataLoader(
        ood_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)


    testset = torchvision.datasets.CIFAR10(
        root='./weights', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = resnet20()
    # load_from_dict(net, device, '../models/pretrained_models/resnet20-trained.th')
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)

    best_acc = 0
    for epoch in range(0, 100):
        train(epoch, net, train_loader_in, train_loader_out, criterion, optimizer, batches=int(len(trainset)/args.batch_size))
        best_acc = test(epoch, net, testloader, device, criterion, best_acc, batches=int(len(testset)/args.batch_size))


# Training
def train(epoch, net, train_loader_in, train_loader_out, criterion, optimizer, batches, m_in=-25, m_out=-7):
    print('\nEpoch: %d' % epoch)
    net.train()

    total = 0
    correct = 0
    train_loss = 0

    pbar = tqdm(zip(train_loader_in, train_loader_out), total=batches, desc='Training')
    for in_set, out_set in pbar:
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        # forward
        x = net(data)

        # backward

        loss = criterion(x[:len(in_set[0])], target)
        # cross-entropy from softmax distribution to uniform distribution
        Ec_out = -torch.logsumexp(x[len(in_set[0]):], dim=1)
        Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=1)
        loss += 0.1*(torch.pow(F.relu(Ec_in-m_in), 2).mean() + torch.pow(F.relu(m_out-Ec_out), 2).mean())

        loss.backward()
        optimizer.step()

        _, predicted = x[:len(in_set[0])].max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        pbar.set_postfix({'Loss': loss.item(),
                          'Accuracy': f'{100 * correct/total:.1f}'})

        train_loss += loss.item()


def test(epoch, net, testloader, device, criterion, best_acc, batches):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(testloader, total=batches, desc='Testing')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'Accuracy': f'{100 * correct / total:.1f}'})
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('tuned_models'):
            os.mkdir('tuned_models')
        torch.save(state, './tuned_models/energy.pth')
        best_acc = acc
    return best_acc


if __name__ == '__main__':
    main()
