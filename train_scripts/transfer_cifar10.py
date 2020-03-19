'''Transfer CIFAR-10 model'''
from __future__ import print_function

import logging
import os
import pdb

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from lib.cifar_resnet import *
from lib.dataset_utils import *
from lib.nin import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def evaluate(net, dataloader, device):

    # net.eval()
    net.fc.training = False
    val_loss = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = loss_function(outputs, targets)
            val_loss += loss.item()
            val_total += targets.size(0)

    return val_loss / val_total


def train(net, trainloader, validloader, optimizer, epoch, device,
          log, save_best_only=True, best_loss=1e9, model_path='./model.pt'):

    # net.train()
    net.fc.training = True
    train_loss = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_total += targets.size(0)

    val_loss = evaluate(net, validloader, device)

    log.info(' %5d | %.4f | %.4f', epoch, train_loss / train_total, val_loss)

    # Save model weights
    if not save_best_only or (save_best_only and val_loss < best_loss):
        log.info('Saving model...')
        torch.save(net.state_dict(), model_path)
        best_loss = val_loss
    return best_loss


def loss_function(outputs, targets):

    batch_size = outputs.size(0)
    loss = torch.zeros(1).cuda()
    for i in range(batch_size):
        mask_same = (targets[i] == targets).type(torch.float32)
        # if mask_same.sum() == 1:
        #     break
        mask_diff = (targets[i] != targets).type(torch.float32)
        mask_self = torch.ones(batch_size).cuda()
        mask_self[i] = 0
        dist = ((outputs[i] - outputs) ** 2).sum(1)
        # upper bound distance to prevent overflow
        # exp = torch.exp(- torch.min(dist, torch.tensor(50.).cuda()))
        # exp = torch.exp(- dist)
        # loss -= torch.log(torch.sum(mask_same * mask_self * exp) /
        #                   torch.sum(mask_self * exp))
        if mask_diff.sum() > 0:
            loss -= torch.min(torch.min(1e20 * mask_same + dist),
                              torch.tensor(100.).cuda())
        # additional regularization to pull same class
        const = 1e0
        # exp = torch.exp(- torch.min(dist * self.it.exp(),
        #                             torch.tensor(50.).cuda()))
        # loss -= const * torch.log(torch.sum(mask_same * mask_self * exp) /
        #                           torch.sum(mask_self * exp))
        # TODO
        # loss += const * \
        #     torch.log(torch.sum(mask_diff * exp) / torch.sum(mask_self * exp))
        if mask_same.sum() > 1:
            loss += const * \
                torch.min(1e20 * (mask_diff + 1 - mask_self) + dist)

    # loss = F.cross_entropy(outputs, targets, reduction='sum')

    # class mean clustering
    # batch_size = outputs.size(0)
    # loss = torch.zeros(1).cuda()
    # mean = torch.zeros((10, outputs.size(1))).cuda()
    # for i in range(10):
    #     mask = targets == i
    #     mean[i] = outputs[mask].mean(0).detach()
    # for i in range(batch_size):
    #     dist = torch.sum((outputs[i] - mean) ** 2, 1)
    #     exp = torch.exp(- torch.min(dist, torch.tensor(50.).cuda()))
    #     loss -= torch.log(exp[targets[i]] / exp.sum())

    return loss


def main():

    # Set experiment id
    exp_id = 18
    model_name = 'transfer_cifar10_exp%d' % exp_id

    # Training parameters
    batch_size = 32
    epochs = 120
    data_augmentation = False
    learning_rate = 1e-3
    l1_reg = 0
    l2_reg = 1e-2
    block = 3

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = False

    # Set all random seeds
    seed = 2019
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up model directory
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name + '.h5')

    # Get logger
    log_file = model_name + '.log'
    log = logging.getLogger('train_cifar10')
    log.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s %(name)s] %(message)s')
    # Create file handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    log.info(log_file)
    log.info(('CIFAR-10 | exp_id: {}, seed: {}, init_learning_rate: {}, ' +
              'batch_size: {}, l2_reg: {}, l1_reg: {}, epochs: {}, ' +
              'data_augmentation: {}, subtract_pixel_mean: {}').format(
                  exp_id, seed, learning_rate, batch_size, l2_reg, l1_reg,
                  epochs, data_augmentation, subtract_pixel_mean))

    log.info('Preparing data...')
    trainloader, validloader, testloader = load_cifar10(batch_size,
                                                        data_dir='/data',
                                                        val_size=0.1,
                                                        normalize=False,
                                                        augment=False,
                                                        shuffle=True,
                                                        seed=seed)

    log.info('Building model...')
    # net = PreActResNet(PreActBlock, [2, 2, 2, 2])
    # net = net.to(device)

    # opt = {'num_classes': 4, 'num_stages': 4}
    # net = NetworkInNetwork(opt)
    # net.load_state_dict(torch.load(
    #     'saved_models/model_net_epoch200')['network'])
    # net = net._feature_blocks
    # net_wrap = NINWrapper(net, block=block)
    # for param in net_wrap.parameters():
    #     param.requires_grad = False
    # # net_wrap.fc = nn.Linear(3072, 128)
    # if block == 2:
    #     net_wrap.fc = nn.Sequential(
    #         nn.BatchNorm1d(12288),
    #         nn.Linear(12288, 2000),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm1d(2000),
    #         nn.Linear(2000, 400),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm1d(400),
    #         nn.Linear(400, 128),
    #     )
    # elif block == 3:
    #     net_wrap.fc = nn.Sequential(
    #         nn.Linear(3072, 200),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(200, 200),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(200, 128),
    #     )
    # net_wrap = net_wrap.to('cuda')

    net = PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=4)
    net.load_state_dict(torch.load('saved_models/rot_cifar10_exp0.h5'))
    net_wrap = ResNetWrapper(net, block=block, dim=16384)
    for param in net_wrap.parameters():
        param.requires_grad = False
    # net_wrap.fc = nn.Linear(3072, 128)
    net_wrap.eval()
    if block == 4:
        net_wrap.fc = nn.Sequential(
            nn.Linear(8192, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 128),
        )
    elif block == 3:
        net_wrap.fc = nn.Sequential(
            nn.BatchNorm1d(16384),
            nn.Linear(16384, 2000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2000),
            nn.Linear(2000, 400),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(400),
            nn.Linear(400, 128),
        )
    net_wrap = net_wrap.to('cuda')

    # mean = pickle.load(open('resnet_block3_mean.p', 'rb'))
    # std = pickle.load(open('resnet_block3_std.p', 'rb'))
    # net_wrap.mean.data = torch.tensor(mean).cuda()
    # net_wrap.std.data = torch.tensor(std).cuda()

    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    optimizer = optim.Adam(
        net_wrap.parameters(), lr=learning_rate, weight_decay=l2_reg)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [50, 70, 90], gamma=0.1)

    log.info(' epoch | loss | val_loss')
    best_loss = 1e9
    for epoch in range(epochs):
        lr_scheduler.step()
        best_loss = train(net_wrap, trainloader, validloader, optimizer,
                          epoch, device, log, save_best_only=True,
                          best_loss=best_loss, model_path=model_path)

    test_loss = evaluate(net_wrap, testloader, device)
    log.info('Test loss: %.4f', test_loss)


if __name__ == '__main__':
    main()
