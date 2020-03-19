'''Fine-tune MNIST model'''
from __future__ import print_function

import logging
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from lib.adv_model import *
from lib.dataset_utils import *
from lib.lip_model import *
from lib.mnist_model import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def row_sum(w):
    return torch.max(w.abs().sum(1) - 0, torch.tensor(0.).cuda()).sum()


def infty_norm_reg(w):
    wwt = torch.matmul(w, w.transpose(0, 1))
    wtw = torch.matmul(w.transpose(0, 1), w)
    reg = torch.min(row_sum(wwt), row_sum(wtw))
    return reg


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def evaluate(net, dataloader, device):

    net.eval()
    val_loss = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = loss_function(net, outputs, targets)
            val_loss += loss.item()
            val_total += targets.size(0)

    return val_loss / val_total


def train(net, trainloader, validloader, optimizer, epoch, device,
          log, save_best_only=True, best_loss=1e9, model_path='./model.pt'):

    net.train()
    train_loss = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(net, outputs, targets)
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


def loss_function(net, output, label, alpha=1e-1, beta=1e-2):

    loss = torch.tensor(0.).cuda()
    for i in range(output.size(0)):
        mask_same = (label[i] == label).type(torch.float32)
        mask_self = torch.ones(output.size(0)).cuda()
        mask_self[i] = 0
        mask_diff = (label[i] != label).type(torch.float32)
        dist = ((output[i] - output) ** 2).sum(1)
        # upper bound distance to prevent overflow
        exp = torch.exp(- torch.min(dist * 1e-2, torch.tensor(50.).cuda()))
        # exp = torch.exp(- dist)
        # loss += torch.log(torch.sum(mask_diff * exp) /
        #                   torch.sum(mask_self * exp))
        loss -= torch.log(torch.sum(mask_same * mask_self * exp) /
                          torch.sum(mask_self * exp))
        # loss += (1e20 * mask_same + exp).min() / torch.sum(mask_self * exp)
        # loss -= torch.min(torch.min(1e20 * mask_same + dist),
        #                   torch.tensor(4.).cuda())
        # loss -= torch.min(mask_diff * dist, torch.tensor(4.).cuda()).sum()
        # pull same
        # loss += alpha * \
        #     torch.min(1e20 * (mask_diff + 1 - mask_self) + dist)
        # loss += alpha * torch.sum(mask_same * dist)

    # Lipschitz loss
    reg = torch.tensor(0.).cuda()
    reg += infty_norm_reg(net.conv1.weight.reshape(8 ** 2, 64))
    reg += infty_norm_reg(net.conv2.weight.reshape(64 * 6 ** 2, 128))
    reg += infty_norm_reg(net.conv3.weight.reshape(128 * 5 ** 2, 128))
    loss += beta * reg * output.size(0)

    return loss


def main():

    # Set experiment id
    exp_id = 2

    # orig_model = 'train_mnist_exp%d'
    # orig_model = 'dist_mnist_ce_exp%d'
    orig_model = 'adv_mnist_exp2'

    model_name = 'tune%d_%s' % (exp_id, orig_model)

    # Training parameters
    batch_size = 128
    epochs = 50
    data_augmentation = False
    learning_rate = 1e-4
    l1_reg = 0
    l2_reg = 1e-4

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
    log = logging.getLogger('train_mnist')
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
    log.info(('MNIST | exp_id: {}, seed: {}, init_learning_rate: {}, ' +
              'batch_size: {}, l2_reg: {}, l1_reg: {}, epochs: {}, ' +
              'data_augmentation: {}, subtract_pixel_mean: {}').format(
                  exp_id, seed, learning_rate, batch_size, l2_reg, l1_reg,
                  epochs, data_augmentation, subtract_pixel_mean))

    log.info('Preparing data...')
    trainloader, validloader, testloader = load_mnist(
        batch_size, data_dir='/data', val_size=0.1, shuffle=True, seed=seed)

    log.info('Building model...')

    net = BasicModel()

    # net = NeighborModel(num_classes=10, init_it=1, train_it=False)

    config = {'epsilon': 0.3,
              'num_steps': 40,
              'step_size': 0.01,
              'random_start': True,
              'loss_func': 'xent'}
    net = PGDModel(net, config)

    net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    net.load_state_dict(torch.load('saved_models/' + orig_model + '.h5'))

    net = net.basic_net

    # replace final layer with an identity
    net.fc = Identity()

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    log.info(' epoch | loss | val_loss')
    best_loss = 1e9
    for epoch in range(epochs):
        best_loss = train(net, trainloader, validloader, optimizer,
                          epoch, device, log, save_best_only=True,
                          best_loss=best_loss, model_path=model_path)

    test_loss = evaluate(net, testloader, device)
    log.info('Test loss: %.4f', test_loss)


if __name__ == '__main__':
    main()
