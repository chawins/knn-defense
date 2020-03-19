'''Train MNIST model'''
from __future__ import print_function

import logging
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from lib.dataset_utils import *
from lib.lip_model import *
from lib.mnist_model import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def evaluate(net, dataloader, criterion, device):

    net.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    return val_loss / val_total, val_correct / val_total


def train(net, trainloader, validloader, criterion, optimizer, epoch, device,
          log, save_best_only=True, best_acc=0, model_path='./model.pt'):

    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

    val_loss, val_acc = evaluate(net, validloader, criterion, device)

    log.info(' %5d | %.4f, %.4f | %8.4f, %7.4f', epoch,
             train_loss / train_total, train_correct / train_total,
             val_loss, val_acc)

    # Save model weights
    if not save_best_only or (save_best_only and val_acc > best_acc):
        log.info('Saving model...')
        torch.save(net.state_dict(), model_path)
        best_acc = val_acc
    return best_acc


def main():

    # Set experiment id
    exp_id = 20
    # model_name = 'train_mnist_exp%d' % exp_id
    model_name = 'dist_mnist_ce_exp%d' % exp_id

    # Training parameters
    batch_size = 128
    epochs = 200
    data_augmentation = False
    learning_rate = 1e-3
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
    # net = BasicModel()
    init_it = 1
    train_it = False
    net = NeighborModel(num_classes=10, init_it=init_it, train_it=train_it)
    net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    log.info(' epoch | loss  , acc    | val_loss, val_acc')
    best_acc = 0
    for epoch in range(epochs):
        best_acc = train(net, trainloader, validloader, criterion, optimizer,
                         epoch, device, log, save_best_only=True,
                         best_acc=best_acc, model_path=model_path)

    test_loss, test_acc = evaluate(net, testloader, criterion, device)
    log.info('Test loss: %.4f, Test acc: %.4f', test_loss, test_acc)


if __name__ == '__main__':
    main()
