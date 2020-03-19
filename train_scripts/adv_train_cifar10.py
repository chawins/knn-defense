'''Train CIFAR-10 model with adversarial training'''
from __future__ import print_function

import logging
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from lib.adv_model import *
from lib.cifar_resnet import *
from lib.dataset_utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def evaluate(net, dataloader, criterion, device, adv=False):

    net.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs, targets, attack=adv)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    return val_loss / val_total, val_correct / val_total


def train(net, trainloader, validloader, criterion, optimizer, epoch, device,
          log, save_best_only=True, best_loss=1e9, model_path='./model.pt'):

    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs, targets, attack=True)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

    adv_loss, adv_acc = evaluate(net, validloader, criterion, device, adv=True)
    val_loss, val_acc = evaluate(
        net, validloader, criterion, device, adv=False)

    log.info(' %5d | %.4f, %.4f | %.4f, %.4f | %.4f, %.4f | ', epoch,
             train_loss / train_total, train_correct / train_total,
             adv_loss, adv_acc, val_loss, val_acc)

    # Save model weights
    if not save_best_only or (save_best_only and adv_loss < best_loss):
        log.info('Saving model...')
        torch.save(net.state_dict(), model_path)
        best_loss = adv_loss
    return best_loss


def main():

    # Set experiment id
    exp_id = 6
    model_name = 'adv_cifar10_exp%d' % exp_id

    # Training parameters
    batch_size = 128
    epochs = 200
    data_augmentation = True
    learning_rate = 1e-3
    l1_reg = 0
    l2_reg = 0

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
    log.info(('CIFAR10 | exp_id: {}, seed: {}, init_learning_rate: {}, ' +
              'batch_size: {}, l2_reg: {}, l1_reg: {}, epochs: {}, ' +
              'data_augmentation: {}, subtract_pixel_mean: {}').format(
                  exp_id, seed, learning_rate, batch_size, l2_reg, l1_reg,
                  epochs, data_augmentation, subtract_pixel_mean))

    log.info('Preparing data...')
    trainloader, validloader, testloader = load_cifar10(batch_size,
                                                        data_dir='/data',
                                                        val_size=0.1,
                                                        normalize=False,
                                                        augment=data_augmentation,
                                                        shuffle=True,
                                                        seed=seed)

    log.info('Building model...')
    net = PreActResNet(PreActBlock, [2, 2, 2, 2])
    # net = AlexNet()

    # config = {'epsilon': 0.3,
    #           'num_steps': 40,
    #           'step_size': 0.01,
    #           'random_start': True,
    #           'loss_func': 'xent'}
    # net = PGDModel(basic_net, config)
    config = {'epsilon': 1,
              'num_steps': 10,
              'step_size': 0.2,
              'random_start': True,
              'loss_func': 'xent'}
    net = PGDL2Model(net, config)

    net = net.to(device)
    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [80, 120, 160], gamma=0.1)

    log.info(' epoch | loss  , acc    | adv_l , adv_a  | val_l , val_a |')
    best_loss = 1e9
    for epoch in range(epochs):
        best_loss = train(net, trainloader, validloader, criterion, optimizer,
                          epoch, device, log, save_best_only=True,
                          best_loss=best_loss, model_path=model_path)
        lr_scheduler.step()

    test_loss, test_acc = evaluate(
        net, testloader, criterion, device, adv=False)
    log.info('Test loss: %.4f, Test acc: %.4f', test_loss, test_acc)
    test_loss, test_acc = evaluate(
        net, testloader, criterion, device, adv=True)
    log.info('Test adv loss: %.4f, Test adv acc: %.4f', test_loss, test_acc)


if __name__ == '__main__':
    main()
