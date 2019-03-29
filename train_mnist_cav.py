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
from lib.mnist_model import *


def evaluate(net, dataloader, device):

    net.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            en_mu, en_logvar, de_mu, de_logvar, y = net(inputs)
            loss = loss_function(
                inputs, targets, y, en_mu, en_logvar, de_mu, de_logvar)
            val_loss += loss.item()
            _, predicted = y.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    return val_loss / val_total, val_correct / val_total


def train(net, trainloader, validloader, optimizer, epoch, device,
          log, save_best_only=True, best_acc=0, model_path='./model.pt'):

    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        en_mu, en_logvar, de_mu, de_logvar, y = net(inputs)
        loss = loss_function(
            inputs, targets, y, en_mu, en_logvar, de_mu, de_logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = y.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

    val_loss, val_acc = evaluate(net, validloader, device)

    log.info(' %5d | %.4f, %.4f | %8.4f, %7.4f', epoch,
             train_loss / train_total, train_correct / train_total,
             val_loss, val_acc)

    # Save model weights
    if not save_best_only or (save_best_only and val_acc > best_acc):
        log.info('Saving model...')
        torch.save(net.state_dict(), model_path)
        best_acc = val_acc
    return best_acc


def loss_function(x, y_true, y_pred, en_mu, en_logvar, de_mu, de_logvar):

    # constants that balance loss terms
    beta = 1
    gamma = 1

    # elbo
    # normal = Normal(de_mu, torch.exp(0.5 * de_logvar))
    # logprob = - torch.sum(normal.log_prob(x)) / (np.log(2) * 784)
    logprob = - torch.sum(x * torch.log(de_mu) + (1 - x)
                          * torch.log(1 - de_mu)) / (np.log(2) * 784)
    kld = -0.5 * torch.sum(
        1 + en_logvar - en_mu.pow(2) - en_logvar.exp()) / (np.log(2) * 784)

    # auxilary loss
    aux_loss = nn.CrossEntropyLoss()(y_pred, y_true)

    return logprob + beta * kld + gamma * aux_loss


def main():

    # Set experiment id
    exp_id = 2

    # Training parameters
    batch_size = 128
    epochs = 50
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
    model_name = 'train_mnist_cav_exp%d.h5' % exp_id
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    # Get logger
    log_file = 'train_mnist_cav_exp{}.log'.format(exp_id)
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
    net = ClassAuxVAE((1, 28, 28), num_classes=10, latent_dim=20)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    log.info(' epoch | loss  , acc    | val_loss, val_acc')
    best_acc = 0
    for epoch in range(epochs):
        best_acc = train(net, trainloader, validloader, optimizer,
                         epoch, device, log, save_best_only=True,
                         best_acc=best_acc, model_path=model_path)
        with torch.no_grad():
            z = torch.randn(100, net.module.latent_dim).to(device)
            de_mu, _ = net.module.decode(z)
            torchvision.utils.save_image(de_mu, 'epoch%d.png' % epoch, 10)

    test_loss, test_acc = evaluate(net, testloader, device)
    log.info('Test loss: %.4f, Test acc: %.4f', test_loss, test_acc)


if __name__ == '__main__':
    main()
