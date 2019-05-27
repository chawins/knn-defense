'''Train MNIST VAE model'''
from __future__ import print_function

import logging
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from lib.cifar10_model import *
from lib.cifar_resnet import *
from lib.dataset_utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def evaluate(net, dataloader, device):

    net.eval()
    val_loss = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            en_mu, en_logvar, out = net(inputs)
            loss = loss_function(inputs, en_mu, en_logvar, out)
            val_loss += loss.item()
            val_total += targets.size(0)

    return val_loss / val_total


def train(net, trainloader, validloader, optimizer, epoch, device,
          log, save_best_only=True, best_loss=np.inf, model_path='./model.pt'):

    net.train()
    train_loss = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        en_mu, en_logvar, out = net(inputs)
        loss = loss_function(inputs, en_mu, en_logvar, out)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_total += targets.size(0)

    val_loss = evaluate(net, validloader, device)

    log.info(' %5d | %.4f | %.4f', epoch, train_loss / train_total, val_loss)

    # Save model weights
    # if not save_best_only or (save_best_only and val_acc > best_acc):
    if not save_best_only or (save_best_only and val_loss < best_loss):
        log.info('Saving model...')
        torch.save(net.state_dict(), model_path)
        best_loss = val_loss
    return best_loss


def loss_function(x, en_mu, en_logvar, out):

    # constants that balance loss terms
    beta = 1

    # elbo
    # from torch.distributions.normal import Normal
    # de_mu, de_logvar = out
    # normal = Normal(de_mu, torch.exp(0.5 * de_logvar))
    # logprob = - torch.sum(normal.log_prob(x))
    # part1 = torch.sum(de_logvar)
    # sigma = de_logvar.mul(0.5).exp_()
    # part2 = torch.sum(((x - de_mu) / sigma) ** 2)
    # logprob = 0.5 * (part1 + part2)
    # logprob = - torch.sum(x * torch.log(out) + (1 - x) * torch.log(1 - out))
    logprob = F.binary_cross_entropy(out, x, reduction='sum')
    # logprob = torch.sum((out - x) ** 2)
    kld = -0.5 * torch.sum(1 + en_logvar - en_mu.pow(2) - en_logvar.exp())

    # auxilary loss
    # aux_loss = nn.CrossEntropyLoss()(y_pred, y_true)

    # return (logprob + beta * kld) / (np.log(2) * 3072)
    return logprob + beta * kld


def main():

    # Set experiment id
    exp_id = 5
    model_name = 'train_cifar10_vae_exp%d' % exp_id

    # Training parameters
    batch_size = 256
    epochs = 100
    data_augmentation = False
    learning_rate = 1e-3
    l1_reg = 0
    l2_reg = 1e-4
    latent_dim = 128

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
    log = logging.getLogger('cifar10_vae')
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
                                                        augment=data_augmentation,
                                                        shuffle=True,
                                                        seed=seed)

    log.info('Building model...')
    # net = PreActResNet_VAE(PreActBlock, [2, 2, 2, 2], latent_dim=latent_dim)
    net = CIFAR10_VAE((3, 32, 32), latent_dim=latent_dim)
    net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, alpha=0.9)

    log.info(' epoch | loss')
    best_loss = np.inf
    for epoch in range(epochs):
        best_loss = train(net, trainloader, validloader, optimizer,
                          epoch, device, log, save_best_only=True,
                          best_loss=best_loss, model_path=model_path)
        with torch.no_grad():
            z = torch.randn(100, net.latent_dim).to(device)
            out = net.decode(z)
            torchvision.utils.save_image(
                out, 'vis/cifar10_vae_epoch%d.png' % epoch, 10)

    test_loss = evaluate(net, testloader, device)
    log.info('Test loss: %.4f', test_loss)


if __name__ == '__main__':
    main()
