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


def main():

    # Set experiment id
    exp_id = 2

    # Training parameters
    batch_size = 128
    epochs = 100
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

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist_all(
        '/data', val_size=0.1, seed=seed)

    net = ClassAuxVAE((1, 28, 28), num_classes=10, latent_dim=20)
    # net = BasicModel()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(torch.load(model_path))
    net = net.module
    net.eval()

    # test accuracy
    # y_pred = net(x_test.to(device))[4]
    en_mu, _ = net.encode(x_test.to(device))
    y_pred = net.auxilary(en_mu)
    print((y_pred.argmax(1) == y_test.to(device)).sum())

    from lib.pgd_attack import PGDAttack
    attack = PGDAttack()
    x_adv = attack(net, x_test.cuda(), y_test.cuda(),
                   targeted=False, epsilon=0.05, max_epsilon=0.2,
                   max_iterations=20, random_restart=3)
    # y_pred = net(x_adv)[4]
    # en_mu, _ = net.encode(x_adv)
    en_mu, en_logvar = net.encode(x_adv)
    y_pred = torch.zeros((x_test.size(0), ))
    for i in range(x_test.size(0)):
        normal = Normal(en_mu[i], (0.5 * en_logvar[i]).exp())
        z = normal.sample((100, ))
        y = net.auxilary(z)
        import pdb
        pdb.set_trace()
    print((y_pred.argmax(1) == y_test.to(device)).sum())

    # test samples
    z = torch.randn(100, net.latent_dim).to(device)
    de_mu, _ = net.decode(z)
    # torchvision.utils.save_image(
    #     de_mu, 'test.png', 10, normalize=True, range=(0, 1))
    torchvision.utils.save_image(de_mu, 'test.png', 10)


if __name__ == '__main__':
    main()
