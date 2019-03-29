'''Test DkNN model on MNIST'''
from __future__ import print_function

import logging
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from lib.cwl2_attack import CWL2Attack
from lib.dataset_utils import *
from lib.dknn_attack import DKNNAttack
from lib.mnist_model import *
from lib.pgd_attack import PGDAttack


def main():

    # Set experiment id
    exp_id = 0

    # Training parameters
    batch_size = 128
    epochs = 15
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
    model_name = 'train_mnist_exp%d.h5' % exp_id
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    net = BasicModel()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(torch.load(model_path))
    # use when passing to dknn
    net = net.module
    net.eval()

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist_all(
        '/data', val_size=0.1, seed=seed)
    x_train, x_valid = x_train.to(device), x_valid.to(device)
    # net = net.to('cpu')

    # attack = CWL2Attack()
    # attack = PGDAttack()
    attack = DKNNAttack()

    # layers = ['relu1', 'relu2', 'relu3', 'fc']
    layers = ['relu3']
    dknn = DKNN(net, x_train, y_train, x_valid, y_valid, layers,
                k=75, num_classes=10)

    start = time.time()
    # x_adv = attack(net, x_test[:100].cuda(), y_test[:100].cuda(),
    #                targeted=False, binary_search_steps=10, max_iterations=1000,
    #                confidence=0, learning_rate=1e-1, initial_const=1,
    #                abort_early=True)
    # x_adv = attack(net, x_test[:100].cuda(), y_test[:100].cuda(),
    #                targeted=False, epsilon=0.1, max_epsilon=0.3,
    #                max_iterations=20, random_restart=3)
    # x_adv = attack(dknn, x_test[:100].cuda(), y_test[:100].cuda(),
    #                guide_layer='relu1', binary_search_steps=5,
    #                max_iterations=500, learning_rate=1e-1,
    #                initial_const=1, abort_early=True)
    end = time.time()
    print(end - start)

    # layers = ['relu1', 'relu2', 'relu3', 'fc']
    #
    with torch.no_grad():
        #     start = time.time()
        #     dknn = DKNN(net, x_train, y_train, x_valid, y_valid, layers,
        #                 k=75, num_classes=10)
        #     # out = dknn.get_neighbors(x_train[:10], 1)
        #     end = time.time()
        #     print(end - start)
        #
        #     start = time.time()
        #     # out = dknn.classify(x_test.to(device), 75)
        y_pred = dknn.classify(x_test.cuda()).argmax(1)
        print(np.mean(y_pred == y_test.numpy()))
    #     end = time.time()
    #     print(end - start)


if __name__ == '__main__':
    main()
