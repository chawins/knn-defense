import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import foolbox
from lib.adv_model import *
from lib.cwl2_attack import CWL2Attack
from lib.dataset_utils import *
from lib.dknn import DKNNL2
from lib.dknn_attack import DKNNAttack
from lib.dknn_attack_exp import DKNNExpAttack
from lib.dknn_attack_l2 import DKNNL2Attack
from lib.dknn_attack_linf import DKNNLinfAttack
from lib.lip_model import *
from lib.mnist_model import *
from lib.utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set all random seeds
seed = 2019
np.random.seed(seed)
torch.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist_all(
    '/data', val_size=0.1, seed=seed)


for epoch in range(100):

    print('epoch %d' % epoch)
    model_name = 'adv_mnist_nca_exp21_epoch%d.h5' % epoch
    net = NCAModel(output_dim=100, init_it=1)

    # Set up model directory
    save_dir = os.path.join(os.getcwd(), 'saved_models/')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    net = net.to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    dknn = DKNNL2(net, x_train, y_train, x_valid, y_valid, ['fc'],
                  k=5, num_classes=10)

    with torch.no_grad():
        y_pred = dknn.classify(x_test)
        ind = np.where(y_pred.argmax(1) == y_test.numpy())[0]
        print('  accuracy: %.4f' %
              ((y_pred.argmax(1) == y_test.numpy()).sum() / y_test.size(0)))

    attack = DKNNExpAttack(dknn)

    def attack_batch(x, y, batch_size):
        x_adv = torch.zeros_like(x)
        total_num = x.size(0)
        num_batches = total_num // batch_size
        for i in range(num_batches):
            begin = i * batch_size
            end = (i + 1) * batch_size
            x_adv[begin:end] = attack(
                x[begin:end], y[begin:end],
                guide_layer='fc', m=40, binary_search_steps=10,
                max_iterations=1000, learning_rate=1e-1,
                initial_const=1e1, random_start=False,
                thres_steps=200, check_adv_steps=200, verbose=False,
                max_linf=None)
        return x_adv

    num = 100
    x_adv = attack_batch(x_test[ind][:num].cuda(), y_test[ind][:num], 100)

    with torch.no_grad():
        y_pred = dknn.classify(x_adv)
        ind_adv = np.where(y_pred.argmax(1) != y_test[ind][:num].numpy())[0]
        print('  adv accuracy: %.4f' % (
            (y_pred.argmax(1) == y_test[ind][:num].numpy()).sum() / y_pred.shape[0]))
    dist = (x_adv.cpu() - x_test[ind][:num]).view(num, -1).norm(2, 1)[ind_adv]
    print('  mean dist: %.4f' % dist.mean().item())
