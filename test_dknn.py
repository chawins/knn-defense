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
from lib.cwl2_attack import CWL2Attack
from lib.dataset_utils import *
from lib.dknn import DKNN
from lib.dknn_attack import DKNNAttack, SoftDKNNAttack
from lib.mnist_model import *

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

exp_id = 0

model_name = 'train_mnist_exp%d.h5' % exp_id
net = BasicModel()

# model_name = 'train_mnist_snnl_exp%d.h5' % exp_id
# net = SNNLModel(train_it=True)

# model_name = 'train_mnist_hidden_mixup_exp%d.h5' % exp_id
# net = HiddenMixupModel()

# model_name = 'train_mnist_vae_exp%d.h5' % exp_id
# net = VAE((1, 28, 28), num_classes=10, latent_dim=20)
# net = VAE2((1, 28, 28), num_classes=10, latent_dim=20)

# model_name = 'train_mnist_cav_exp%d.h5' % exp_id
# net = ClassAuxVAE((1, 28, 28), num_classes=10, latent_dim=20)

# Set all random seeds
seed = 2019
np.random.seed(seed)
torch.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set up model directory
save_dir = os.path.join(os.getcwd(), 'saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net.load_state_dict(torch.load(model_path))
net = net.module
net.eval()

(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist_all(
    '/data', val_size=0.1, seed=seed)

# layers = ['relu1', 'relu2', 'relu3', 'fc']
# layers = ['relu1', 'relu2', 'relu3']
layers = ['relu3']
# layers = ['en_conv3']
# layers = ['en_mu']
# net = net.cpu()
with torch.no_grad():
    dknn = DKNN(net, x_train.cuda(), y_train, x_valid.cuda(), y_valid, layers,
                k=75, num_classes=10)
    y_pred = dknn.classify(x_test)


attack = SoftDKNNAttack()


def attack_batch(x, y, batch_size):
    x_a = torch.zeros_like(x)
    total_num = x.size(0)
    num_batches = total_num // batch_size
    for i in range(num_batches):
        begin = i * batch_size
        end = (i + 1) * batch_size
        x_a[begin:end] = attack(
            dknn, x[begin:end], y[begin:end],
            layer=layers[0], m=100, binary_search_steps=5,
            max_iterations=500, learning_rate=1e-1,
            initial_const=1, abort_early=True)
    return x_a


x_adv = attack_batch(x_test[:1000].cuda(), y_test[:1000].cuda(), 100)
