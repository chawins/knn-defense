import logging
import os
import pdb
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import foolbox
from lib.adv_model import *
from lib.blackbox_attack import attack_untargeted
from lib.cwl2_attack import CWL2Attack
from lib.dataset_utils import *
from lib.dknn import DKNN, DKNNL2
from lib.dknn_attack import *
from lib.dknn_attack_l2 import *
from lib.knn import *
from lib.lip_model import *
from lib.mnist_model import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

exp_id = 2

# model_name = 'train_mnist_exp%d.h5' % exp_id
# net = BasicModel()

# model_name = 'train_mnist_snnl_exp%d.h5' % exp_id
# net = SNNLModel(train_it=True)

# model_name = 'train_mnist_hidden_mixup_exp%d.h5' % exp_id
# net = HiddenMixupModel()

# model_name = 'train_mnist_vae_exp%d.h5' % exp_id
# net = VAE((1, 28, 28), num_classes=10, latent_dim=20)
# net = VAE2((1, 28, 28), num_classes=10, latent_dim=20)

# model_name = 'train_mnist_cav_exp%d.h5' % exp_id
# net = ClassAuxVAE((1, 28, 28), num_classes=10, latent_dim=20)

# model_name = 'dist_mnist_exp%d.h5' % exp_id
# init_it = 1
# train_it = False
# net = NeighborModel(num_classes=10, init_it=init_it, train_it=train_it)

# model_name = 'lipae_mnist_exp%d.h5' % exp_id
# init_it = 1
# train_it = False
# latent_dim = 128
# alpha = 1e2
# net = NCA_AE(latent_dim=latent_dim, init_it=init_it,
#              train_it=train_it, alpha=alpha)

model_name = 'adv_mnist_exp%d.h5' % exp_id
basic_net = BasicModel()
# basic_net = BasicModelV2()
config = {'epsilon': 0.3,
          'num_steps': 40,
          'step_size': 0.01,
          'random_start': True,
          'loss_func': 'xent'}
net = PGDL2Model(basic_net, config)

# layers = ['relu1', 'relu2', 'relu3', 'fc']
# layers = ['gs1', 'gs2', 'gs3', 'fc']
# layers = ['relu1', 'relu2', 'relu3', 'fc1', 'fc2']
# layers = ['conv1', 'relu1', 'conv2', 'relu2', 'relu3', 'fc1', 'fc2']
layers = ['relu3']

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
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
net.load_state_dict(torch.load(model_path))
# net = net.module
net = net.basic_net
net.eval()

(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist_all(
    '/data', val_size=0.1, seed=seed)


num = 10
dknn = DKNNL2(net, x_train, y_train, x_valid, y_valid, layers,
              k=75, num_classes=10)

x_adv = x_test[:num].clone()

for i in range(num):
    x_adv[i] = attack_untargeted(
        dknn, list(zip(x_train, y_train)), x_test[i], y_test[i], alpha=2, beta=0.005,
        iterations=1000)

y_pred = dknn.classify(x_adv)
acc = (y_pred.argmax(1) == y_test[:num].numpy()).sum() / len(y_pred)

y_clean = dknn.classify(x_test[:num])
ind = (y_clean.argmax(1) == y_test[:num].numpy()) & (
    y_pred.argmax(1) != y_test[:num].numpy())
dist = np.sqrt(np.sum((x_adv.cpu().detach().numpy()[ind] -
                       x_test.numpy()[:num][ind])**2, (1, 2, 3)))

print(dist)
print('acc %.4f, dknn attack %.4f' % (acc, np.mean(dist)))
