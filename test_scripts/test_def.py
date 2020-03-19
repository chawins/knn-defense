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
from lib.cifar10_model import *
from lib.cifar_resnet import *
from lib.cwl2_attack import CWL2Attack
from lib.dataset_utils import *
from lib.dknn import DKNN, DKNNL2
from lib.dknn_attack import *
from lib.dknn_attack_l2 import *
from lib.knn import *
from lib.lip_model import *

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

exp_id = 0

model_name = 'adv_cifar10_exp%d.h5' % exp_id
# model_name = 'train_cifar10_vae_exp%d.h5' % exp_id
# model_name = 'rot_cifar10_exp%d.h5' % exp_id
# model_name = 'ae_cifar10_exp%d.h5' % exp_id
# model_name = 'adv_cifar10_exp4.h5'

# net = PreActResNet(PreActBlock, [2, 2, 2, 2]).eval()
# config = {'num_steps': 8,
#           'step_size': 0.05,
#           'random_start': True,
#           'loss_func': 'xent'}
# net = PGDL2Model(net, config)

# net = PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=4)
# net.load_state_dict(torch.load('saved_models/' + model_name))
# net = net.eval().to('cuda')

# layers = ['relu1', 'relu2', 'relu3', 'fc']
# layers = ['gs1', 'gs2', 'gs3', 'fc']
# layers = ['relu1', 'relu2', 'relu3', 'fc1', 'fc2']
# layers = ['conv1', 'relu1', 'conv2', 'relu2', 'relu3', 'fc1', 'fc2']
layers = ['layer4']

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

net = PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=10)
config = {'num_steps': 20,
          'step_size': 0.05,
          'random_start': True,
          'loss_func': 'xent'}
net = PGDL2Model(net, config)
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

net.load_state_dict(torch.load(model_path))
# net = net.module
net = net.basic_net
net.eval()
# print(net)

(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_cifar10_all(
    './data', val_size=0.1, seed=seed)


num = 100
dknn = DKNNL2(net, x_train, y_train, x_valid, y_valid, layers,
              k=75, num_classes=10)

with torch.no_grad():
    y_pred = dknn.classify(x_test)
    ind = np.where(y_pred.argmax(1) == y_test.numpy())[0]
    print((y_pred.argmax(1) == y_test.numpy()).sum() / y_test.size(0))

# attack = SoftDKNNAttack()
attack = DKNNL2Attack()


def attack_batch(x, y, batch_size, layer):
    x_a = torch.zeros_like(x)
    total_num = x.size(0)
    num_batches = total_num // batch_size
    for i in range(num_batches):
        print(i)
        begin = i * batch_size
        end = (i + 1) * batch_size
        x_a[begin:end] = attack(
            dknn, x[begin:end], y[begin:end],
            guide_layer=layer, m=300, binary_search_steps=10,
            max_iterations=500, learning_rate=1e-2, initial_const=1e-5,
            abort_early=False, random_start=True, guide=2)
    return x_a


for layer in layers:

    x_adv = attack_batch(x_test[ind][:num].cuda(),
                         y_test[ind][:num], 100, layer)

    # pickle.dump(x_adv.cpu().detach(), open(
    #     'x_adv105_' + model_name + '.p', 'wb'))

    y_pred = dknn.classify(x_adv)
    acc = (y_pred.argmax(1) == y_test[ind][:num].numpy()).sum() / len(y_pred)

    # y_clean = dknn.classify(x_test[:num])
    # ind = (y_clean.argmax(1) == y_test[:num].numpy()) & (
    #     y_pred.argmax(1) != y_test[:num].numpy())
    dist = np.mean(np.sqrt(np.sum((x_adv.cpu().detach().numpy() -
                                   x_test.numpy()[ind][:num])**2, (1, 2, 3))))

    print('(' + layer + ') dknn attack %.4f (%.4f)' % (dist, acc))
    ind_adv = np.where(y_pred.argmax(1) != y_test[ind][:num].numpy())[0]
    dist = np.sqrt(np.sum((x_adv.cpu().detach().numpy() -
                           x_test.numpy()[ind][:num])[ind_adv]**2, (1, 2, 3)))
    print('Mean distance: %.4f' % np.mean(dist))
