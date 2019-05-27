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
from lib.cwl2_attack import CWL2Attack
from lib.dataset_utils import *
from lib.dknn import DKNN, DKNNL2
from lib.dknn_attack import DKNNAttack, SoftDKNNAttack
from lib.knn import *
from lib.lip_model import *
from lib.mnist_model import *
from tune_mnist import Identity

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

exp_id = 0

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

orig_model = 'adv_mnist_exp2.h5'
model_name = 'tune%d_%s' % (exp_id, orig_model)
net = BasicModel()
net.fc = Identity()

layers = ['relu1', 'relu2', 'relu3']
# layers = ['relu1', 'relu2', 'relu3', 'fc']
# layers = ['gs1', 'gs2', 'gs3', 'fc']
# layers = ['relu1', 'relu2', 'relu3', 'fc1', 'fc2']
# layers = ['conv1', 'relu1', 'conv2', 'relu2', 'relu3', 'fc1', 'fc2']

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
net.eval()

(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist_all(
    '/data', val_size=0.1, seed=seed)

# layers = ['relu1', 'relu2', 'relu3', 'fc']
# layers = ['relu1', 'relu2', 'relu3']
# layers = ['relu3']
# layers = ['en_conv3']
# layers = ['en_mu']
# net = net.cpu()

# dknn = DKNNL2(net, x_train, y_train, x_valid, y_valid, layers,
#               k=1, num_classes=10)
# attack = SoftDKNNAttack()
#
# def attack_batch(x, y, batch_size):
#     x_a = torch.zeros_like(x)
#     total_num = x.size(0)
#     num_batches = total_num // batch_size
#     for i in range(num_batches):
#         print(i)
#         begin = i * batch_size
#         end = (i + 1) * batch_size
#         x_a[begin:end] = attack(
#             dknn, x[begin:end], y[begin:end],
#             layer=layers[0], m=100, binary_search_steps=5,
#             max_iterations=500, learning_rate=1e-1,
#             initial_const=1, abort_early=True)
#     return x_a
#
#
# x_adv = attack_batch(x_test[:500].cuda(), y_test[:500].cuda(), 100)
#
# y_pred = dknn.classify(x_adv)
# print((y_pred.argmax(1) == y_test[:500].numpy()).sum() / len(y_pred))
#
# y_clean = dknn.classify(x_test[:500])
# ind = (y_clean.argmax(1) == y_test[:500].numpy()) & (
#     y_pred.argmax(1) != y_test[:500].numpy())
# dist = np.mean(np.sqrt(np.sum((x_adv.cpu().detach().numpy()[ind] -
#                                x_test.numpy()[:500][ind])**2, (1, 2, 3))))
# print(dist)

# layer = 'relu1'
# print(layer)
# dknn = DKNNL2(net, x_train, y_train, x_valid, y_valid, [layer],
#               k=1, num_classes=10)
#
# with torch.no_grad():
#     y_pred = dknn.classify(x_test)
#     print((y_pred.argmax(1) == y_test.numpy()).sum() / y_test.size(0))
#
# rep_train = dknn.get_activations(
#     x_train, requires_grad=False, device='cpu')[layer]
# rep_test = dknn.get_activations(
#     x_test, requires_grad=False, device='cpu')[layer]
# rep_valid = dknn.get_activations(
#     x_valid, requires_grad=False, device='cpu')[layer]
# rep_train = rep_train.numpy()
# rep_test = rep_test.numpy()
# rep_valid = rep_valid.numpy()
#
# knn = KNNL2NP(rep_train, y_train.numpy(),
#               rep_valid, y_valid.numpy(),
#               k=1, num_classes=10)
#
# rep_adv = knn.opt_attack(rep_test[:10], y_test.numpy()[:10], iterations=10)
# pickle.dump(rep_adv, open('rep_adv.p', 'rb'))
#
# dist = np.linalg.norm(((rep_test - rep_adv)**2).reshape(rep_test.shape[0], -1))
# print(dist.mean())
#
# perts = [1, 2, 3]
# a = []
# for pert in perts:
#     a.append((dist > pert).mean())
# print(a)

gaps = []
for layer in layers:
    output = '(' + layer + ') '
    dknn = DKNNL2(net, x_train, y_train, x_valid, y_valid, [layer],
                  k=1, num_classes=10)

    with torch.no_grad():
        y_pred = dknn.classify(x_test)
        output += '%.4f, ' % ((y_pred.argmax(1) ==
                               y_test.numpy()).sum() / y_test.size(0))

    rep_train = dknn.get_activations(
        x_train, requires_grad=False, batch_size=100, device='cpu')[layer]
    rep_test = dknn.get_activations(
        x_test, requires_grad=False, batch_size=100, device='cpu')[layer]
    rep_valid = dknn.get_activations(
        x_valid, requires_grad=False, batch_size=100, device='cpu')[layer]

    rep_train = rep_train.numpy()
    rep_test = rep_test.numpy()
    rep_valid = rep_valid.numpy()

    knn = KNNL2NP(rep_train, y_train, rep_valid, y_valid, k=1, num_classes=10)

    # print('finding adv neighbors...')
    rep_nn = knn.find_nn_diff_class(rep_test, y_test)
    # rep_adv = knn.get_min_dist(rep_test, y_test, rep_nn, iterations=10)
    # print('calculating margin...')
    gap, ind = knn.get_margin_bound(rep_test, y_test, rep_nn)

    # move perpendicular to edge
    # rep_adv = knn.opt_attack(rep_test, y_test, pert_bound=2, iterations=10)

    output += '%.4f, [' % gap[ind].mean().item()

    perts = [0.5, 1, 1.5, 2]
    a = []
    for pert in perts[:-1]:
        output += '%.4f, ' % (gap[ind] > 2 * pert).mean()
    output += '%.4f]' % (gap[ind] > 2 * perts[-1]).mean()

    print(output)
    gaps.append(gap[ind])

    # dist = ((rep_test - rep_adv)**2).view(rep_test.size(0), -1).sum(1).sqrt()
    # dist = ((rep_test - rep_adv) **
    #         2).reshape(rep_test.shape[0], -1).sum(1).sqrt()
    # dist = dist[dist != 0]
    # print(dist.mean())
    #
    # perts = [0.5, 1, 1.5, 2]
    # a = []
    # for pert in perts:
    #     a.append((dist > pert).mean())
    # print(a)

# pickle.dump(gaps, open('gaps_dist11.p', 'wb'))
