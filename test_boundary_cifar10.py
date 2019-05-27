import os
import pdb
import pickle
import time

import numpy as npr
import torch
import torch.backends.cudnn as cudnn

from foolbox.criteria import Misclassification
from foolbox.distances import MeanSquaredDistance
from lib.adv_model import *
from lib.cifar_resnet import *
from lib.dataset_utils import *
from lib.dknn import *
from lib.foolbox_model import *
from lib.lip_model import *
from lib.utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

exp_id = 0

model_name = 'adv_cifar10_exp%d.h5' % exp_id
# model_name = 'train_cifar10_vae_exp%d.h5' % exp_id
# model_name = 'rot_cifar10_exp%d.h5' % exp_id
# model_name = 'ae_cifar10_exp%d.h5' % exp_id

net = PreActResNet(PreActBlock, [2, 2, 2, 2]).eval()
config = {'num_steps': 8,
          'step_size': 0.05,
          'random_start': True,
          'loss_func': 'xent'}
net = PGDL2Model(net, config)

# net = PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=4)
# net.load_state_dict(torch.load('saved_models/' + model_name))
# net = net.eval().to('cuda')

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

(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_cifar10_all(
    '/data', val_size=0.1, seed=seed)


layers = ['layer4']
# net = net.cpu()
with torch.no_grad():
    # dknn = DKNN(net, x_train, y_train, x_valid, y_valid, layers,
    #             k=75, num_classes=10)
    dknn = DKNNL2(net, x_train, y_train, x_valid, y_valid, layers,
                  k=75, num_classes=10)
    # dknn = DKNNL2Approx(net, x_train, y_train, x_valid, y_valid, layers,
    #                     k=1, num_classes=10)
    y_pred = dknn.classify(x_test)
    ind = np.where(y_pred.argmax(1) == y_test.numpy())[0]
    print((y_pred.argmax(1) == y_test.numpy()).sum() / y_test.size(0))

dknn_fb = DkNNFoolboxModel(dknn, (0, 1), 1, preprocessing=(0, 1))
criterion = Misclassification()
distance = MeanSquaredDistance

attack = foolbox.attacks.BoundaryAttack(
    model=dknn_fb, criterion=criterion, distance=distance)

attack_params = {
    'iterations': 5000,
    'max_directions': 25,
    'starting_point': None,
    'initialization_attack': None,
    'log_every_n_steps': 100,
    'spherical_step': 0.5,
    'source_step': 0.05,
    'step_adaptation': 1.5,
    'batch_size': 1,
    'tune_batch_size': True,
    'threaded_rnd': True,
    'threaded_gen': True,
    'alternative_generator': False
}

num = 100
x_adv = np.zeros_like(x_test[:num].numpy())

start_time = time.time()
for i in range(num):
    print(i)
    x_adv[i] = attack(x_test[ind][i].numpy(), label=y_test[ind][i].numpy(),
                      unpack=True, verbose=False, **attack_params)
print(time.time() - start_time)

pickle.dump(x_adv, open('x_ba_cifar10_adv2_0.5_0.05.p', 'wb'))

y_pred = dknn.classify(torch.tensor(x_adv))
print((y_pred.argmax(1) == y_test[ind][:num].numpy()))
print((y_pred.argmax(1) == y_test[ind][:num].numpy()).sum() / num)

# y_clean = dknn.classify(x_test[:num])
# ind = (y_clean.argmax(1) == y_test[:num].numpy()) & (
#     y_pred.argmax(1) != y_test[:num].numpy())
dist = np.sqrt(np.sum((x_adv - x_test.numpy()[ind][:num])**2, (1, 2, 3)))
print(dist)
print(dist.mean())

# pickle.dump(x_adv, open('x_adv_boundary_dist11_0.5_0.05_2.p', 'wb'))
# pickle.dump(x_adv, open('x_adv_boundary_adv2_0.5_0.05.p', 'wb'))
