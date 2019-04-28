import os
import time

import numpy as npr
import torch
import torch.backends.cudnn as cudnn

from foolbox.criteria import Misclassification
from foolbox.distances import MeanSquaredDistance
from lib.adv_model import *
from lib.dataset_utils import *
from lib.dknn import DKNN, DKNNL2
from lib.foolbox_model import *
from lib.mnist_model import *
from lib.utils import *

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

model_name = 'adv_mnist_exp%d.h5' % exp_id
basic_net = BasicModel()
# basic_net = BasicModelV2()
config = {'epsilon': 0.3,
          'num_steps': 40,
          'step_size': 0.01,
          'random_start': True,
          'loss_func': 'xent'}
net = PGDModel(basic_net, config)

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

# layers = ['relu1', 'relu2', 'relu3', 'fc']
# layers = ['relu1', 'relu2', 'relu3', 'en_mu']
# layers = ['relu1', 'relu2', 'relu3']
layers = ['relu3']
# layers = ['en_conv3']
# layers = ['en_mu']
# layers = ['maxpool1', 'maxpool2', 'relu3', 'fc2']
# net = net.cpu()
with torch.no_grad():
    # dknn = DKNN(net, x_train, y_train, x_valid, y_valid, layers,
    #             k=75, num_classes=10)
    dknn = DKNNL2(net, x_train, y_train, x_valid, y_valid, layers,
                  k=1, num_classes=10)

dknn_fb = DkNNFoolboxModel(dknn, (0, 1), 1, preprocessing=(0, 1))
criterion = Misclassification()
distance = MeanSquaredDistance

attack = foolbox.attacks.BoundaryAttack(
    model=dknn_fb, criterion=criterion, distance=distance)

attack_params = {
    'iterations': 5000,
    'max_directions': 50,
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

num = 10
x_adv = np.zeros_like(x_test[:num].numpy())

start_time = time.time()
for i in range(num):
    x_adv[i] = attack(x_test[i].numpy(), label=y_test[i].numpy(),
                      unpack=True, verbose=True, **attack_params)
print(time.time() - start_time)

y_pred = dknn.classify(torch.tensor(x_adv))
print((y_pred.argmax(1) == y_test[:num].numpy()))
print((y_pred.argmax(1) == y_test[:num].numpy()).sum() / num)

y_clean = dknn.classify(x_test[:num])
ind = (y_clean.argmax(1) == y_test[:num].numpy()) & (
    y_pred.argmax(1) != y_test[:num].numpy())
dist = np.sqrt(np.sum((x_adv[ind] - x_test.numpy()[:num][ind])**2, (1, 2, 3)))
print(dist)
print(dist.mean())
