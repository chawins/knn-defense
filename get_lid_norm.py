import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from lib.adv_model import *
from lib.dataset_utils import *
from lib.dknn import DKNN, DKNNL2
from lib.mnist_model import *
from lib.utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

exp_id = 8

# model_name = 'train_mnist_exp%d.h5' % exp_id
# net = BasicModel()

model_name = 'train_mnist_snnl_exp%d.h5' % exp_id
net = SNNLModel(train_it=True)

# model_name = 'train_mnist_hidden_mixup_exp%d.h5' % exp_id
# net = HiddenMixupModel()

# model_name = 'train_mnist_vae_exp%d.h5' % exp_id
# net = VAE((1, 28, 28), num_classes=10, latent_dim=20)
# net = VAE2((1, 28, 28), num_classes=10, latent_dim=20)

# model_name = 'train_mnist_cav_exp%d.h5' % exp_id
# net = ClassAuxVAE((1, 28, 28), num_classes=10, latent_dim=20)

# model_name = 'adv_mnist_exp%d.h5' % exp_id
# # basic_net = BasicModel()
# basic_net = BasicModelV2()
# config = {'epsilon': 0.3,
#           'num_steps': 40,
#           'step_size': 0.01,
#           'random_start': True,
#           'loss_func': 'xent'}
# net = PGDModel(basic_net, config)

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
# net = net.basic_net
net.eval()

(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist_all(
    '/data', val_size=0.1, seed=seed)

layers = ['relu1', 'relu2', 'relu3', 'fc']
# layers = ['relu1', 'relu2', 'relu3', 'en_mu']
# layers = ['relu1', 'relu2', 'relu3']
# layers = ['relu3']
# layers = ['en_conv3']
# layers = ['en_mu']
# layers = ['maxpool1', 'maxpool2', 'relu3', 'fc2']
# net = net.cpu()
with torch.no_grad():
    # dknn = DKNN(net, x_train, y_train, x_valid, y_valid, layers,
    #             k=75, num_classes=10)
    dknn = DKNNL2(net, x_train, y_train, x_valid, y_valid, layers,
                  k=75, num_classes=10)

x = x_test.requires_grad_(True)[:1000]

norms = compute_spnorm(x, dknn, layers)
print(', '.join('%.4f' % i for i in norms.mean(0)))

lid = np.zeros((x.size(0), len(layers)))
reps = dknn.get_activations(x, requires_grad=False)
train_reps = dknn.get_activations(x_train, requires_grad=False)

for l, layer in enumerate(layers):
    lid[:, l] = compute_lid(
        reps[layer], train_reps[layer], 3000, exclude_self=False)
print(', '.join('%.4f' % i for i in lid.mean(0)))
