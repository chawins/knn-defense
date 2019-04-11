'''MNIST models'''

import copy
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class BasicModel(nn.Module):

    def __init__(self, num_classes=10):
        super(BasicModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ClassAuxVAE(nn.Module):

    def __init__(self, input_dim, num_classes=10, latent_dim=20):
        super(ClassAuxVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.input_dim_flat = 1
        for dim in input_dim:
            self.input_dim_flat *= dim
        self.en_conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.en_conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.en_conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        self.en_fc1 = nn.Linear(2048, 128)
        self.relu4 = nn.ReLU(inplace=True)
        self.en_mu = nn.Linear(128, latent_dim)
        self.en_logvar = nn.Linear(128, latent_dim)

        self.de_fc1 = nn.Linear(latent_dim, 128)
        self.de_fc2 = nn.Linear(128, self.input_dim_flat * 2)

        # TODO: experiment with different auxilary architecture
        self.ax_fc1 = nn.Linear(latent_dim, 128)
        self.ax_fc2 = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        x = self.relu1(self.en_conv1(x))
        x = self.relu2(self.en_conv2(x))
        x = self.relu3(self.en_conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.en_fc1(x))
        en_mu = self.en_mu(x)
        # TODO: use tanh activation on logvar if unstable
        # en_std = torch.exp(0.5 * x[:, self.latent_dim:])
        en_logvar = self.en_logvar(x)
        return en_mu, en_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.de_fc1(z))
        x = self.de_fc2(x)
        de_mu = x[:, :self.input_dim_flat]
        # de_std = torch.exp(0.5 * x[:, self.input_dim_flat:])
        de_logvar = x[:, self.input_dim_flat:].tanh()
        out_dim = (z.size(0), ) + self.input_dim
        return de_mu.view(out_dim).sigmoid(), de_logvar.view(out_dim)

    def auxilary(self, z):
        x = F.relu(self.ax_fc1(z))
        x = self.ax_fc2(x)
        return x

    def forward(self, x):
        en_mu, en_logvar = self.encode(x)
        z = self.reparameterize(en_mu, en_logvar)
        de_mu, de_logvar = self.decode(z)
        y = self.auxilary(z)
        return en_mu, en_logvar, de_mu, de_logvar, y


class VAE2(nn.Module):

    def __init__(self, input_dim, num_classes=10, latent_dim=20):
        super(VAE2, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.input_dim_flat = 1
        for dim in input_dim:
            self.input_dim_flat *= dim
        self.en_conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.en_conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.en_conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0)
        # self.relu3 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU()
        self.en_fc1 = nn.Linear(2048, 400)
        self.relu4 = nn.ReLU(inplace=True)
        self.en_mu = nn.Linear(400, latent_dim)
        self.en_logvar = nn.Linear(400, latent_dim)

        self.de_fc1 = nn.Linear(latent_dim, 400)
        self.de_relu1 = nn.ReLU(inplace=True)
        self.de_fc2 = nn.Linear(400, self.input_dim_flat)

    def encode(self, x):
        x = self.relu1(self.en_conv1(x))
        x = self.relu2(self.en_conv2(x))
        x = self.relu3(self.en_conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.en_fc1(x))
        en_mu = self.en_mu(x)
        # TODO: use tanh activation on logvar if unstable
        # en_std = torch.exp(0.5 * x[:, self.latent_dim:])
        en_logvar = self.en_logvar(x)
        return en_mu, en_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.de_relu1(self.de_fc1(z))
        x = self.de_fc2(x)
        out_dim = (z.size(0), ) + self.input_dim
        return x.view(out_dim).sigmoid()

    def forward(self, x):
        en_mu, en_logvar = self.encode(x)
        z = self.reparameterize(en_mu, en_logvar)
        output = self.decode(z)
        return en_mu, en_logvar, output


class VAE(nn.Module):

    def __init__(self, input_dim, num_classes=10, latent_dim=20):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.input_dim_flat = 1
        for dim in input_dim:
            self.input_dim_flat *= dim
        self.en_fc1 = nn.Linear(self.input_dim_flat, 400)
        self.en_relu1 = nn.ReLU(inplace=True)
        self.en_fc2 = nn.Linear(400, 400)
        self.en_relu2 = nn.ReLU(inplace=True)
        self.en_mu = nn.Linear(400, latent_dim)
        self.en_logvar = nn.Linear(400, latent_dim)

        self.de_fc1 = nn.Linear(latent_dim, 400)
        self.de_relu1 = nn.ReLU(inplace=True)
        self.de_fc2 = nn.Linear(400, self.input_dim_flat)

    def encode(self, x):
        x = x.view(-1, self.input_dim_flat)
        x = self.en_relu1(self.en_fc1(x))
        x = self.en_relu2(self.en_fc2(x))
        en_mu = self.en_mu(x)
        # TODO: use tanh activation on logvar if unstable
        # en_std = torch.exp(0.5 * x[:, self.latent_dim:])
        en_logvar = self.en_logvar(x)
        return en_mu, en_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.de_relu1(self.de_fc1(z))
        x = self.de_fc2(x)
        out_dim = (z.size(0), ) + self.input_dim
        return x.view(out_dim).sigmoid()

    def forward(self, x):
        en_mu, en_logvar = self.encode(x)
        z = self.reparameterize(en_mu, en_logvar)
        output = self.decode(z)
        return en_mu, en_logvar, output


class SNNLModel(nn.Module):

    def __init__(self, num_classes=10, train_it=False):
        super(SNNLModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2048, num_classes)

        # initialize inverse temperature for each layer
        self.it = torch.nn.Parameter(
            data=torch.tensor([-4.6, -4.6, -4.6]), requires_grad=train_it)

        # set up hook to get representations
        self.layers = ['relu1', 'relu2', 'relu3']
        self.activations = {}
        for name, module in self.named_children():
            if name in self.layers:
                module.register_forward_hook(self._get_activation(name))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output
        return hook

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def loss_function(self, x, y_target, alpha=-1):
        """soft nearest neighbor loss"""
        snn_loss = torch.zeros(1).cuda()
        y_pred = self.forward(x)
        for l, layer in enumerate(self.layers):
            rep = self.activations[layer]
            rep = rep.view(x.size(0), -1)
            for i in range(x.size(0)):
                mask_same = (y_target[i] == y_target).type(torch.float32)
                mask_self = torch.ones(x.size(0)).cuda()
                mask_self[i] = 0
                dist = ((rep[i] - rep) ** 2).sum(1) * self.it[l].exp()
                # dist = ((rep[i] - rep) ** 2).sum(1) * 0.01
                # TODO: get nan gradients at
                # Function 'MulBackward0' returned nan values in its 1th output.
                exp = torch.exp(- torch.min(dist, torch.tensor(50.).cuda()))
                # exp = torch.exp(- dist)
                snn_loss += torch.log(torch.sum(mask_self * mask_same * exp) /
                                      torch.sum(mask_self * exp))

        ce_loss = F.cross_entropy(y_pred, y_target)
        return y_pred, ce_loss - alpha / x.size(0) * snn_loss


class HiddenMixupModel(nn.Module):

    def __init__(self, num_classes=10):
        super(HiddenMixupModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, target=None, mixup_hidden=False, mixup_alpha=0.1,
                layer_mix=None):

        if mixup_hidden:
            if layer_mix is None:
                # TODO: which layers?
                layer_mix = random.randint(0, 4)

            if layer_mix == 0:
                x, y_a, y_b, lam = self.mixup_data(x, target, mixup_alpha)
            x = self.conv1(x)
            x = self.relu1(x)

            if layer_mix == 1:
                x, y_a, y_b, lam = self.mixup_data(x, target, mixup_alpha)
            x = self.conv2(x)
            x = self.relu2(x)

            if layer_mix == 2:
                x, y_a, y_b, lam = self.mixup_data(x, target, mixup_alpha)
            x = self.conv3(x)
            x = self.relu3(x)

            if layer_mix == 3:
                x, y_a, y_b, lam = self.mixup_data(x, target, mixup_alpha)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            if layer_mix == 4:
                x, y_a, y_b, lam = self.mixup_data(x, target, mixup_alpha)

            # lam = torch.tensor(lam).cuda()
            # lam = lam.repeat(y_a.size())
            return x, y_a, y_b, lam

        else:
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    @staticmethod
    def loss_function(y_pred, y_a, y_b, lam):
        loss = lam * F.cross_entropy(y_pred, y_a) + \
            (1 - lam) * F.cross_entropy(y_pred, y_b)
        return loss

    @staticmethod
    def mixup_data(x, y, alpha):
        '''
        Compute the mixup data. Return mixed inputs, pairs of targets, and
        lambda. Code from
        https://github.com/vikasverma1077/manifold_mixup/blob/master/supervised/models/utils.py
        '''
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        index = torch.randperm(x.size(0)).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
