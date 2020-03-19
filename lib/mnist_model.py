'''
Define MNIST models
'''

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class KNNModel(nn.Module):
    '''
    A Pytorch model that apply an identiy function to the input (i.e. output =
    input). It is used to simulate kNN on the input space so that it is
    compatible with attacks implemented for DkNN.
    '''

    def __init__(self):
        super(KNNModel, self).__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.identity(x)
        return x


# ============================================================================ #


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


# ============================================================================ #


class BasicModelV2(nn.Module):

    def __init__(self, num_classes=10):
        super(BasicModelV2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================================ #


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


# ============================================================================ #


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


# ============================================================================ #


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


# ============================================================================ #


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


# ============================================================================ #


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


# ============================================================================ #


class Autoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim=20):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.input_dim_flat = 1
        for dim in input_dim:
            self.input_dim_flat *= dim
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2048, 400)
        self.relu4 = nn.ReLU(inplace=True)
        self.latent = nn.Linear(400, latent_dim)

        self.de_fc1 = nn.Linear(latent_dim, 400)
        self.relu5 = nn.ReLU(inplace=True)
        self.de_fc2 = nn.Linear(400, self.input_dim_flat)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc(x))
        x = self.latent(x)
        return x

    def decode(self, z):
        x = self.relu5(self.de_fc1(z))
        x = self.de_fc2(x)
        out_dim = (z.size(0), ) + self.input_dim
        return x.view(out_dim)

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return z, out

    def loss_function(self, latent, x_recon, inputs, targets):
        # MSE loss
        return torch.sum((inputs - x_recon) ** 2)


# ============================================================================ #


class NCAModelV3(nn.Module):

    def __init__(self, normalize=False, output_dim=100, num_classes=10,
                 init_it=1e-2, train_it=False, train_data=None):
        super(NCAModelV3, self).__init__()
        self.normalize = normalize
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc_ = nn.Linear(2048, output_dim)
        # self.fc = nn.Identity()
        self.fc = nn.Sigmoid()

        # initialize inverse temperature for each layer
        self.log_it = torch.nn.Parameter(
            data=torch.tensor(np.log(init_it)), requires_grad=train_it)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.train_data = train_data
        self.train_rep = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc_(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        x = self.fc(x)
        return x

    def forward_adv(self, x_orig, y_target, params):
        """
        """
        epsilon = params['epsilon']
        step_size = params['step_size']
        num_steps = params['num_steps']
        rand = params['random_start']

        # training samples that we want to query against should not be perturbed
        # so we keep an extra copy and detach it from gradient computation
        with torch.no_grad():
            outputs_orig = self.forward(x_orig).detach()

        x = x_orig.clone()
        if rand:
            noise = torch.zeros_like(x).normal_(0, 1).view(x.size(0), -1)
            x += noise.renorm(2, 0, epsilon).view(x.size())

        for _ in range(num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                outputs = self.forward(x)
                p_target = self.get_prob(
                    outputs, y_target, x_orig=outputs_orig)
                loss = - torch.log(p_target).sum()
            grad = torch.autograd.grad(loss, x)[0].detach()
            grad_norm = grad.view(x.size(0), -1).norm(2, 1).clamp(1e-5, 1e9)
            delta = step_size * grad / grad_norm.view(x.size(0), 1, 1, 1)
            x = x.detach() + delta
            diff = (x - x_orig).view(x.size(0), -1).renorm(2, 0, epsilon)
            x = diff.view(x.size()) + x_orig
            x.clamp_(0, 1)

        return outputs_orig, self.forward(x)

    def get_prob(self, x, y_target, x_orig=None):
        """
        If x_orig is given, compute distance w.r.t. x_orig instead of samples
        in the same batch (x). It is intended to be used with adversarial
        training.
        """
        if x_orig is not None:
            assert x.size(0) == x_orig.size(0)

        batch_size = x.size(0)
        device = x.device
        x = x.view(batch_size, -1)
        if x_orig is not None:
            x_orig = x_orig.view(batch_size, -1)
            x_repeat = x_orig.repeat(batch_size, 1, 1).transpose(0, 1)
        else:
            x_repeat = x.repeat(batch_size, 1, 1).transpose(0, 1)
        dist = ((x_repeat - x) ** 2).sum(2) * self.log_it.exp()
        exp = torch.exp(- dist.clamp(- 50, 50))
        mask_not_self = 1 - torch.eye(batch_size, device=device)
        mask_same = (y_target.repeat(batch_size, 1).transpose(0, 1) ==
                     y_target).float()
        p_target = ((mask_not_self * mask_same * exp).sum(0) /
                    (mask_not_self * exp).sum(0))
        # p_target = (mask_same * exp).sum(0) / exp.sum(0)
        # Prevent the case where there's only one sample in the batch from a
        # certain clss, resulting in p_target being 0
        p_target = torch.max(p_target, torch.tensor(1e-30).to(device))

        return p_target

    def loss_function(self, output, y_target, orig=None):
        """soft nearest neighbor loss"""
        p_target = self.get_prob(output, y_target, x_orig=orig)
        # y_pred = p_target.max(1)
        loss = - torch.log(p_target)
        return loss.mean()

    # def loss_function(self, x, y_target, orig=None):
    #     """"""
    #     x_orig = orig
    #     if x_orig is not None:
    #         assert x.size(0) == x_orig.size(0)
    #
    #     batch_size = x.size(0)
    #     device = x.device
    #     x = x.view(batch_size, -1)
    #     if x_orig is not None:
    #         x_orig = x_orig.view(batch_size, -1)
    #         x_repeat = x_orig.repeat(batch_size, 1, 1).transpose(0, 1)
    #     else:
    #         x_repeat = x.repeat(batch_size, 1, 1).transpose(0, 1)
    #     dist = ((x_repeat - x) ** 2).sum(2)
    #     mask_same = (y_target.repeat(batch_size, 1).transpose(0, 1) ==
    #                  y_target).float()
    #     # (1) fixed distance
    #     # (2) pick k-th NN distance
    #     dist_k = torch.topk(
    #         dist, 10, largest=False, dim=0)[0][-1, :].unsqueeze(0).detach()
    #     loss = (F.relu(1 + dist - dist_k) * mask_same).sum(0)
    #     loss += (F.relu(1 - dist + dist_k) * (1 - mask_same)).sum(0)
    #     # (3) all pairs
    #     # (4) take mean
    #     # dist_same = (dist * mask_same).sum(0) / mask_same.sum(0)
    #     # dist_diff = (dist * (1 - mask_same)).sum(0) / \
    #     #     (batch_size - mask_same.sum(0))
    #     # loss = F.relu(1 + dist_same - dist_diff)
    #
    #     return loss.mean()

    def get_train_rep(self, batch_size=200, requires_grad=False):
        """update self.train_rep by running it through the current model"""
        if self.train_data is None:
            raise ValueError(
                'Cannot compute train rep as train data is not provided.')
        x_train, _ = self.train_data
        device = self._get_device()
        train_rep = torch.zeros((x_train.size(0), self.output_dim),
                                device=device, requires_grad=requires_grad)
        num_batches = np.ceil(x_train.size(0) // batch_size).astype(np.int32)
        with torch.set_grad_enabled(requires_grad):
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                train_rep[start:end] = self.forward(
                    x_train[start:end].to(device))
        return train_rep

    def recompute_train_rep(self):
        self.train_rep = self.get_train_rep(requires_grad=False)

    def compute_logits(self, x, recompute_train_rep=False, requires_grad=False,
                       from_outputs=False):

        if recompute_train_rep:
            self.recompute_train_rep()
        _, y_train = self.train_data
        device = self._get_device()
        # logits = torch.zeros((x.size(0), self.num_classes), device=x.device,
        #                      requires_grad=requires_grad)
        logits = []
        with torch.set_grad_enabled(requires_grad):
            if not from_outputs:
                rep = self.forward(x.to(device))
            else:
                rep = x
            dist = ((self.train_rep - rep.unsqueeze(1)) ** 2).sum(2)
            exp = torch.exp(- dist.clamp(- 50, 50) * self.log_it.exp())
            for j in range(self.num_classes):
                mask_j = (y_train == j).float().to(device)
                logits.append(
                    ((mask_j * exp).sum(1) / exp.sum(1)).unsqueeze(-1))
        return torch.cat(logits, dim=-1)

    def _get_device(self):
        return next(self.parameters()).device


class WeightedNCA(NCAModelV3):
    def __init__(self, normalize=False, output_dim=100, num_classes=10,
                 init_it=1e-2, train_it=False, train_data=None):
        super().__init__(normalize=normalize, output_dim=output_dim,
                         num_classes=num_classes, init_it=init_it,
                         train_it=train_it, train_data=train_data)
        self.weights = torch.nn.Parameter(
            data=torch.ones(len(self.train_data[0])), requires_grad=True)

    def compute_logits(self, x, recompute_train_rep=False, requires_grad=False,
                       from_outputs=False):

        if recompute_train_rep:
            self.recompute_train_rep()
        _, y_train = self.train_data
        device = self._get_device()
        # logits = torch.zeros((x.size(0), self.num_classes), device=x.device,
        #                      requires_grad=requires_grad)
        logits = []
        with torch.set_grad_enabled(requires_grad):
            if not from_outputs:
                rep = self.forward(x.to(device))
            else:
                rep = x
            dist = ((self.train_rep - rep.unsqueeze(1)) ** 2).sum(2)
            exp = torch.exp(- dist.clamp(- 50, 50) * self.log_it.exp())
            exp *= self.weights.sigmoid()
            for j in range(self.num_classes):
                mask_j = (y_train == j).float().to(device)
                logits.append(
                    ((mask_j * exp).sum(1) / exp.sum(1)).unsqueeze(-1))
        return torch.cat(logits, dim=-1)


class SoftLabelNCA(NCAModelV3):
    def __init__(self, ys_train, normalize=False, output_dim=100, num_classes=10,
                 init_it=1e-2, train_it=False, train_data=None):
        super().__init__(normalize=normalize, output_dim=output_dim,
                         num_classes=num_classes, init_it=init_it,
                         train_it=train_it, train_data=train_data)
        self.ys_train = ys_train

    def recompute_ys_train(self, k):
        self.recompute_train_rep()
        y_train = self.train_data[1]
        for i in range(len(y_train)):
            dist = ((self.train_rep[i] - self.train_rep) ** 2).sum(1)
            nb = torch.topk(dist, k, largest=False)[1]
            ys = np.bincount(
                y_train[nb].numpy(), minlength=self.num_classes) / k
            self.ys_train[i] = torch.tensor(ys, device='cuda').float()

    # def get_prob(self, x, y_target, x_orig=None):
    #     """
    #     If x_orig is given, compute distance w.r.t. x_orig instead of samples
    #     in the same batch (x). It is intended to be used with adversarial
    #     training.
    #     """
    #     if x_orig is not None:
    #         assert x.size(0) == x_orig.size(0)
    #
    #     batch_size = x.size(0)
    #     device = x.device
    #     x = x.view(batch_size, -1)
    #     if x_orig is not None:
    #         x_orig = x_orig.view(batch_size, -1)
    #         x_repeat = x_orig.repeat(batch_size, 1, 1).transpose(0, 1)
    #     else:
    #         x_repeat = x.repeat(batch_size, 1, 1).transpose(0, 1)
    #     dist = ((x_repeat - x) ** 2).sum(2) * self.log_it.exp()
    #     exp = torch.exp(- dist.clamp(- 50, 50))
    #     mask_same = (y_target.repeat(batch_size, 1).transpose(0, 1) ==
    #                  y_target).float()
    #     import pdb
    #     pdb.set_trace()
    #     p_target = (mask_same * (exp @ self.ys_train)).sum(0) / exp.sum(0)
    #     # Prevent the case where there's only one sample in the batch from a
    #     # certain clss, resulting in p_target being 0
    #     p_target = torch.max(p_target, torch.tensor(1e-30).to(device))
    #
    #     return p_target

    def compute_logits(self, x, recompute_train_rep=False, requires_grad=False,
                       from_outputs=False):

        if recompute_train_rep:
            self.recompute_train_rep()
        _, y_train = self.train_data
        device = self._get_device()
        # logits = torch.zeros((x.size(0), self.num_classes), device=x.device,
        #                      requires_grad=requires_grad)
        logits = []
        with torch.set_grad_enabled(requires_grad):
            if not from_outputs:
                rep = self.forward(x.to(device))
            else:
                rep = x
            dist = ((self.train_rep - rep.unsqueeze(1)) ** 2).sum(2)
            exp = torch.exp(- (dist * self.log_it.exp()).clamp(- 50, 50))
            probs = exp @ self.ys_train
            logits = probs / exp.sum(1).unsqueeze(1)
        return logits
