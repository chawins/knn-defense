'''
Define CIFAR-10 CNN models excluding ResNet models which are in cifar_resnet.py
'''

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from lib.cifar_resnet import *


class CIFAR10_VAE(nn.Module):

    def __init__(self, input_dim, num_classes=10, latent_dim=2):
        super(CIFAR10_VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.input_dim_flat = 1
        for dim in input_dim:
            self.input_dim_flat *= dim
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(8192, 128)
        self.relu5 = nn.ReLU(inplace=True)
        self.en_mu = nn.Linear(128, latent_dim)
        self.en_logvar = nn.Linear(128, latent_dim)

        self.de_fc1 = nn.Linear(latent_dim, 128)
        self.de_relu1 = nn.ReLU(inplace=True)
        self.de_fc2 = nn.Linear(128, 8192)
        self.de_relu2 = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)
        self.de_relu3 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)
        self.de_relu4 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0)
        self.de_relu5 = nn.ReLU(inplace=True)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1)
        self.sig = nn.Sigmoid()

    def encode(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.relu5(self.fc(x))
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
        x = self.de_relu2(self.de_fc2(x))
        x = x.view(z.size(0), 32, 16, 16)
        x = self.de_relu3(self.deconv1(x))
        x = self.de_relu4(self.deconv2(x))
        x = self.de_relu5(self.deconv3(x))
        x = self.sig(self.deconv4(x))
        return x

    def forward(self, x):
        en_mu, en_logvar = self.encode(x)
        z = self.reparameterize(en_mu, en_logvar)
        output = self.decode(z)
        return en_mu, en_logvar, output


# ============================================================================ #


class CIFAR10_AE(nn.Module):

    def __init__(self, input_dim, latent_dim=20):
        super(CIFAR10_AE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.input_dim_flat = 1
        for dim in input_dim:
            self.input_dim_flat *= dim
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(8192, 128)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, latent_dim)

        self.de_fc1 = nn.Linear(latent_dim, 128)
        self.de_relu1 = nn.ReLU(inplace=True)
        self.de_fc2 = nn.Linear(128, 8192)
        self.de_relu2 = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)
        self.de_relu3 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)
        self.de_relu4 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0)
        self.de_relu5 = nn.ReLU(inplace=True)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1)
        # self.sig = nn.Sigmoid()

    def encode(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.relu5(self.fc1(x))
        x = self.fc2(x)
        return x

    def decode(self, z):
        x = self.de_relu1(self.de_fc1(z))
        x = self.de_relu2(self.de_fc2(x))
        x = x.view(z.size(0), 32, 16, 16)
        x = self.de_relu3(self.deconv1(x))
        x = self.de_relu4(self.deconv2(x))
        x = self.de_relu5(self.deconv3(x))
        # x = self.sig(self.deconv4(x))
        x = self.deconv4(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return z, out

    def loss_function(self, latent, x_recon, inputs, targets):
        # MSE loss
        return torch.sum((inputs - x_recon) ** 2)


# ============================================================================ #


class NCAModel(nn.Module):

    def __init__(self, block, num_blocks, normalize=False, output_dim=100,
                 num_classes=10, init_it=1e-2, train_it=False, train_data=None):
        super(NCAModel, self).__init__()
        self.normalize = normalize
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.in_planes = 64
        self.mean = nn.Parameter(
            data=torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1),
            requires_grad=False)
        self.std = nn.Parameter(
            data=torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1),
            requires_grad=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc_ = nn.Linear(512 * block.expansion, output_dim)
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

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc_(out)
        if self.normalize:
            out = F.normalize(out, p=2, dim=1)
        out = self.fc(out)
        return out

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
