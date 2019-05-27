'''CIFAR-10 CNN models'''

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


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
