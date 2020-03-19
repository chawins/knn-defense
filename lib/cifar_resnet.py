'''
Define ResNet models for CIFAR-10 experiments

Code are adapted from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
https://github.com/kuangliu/pytorch-cifar
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
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
        self.linear = nn.Linear(512 * block.expansion, num_classes)

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
        out = self.linear(out)
        return out


# ============================================================================ #


class PreActResNet_VAE(nn.Module):
    def __init__(self, block, num_blocks, latent_dim=10):
        super(PreActResNet_VAE, self).__init__()
        self.in_planes = 64
        self.latent_dim = latent_dim

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
        self.mu = nn.Linear(512 * block.expansion, latent_dim)
        self.logvar = nn.Linear(512 * block.expansion, latent_dim)

        # decoder
        self.de_fc = nn.Linear(latent_dim, 2048)
        self.de_relu1 = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=0)
        self.de_relu2 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=0)
        self.de_relu3 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2, padding=0)
        self.de_relu4 = nn.ReLU(inplace=True)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 3, stride=1, padding=0)
        self.de_sig = nn.Sigmoid()
        # self.deconv4 = nn.ConvTranspose2d(32, 6, 3, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def encode(self, x):
        x = (x - self.mean) / self.std
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        mu = self.mu(out)
        # TODO: use tanh activation on logvar if unstable
        # en_std = torch.exp(0.5 * x[:, self.latent_dim:])
        logvar = self.logvar(out).tanh()
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.de_relu1(self.de_fc(z))
        x = x.view(z.size(0), 128, 4, 4)
        x = self.de_relu2(self.deconv1(x))
        x = self.de_relu3(self.deconv2(x))
        x = self.de_relu4(self.deconv3(x))
        x = self.de_sig(self.deconv4(x))
        return x
        # x = self.deconv4(x)
        # return x[:, :3], x[:, 3:].tanh()

    def forward(self, x):
        en_mu, en_logvar = self.encode(x)
        z = self.reparameterize(en_mu, en_logvar)
        output = self.decode(z)
        return en_mu, en_logvar, output


# ============================================================================ #


class ResNetWrapper(nn.Module):
    def __init__(self, net, block=4, dim=100):
        super(ResNetWrapper, self).__init__()
        self.block = block

        self.conv1 = net.conv1
        if block >= 1:
            self.block1 = net.layer1
        if block >= 2:
            self.block2 = net.layer2
        if block >= 3:
            self.block3 = net.layer3
        if block >= 4:
            self.block4 = net.layer4
        if block >= 5:
            self.block5 = net.linear
        self.fc = nn.Linear(3072, 128)

        self.mean = nn.Parameter(data=torch.zeros(dim), requires_grad=False)
        self.std = nn.Parameter(data=torch.ones(dim), requires_grad=False)

    def forward(self, x):
        x = self.conv1(x)
        if self.block >= 1:
            x = self.block1(x)
        if self.block >= 2:
            x = self.block2(x)
        if self.block >= 3:
            x = self.block3(x)
        if self.block >= 4:
            x = self.block4(x)
        x = x.view(x.size(0), -1)
        if self.block >= 5:
            x = self.block5(x)
        x = (x - self.mean) / self.std
        x = self.fc(x)
        return x


# ============================================================================ #


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


# ============================================================================ #


class NCAModel(nn.Module):

    def __init__(self, block, num_blocks, output_dim=200, num_classes=10,
                 init_it=1e-2, train_it=False, train_data=None):
        super(NCAModel, self).__init__()
        self.num_classes = num_classes
        self.output_dim = output_dim
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
        self.fc = nn.Linear(512 * block.expansion, output_dim)

        # initialize inverse temperature for each layer
        self.log_it = torch.nn.Parameter(
            data=torch.tensor(np.log(init_it)), requires_grad=train_it)

        if train_data is not None:
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
        out = self.fc(out)
        return out

    def forward_adv(self, x, y_target, step_size, num_steps, rand):
        """
        """
        # training samples that we want to query against should not be perturbed
        # so we keep an extra copy and detach it from gradient computation
        outputs_orig = self.forward(x)

        x = x.detach()
        if rand:
            # x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            x = x + torch.zeros_like(x).normal_(0, step_size)

        for _ in range(num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                outputs = self.forward(x)
                p_target = self.get_prob(
                    outputs, y_target, x_orig=outputs_orig.detach())
                loss = - torch.log(p_target).sum()
            grad = torch.autograd.grad(loss, x)[0].detach()
            grad_norm = torch.max(
                grad.view(x.size(0), -1).norm(2, 1), torch.tensor(1e-5).to(x.device))
            delta = step_size * grad / grad_norm.view(x.size(0), 1, 1, 1)
            x = x.detach() + delta
            # x = torch.min(torch.max(x, inputs - self.epsilon),
            #               inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)

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
        exp = torch.exp(- torch.min(dist, torch.tensor(50.).to(device)))
        mask_not_self = 1 - torch.eye(batch_size, device=device)
        mask_same = (y_target.repeat(batch_size, 1).transpose(0, 1) ==
                     y_target).float()
        p_target = ((mask_not_self * mask_same * exp).sum(0) /
                    (mask_not_self * exp).sum(0))
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

    def get_train_rep(self, batch_size=200, requires_grad=False):
        """update self.train_rep by running it through the current model"""
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

    def compute_logits(self, x, recompute_train_rep=False, requires_grad=False):

        if recompute_train_rep:
            self.train_rep = self.get_train_rep(requires_grad=False)
        _, y_train = self.train_data
        device = self._get_device()
        # logits = torch.zeros((x.size(0), self.num_classes), device=x.device,
        #                      requires_grad=requires_grad)
        logits = []
        with torch.set_grad_enabled(requires_grad):
            rep = self.forward(x.to(device))
            dist = ((self.train_rep - rep.unsqueeze(1)) ** 2).sum(2)
            exp = torch.exp(- torch.min(dist, torch.tensor(50.).to(device)))
            for j in range(self.num_classes):
                mask_j = (y_train == j).float().to(device)
                logits.append(
                    ((mask_j * exp).sum(1) / exp.sum(1)).unsqueeze(-1))
        return torch.cat(logits, dim=-1)

    def _get_device(self):
        return next(self.parameters()).device


class NCAModelV2(nn.Module):

    def __init__(self, block, num_blocks, output_dim=200, num_classes=10,
                 init_it=1e-2, train_it=False, train_data=None):
        super(NCAModelV2, self).__init__()
        self.num_classes = num_classes
        self.output_dim = output_dim
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
        self.fc = nn.Identity()

        # initialize inverse temperature for each layer
        self.log_it = torch.nn.Parameter(
            data=torch.tensor(np.log(init_it)), requires_grad=train_it)

        if train_data is not None:
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
        out = F.normalize(out, p=2, dim=1)
        out = self.fc(out)
        return out

    def forward_adv(self, x, y_target, step_size, num_steps, rand):
        """
        """
        # training samples that we want to query against should not be perturbed
        # so we keep an extra copy and detach it from gradient computation
        outputs_orig = self.forward(x)

        x = x.detach()
        if rand:
            # x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            x = x + torch.zeros_like(x).normal_(0, step_size)

        for _ in range(num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                outputs = self.forward(x)
                p_target = self.get_prob(
                    outputs, y_target, x_orig=outputs_orig.detach())
                loss = - torch.log(p_target).sum()
            grad = torch.autograd.grad(loss, x)[0].detach()
            grad_norm = torch.max(
                grad.view(x.size(0), -1).norm(2, 1), torch.tensor(1e-5).to(x.device))
            delta = step_size * grad / grad_norm.view(x.size(0), 1, 1, 1)
            x = x.detach() + delta
            # x = torch.min(torch.max(x, inputs - self.epsilon),
            #               inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)

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
        exp = torch.exp(- torch.min(dist, torch.tensor(50.).to(device)))
        mask_not_self = 1 - torch.eye(batch_size, device=device)
        mask_same = (y_target.repeat(batch_size, 1).transpose(0, 1) ==
                     y_target).float()
        p_target = ((mask_not_self * mask_same * exp).sum(0) /
                    (mask_not_self * exp).sum(0))
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

    def get_train_rep(self, batch_size=200, requires_grad=False):
        """update self.train_rep by running it through the current model"""
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

    def compute_logits(self, x, recompute_train_rep=False, requires_grad=False):

        if recompute_train_rep:
            self.train_rep = self.get_train_rep(requires_grad=False)
        _, y_train = self.train_data
        device = self._get_device()
        # logits = torch.zeros((x.size(0), self.num_classes), device=x.device,
        #                      requires_grad=requires_grad)
        logits = []
        with torch.set_grad_enabled(requires_grad):
            rep = self.forward(x.to(device))
            dist = ((self.train_rep - rep.unsqueeze(1)) ** 2).sum(2)
            exp = torch.exp(- torch.min(dist, torch.tensor(50.).to(device)))
            for j in range(self.num_classes):
                mask_j = (y_train == j).float().to(device)
                logits.append(
                    ((mask_j * exp).sum(1) / exp.sum(1)).unsqueeze(-1))
        return torch.cat(logits, dim=-1)

    def _get_device(self):
        return next(self.parameters()).device
