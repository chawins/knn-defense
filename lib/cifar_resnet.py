'''
Resnet code taken from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
https://github.com/kuangliu/pytorch-cifar
'''

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

# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)
#
#
# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
#                      bias=False)
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None,
#                  kernel_size=3):
#         super(Bottleneck, self).__init__()
#         self.conv1 = conv1x1(inplanes, planes)
#         self.bn1 = nn.BatchNorm2d(planes)
#         if kernel_size == 3:
#             self.conv2 = conv3x3(planes, planes, stride)
#         else:
#             self.conv2 = conv1x1(planes, planes, stride)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = conv1x1(planes, planes * self.expansion)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
#         super(ResNet, self).__init__()
#
#         self.mean = nn.Parameter(
#             data=torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1),
#             requires_grad=False)
#         self.std = nn.Parameter(
#             data=torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1),
#             requires_grad=False)
#         self.inplanes = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual
#         # block behaves like an identity. This improves the model by 0.2~0.3%
#         # according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = (x - self.mean) / self.std
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x
