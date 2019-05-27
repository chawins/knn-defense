import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def infty_norm(w):
    return w.abs().sum(1).max()


def row_sum(w):
    return torch.max(w.abs().sum(1) - 1, torch.tensor(0.).cuda()).sum()


def infty_norm_ub(w):
    wwt = torch.matmul(w, w.transpose(0, 1))
    wtw = torch.matmul(w.transpose(0, 1), w)
    norm = torch.min(infty_norm(wwt), infty_norm(wtw)).sqrt()
    return norm


def infty_norm_reg(w):
    wwt = torch.matmul(w, w.transpose(0, 1))
    wtw = torch.matmul(w.transpose(0, 1), w)
    reg = torch.min(row_sum(wwt), row_sum(wtw))
    return reg


class NormLinear(nn.Linear):

    def forward(self, input):
        # Frobenius norm is much smaller than sqrt(n) * infty_norm
        # norm = math.sqrt(self.weight.size(0)) * infty_norm(self.weight)
        # norm = self.weight.norm('fro')
        # norm = infty_norm_ub(self.weight)
        # return F.linear(input, self.weight / norm, self.bias)
        return F.linear(input, self.weight, self.bias)


class NormConv2d(nn.Conv2d):

    def forward(self, input):
        scaled_input = input / math.sqrt(
            math.ceil(self.kernel_size[0] / self.stride[0]) *
            math.ceil(self.kernel_size[1] / self.stride[1]))

        weight = self.weight.view(self.out_channels, -1)
        # this infinity norm bound is too loose
        # norm = math.sqrt(weight.size(0)) * infty_norm(weight)
        # norm = weight.norm('fro')
        # norm = infty_norm_ub(weight)
        # return F.conv2d(scaled_input, self.weight / norm, self.bias,
        #                 self.stride, self.padding, self.dilation, self.groups)
        return F.conv2d(scaled_input, self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class GroupSort(nn.Module):

    def __init__(self, num_groups, is_conv=False):
        super(GroupSort, self).__init__()
        self.num_groups = num_groups
        self.is_conv = is_conv

    def forward(self, x):
        group_size = int(x.size(1) / self.num_groups)
        assert self.num_groups * group_size == x.size(1)

        for i in range(self.num_groups):
            start = i * group_size
            end = (i + 1) * group_size
            x[:, start:end] = x[:, start:end].sort(1, descending=True)[0]
        return x


class TwoSidedReLU(nn.Module):

    def __init__(self, axis=1):
        super(TwoSidedReLU, self).__init__()
        self.axis = axis

    def forward(self, x):
        relu = F.relu(x)
        # neg_relu = - F.relu(-x)
        neg_relu = F.relu(-x)
        return torch.cat((relu, neg_relu), self.axis)


class LipschitzModel(nn.Module):

    def __init__(self, num_classes=10):
        super(LipschitzModel, self).__init__()
        self.conv1 = NormConv2d(1, 64, kernel_size=8, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = NormConv2d(64, 128, kernel_size=6, stride=2, padding=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = NormConv2d(128, 128, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc = NormLinear(2048, num_classes)

        # initialize inverse temperature for each layer
        # self.it = torch.nn.Parameter(
        #     data=torch.tensor(-4.6), requires_grad=True)

        for m in self.modules():
            if isinstance(m, NormConv2d):
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

    def loss_function(self, logits, label, margin=1):
        """Calculate margin loss"""
        label = label.view(-1, 1)
        other = self.best_other_class(logits, label)
        margin_loss = other - torch.gather(logits, 1, label) + margin
        margin_loss = torch.max(torch.zeros_like(margin_loss), margin_loss)
        return margin_loss.mean()

    @staticmethod
    def best_other_class(logits, exclude):
        """Returns the index of the largest logit, ignoring the class that
        is passed as `exclude`."""
        y_onehot = torch.zeros_like(logits)
        y_onehot.scatter_(1, exclude, 1)
        # make logits that we want to exclude a large negative number
        other_logits = logits - y_onehot * 1e9
        return other_logits.max(1)[0]


class NeighborModel(nn.Module):

    def __init__(self, num_classes=10, init_it=1, train_it=False):
        super(NeighborModel, self).__init__()
        # self.conv1 = NormConv2d(1, 64, kernel_size=8, stride=2, padding=3)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = NormConv2d(64, 128, kernel_size=6, stride=2, padding=3)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv3 = NormConv2d(128, 128, kernel_size=5, stride=1, padding=0)
        # self.relu3 = nn.ReLU(inplace=True)
        # # self.fc = NormLinear(2048, num_classes)
        # self.fc = NormLinear(2048, 10)

        # GroupSort
        # num_groups = 1
        # self.conv1 = NormConv2d(1, 64, kernel_size=8, stride=2, padding=3)
        # self.gs1 = GroupSort(num_groups, is_conv=True)
        # # self.gs1 = GroupSort(32, is_conv=True)
        # self.conv2 = NormConv2d(64, 128, kernel_size=6, stride=2, padding=3)
        # self.gs2 = GroupSort(num_groups, is_conv=True)
        # # self.gs2 = GroupSort(64, is_conv=True)
        # self.conv3 = NormConv2d(128, 128, kernel_size=5, stride=1, padding=0)
        # self.gs3 = GroupSort(num_groups, is_conv=True)
        # # self.gs3 = GroupSort(64, is_conv=True)
        # self.fc = NormLinear(2048, 128)

        # two-sided ReLU
        self.conv1 = NormConv2d(1, 32, kernel_size=8, stride=2, padding=3)
        self.relu1 = TwoSidedReLU()
        self.conv2 = NormConv2d(64, 64, kernel_size=6, stride=2, padding=3)
        self.relu2 = TwoSidedReLU()
        self.conv3 = NormConv2d(128, 128, kernel_size=5, stride=1, padding=0)
        self.relu3 = TwoSidedReLU()
        self.fc1 = NormLinear(4096, 512)
        self.relu4 = TwoSidedReLU()
        self.fc2 = NormLinear(1024, 128)
        # self.conv1 = NormConv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # self.relu1 = TwoSidedReLU()
        # self.conv2 = NormConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.relu2 = TwoSidedReLU()
        # self.conv3 = NormConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # self.relu3 = TwoSidedReLU()
        # self.fc1 = NormLinear(4096, 512)
        # self.relu4 = TwoSidedReLU()
        # self.fc2 = NormLinear(1024, 10)

        # initialize inverse temperature for each layer
        self.it = torch.nn.Parameter(
            data=torch.tensor(math.log(init_it)), requires_grad=train_it)

        for m in self.modules():
            if isinstance(m, NormConv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.conv3(x)
        # x = self.relu3(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # x = self.conv1(x)
        # x = self.gs1(x)
        # x = self.conv2(x)
        # x = self.gs2(x)
        # x = self.conv3(x)
        # x = self.gs3(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x

    def loss_function(self, logits, label):
        """Calculate neighborhood loss"""
        # SNN (or NCA) loss
        # snn_loss = torch.zeros(1).cuda()
        # const = torch.tensor(1e0).cuda()
        #
        # for i in range(logits.size(0)):
        #     mask_same = (label[i] == label).type(torch.float32)
        #     mask_self = torch.ones(logits.size(0)).cuda()
        #     mask_self[i] = 0
        #     dist = ((logits[i] - logits) ** 2).sum(1) * self.it.exp()
        #     # upper bound distance to prevent overflow
        #     exp = torch.exp(- torch.min(dist, torch.tensor(50.).cuda()))
        #     # exp = torch.exp(- dist)
        #     # SNN loss has log
        #     # snn_loss -= torch.log(torch.sum(mask_self * mask_same * exp) /
        #     #                       torch.sum(mask_self * exp))
        #     # NCA loss does not have log
        #     # snn_loss -= (torch.sum(mask_self * mask_same * exp) /
        #     #              torch.sum(mask_self * exp))
        #     # additionally push diff
        #     snn_loss += const * torch.log(torch.sum((1 - mask_same) * exp) /
        #                                   torch.sum(mask_self * exp))
        # return snn_loss

        # Diff loss - push away different class
        # loss = torch.zeros(1).cuda()
        # for i in range(logits.size(0)):
        #     mask_diff = (label[i] != label).type(torch.float32)
        #     mask_self = torch.ones(logits.size(0)).cuda()
        #     mask_self[i] = 0
        #     dist = ((logits[i] - logits) ** 2).sum(1) * self.it.exp()
        #     # upper bound distance to prevent overflow
        #     # exp = torch.exp(- torch.min(dist, torch.tensor(50.).cuda()))
        #     # exp = torch.exp(- dist)
        #     # loss += torch.log(torch.sum(mask_diff * exp) /
        #     #                   torch.sum(mask_self * exp))
        #     # loss += torch.sum(mask_diff * exp) / torch.sum(mask_self * exp)
        #     thres = torch.tensor(4.).cuda()
        #     loss -= torch.sum(mask_diff * torch.min(dist, thres))
        # return loss

        # Diff loss - push away nearest one
        loss = torch.zeros(1).cuda()
        for i in range(logits.size(0)):
            mask_same = (label[i] == label).type(torch.float32)
            mask_self = torch.ones(logits.size(0)).cuda()
            mask_self[i] = 0
            mask_diff = (label[i] != label).type(torch.float32)
            dist = ((logits[i] - logits) ** 2).sum(1) * self.it.exp()
            # upper bound distance to prevent overflow
            # exp = torch.exp(- torch.min(dist, torch.tensor(50.).cuda()))
            # exp = torch.exp(- dist)
            # loss += torch.log(torch.sum(mask_diff * exp) /
            #                   torch.sum(mask_self * exp))
            # loss += (1e20 * mask_same + exp).min() / torch.sum(mask_self * exp)
            # loss -= torch.min(torch.min(1e20 * mask_same + dist),
            #                   torch.tensor(4.).cuda())
            loss -= torch.min(mask_diff * dist, torch.tensor(4.).cuda()).sum()
            # pull same
            const = 1.
            # loss += const * \
            #     torch.min(1e20 * (mask_diff + 1 - mask_self) + dist)
            loss += const * torch.sum(mask_same * dist)

        # Lipschitz loss
        # reg = torch.tensor(0.).cuda()
        # reg += infty_norm_reg(self.conv1.weight.reshape(64, 32))
        # reg += infty_norm_reg(self.conv2.weight.reshape(64 * 6 ** 2, 64))
        # reg += infty_norm_reg(self.conv3.weight.reshape(128 * 5 ** 2, 128))
        # reg += infty_norm_reg(self.fc1.weight)
        # reg += infty_norm_reg(self.fc2.weight)
        # loss += 1e-1 * reg

        return loss


class NCA_AE(nn.Module):

    def __init__(self, latent_dim=20, init_it=1, train_it=False, alpha=1e2):
        super(NCA_AE, self).__init__()
        self.alpha = alpha
        # V1
        # self.conv1 = NormConv2d(1, 64, kernel_size=8, stride=2, padding=3)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = NormConv2d(64, 128, kernel_size=6, stride=2, padding=3)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv3 = NormConv2d(128, 128, kernel_size=5, stride=1, padding=0)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.fc = NormLinear(2048, latent_dim)

        # V2
        # self.conv1 = NormConv2d(1, 64, kernel_size=8, stride=2, padding=3)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = NormConv2d(64, 128, kernel_size=6, stride=2, padding=3)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv3 = NormConv2d(128, 256, kernel_size=5, stride=1, padding=0)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.fc = NormLinear(4096, latent_dim)

        # V3 - two-sided ReLU
        self.conv1 = NormConv2d(1, 32, kernel_size=8, stride=2, padding=3)
        self.relu1 = TwoSidedReLU()
        self.conv2 = NormConv2d(64, 64, kernel_size=6, stride=2, padding=3)
        self.relu2 = TwoSidedReLU()
        self.conv3 = NormConv2d(128, 128, kernel_size=5, stride=1, padding=0)
        self.relu3 = TwoSidedReLU()
        self.fc1 = NormLinear(4096, 512)
        self.relu4 = TwoSidedReLU()
        self.fc2 = NormLinear(1024, latent_dim)

        self.de_fc = nn.Linear(latent_dim, 2048)
        self.deconv1 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=0)
        self.de_relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=0)
        self.de_relu2 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(64, 1, 5, stride=1, padding=0)
        self.de_sig = nn.Sigmoid()

        # initialize inverse temperature for each layer
        self.it = torch.nn.Parameter(
            data=torch.tensor(math.log(init_it)), requires_grad=train_it)

        for m in self.modules():
            if isinstance(m, NormConv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        # x = self.conv1(x)
        # x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.conv3(x)
        # x = self.relu3(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x

    def decode(self, z):
        x = self.de_fc(z)
        x = x.view(-1, 128, 4, 4)
        x = self.deconv1(x)
        x = self.de_relu1(x)
        x = self.deconv2(x)
        x = self.de_relu2(x)
        x = self.deconv3(x)
        x = self.de_sig(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon

    def loss_function(self, z, x_recon, x, label):
        """Calculate neighborhood loss"""

        batch_size = z.size(0)
        # Reconstruction loss
        recon_loss = ((x_recon - x)**2).view(batch_size, -1).sum()

        # SNN (or NCA) loss
        # snn_loss = torch.zeros(1).cuda()
        # for i in range(batch_size):
        #     mask_same = (label[i] == label).type(torch.float32)
        #     mask_self = torch.ones(batch_size).cuda()
        #     mask_self[i] = 0
        #     dist = ((z[i] - z) ** 2).sum(1) * self.it.exp()
        #     # upper bound distance to prevent overflow
        #     exp = torch.exp(- torch.min(dist, torch.tensor(50.).cuda()))
        #     # exp = torch.exp(- dist)
        #     # SNN loss has log
        #     snn_loss -= torch.log(torch.sum(mask_self * mask_same * exp) /
        #                           torch.sum(mask_self * exp))
        #     # NCA loss does not have log
        #     # snn_loss -= (torch.sum(mask_self * mask_same * exp) /
        #     #              torch.sum(mask_self * exp))
        # return recon_loss + self.alpha * snn_loss

        loss = torch.zeros(1).cuda()
        for i in range(batch_size):
            mask_same = (label[i] == label).type(torch.float32)
            mask_self = torch.ones(batch_size).cuda()
            mask_self[i] = 0
            dist = ((z[i] - z) ** 2).sum(1)
            # loss += (1e20 * mask_same + exp).min() / torch.sum(mask_self * exp)
            loss -= torch.min(torch.min(1e20 * mask_same + dist),
                              torch.tensor(4.).cuda())
            # additional regularization to pull same class
            const = 1e0
            # exp = torch.exp(- torch.min(dist * self.it.exp(),
            #                             torch.tensor(50.).cuda()))
            # loss -= const * torch.log(torch.sum(mask_same * mask_self * exp) /
            #                           torch.sum(mask_self * exp))
            # TODO
            mask_diff = (label[i] != label).type(torch.float32)
            loss += const * \
                torch.min(1e20 * (mask_diff + 1 - mask_self) + dist)

        return recon_loss + self.alpha * loss
