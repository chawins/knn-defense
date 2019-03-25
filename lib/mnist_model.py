'''MNIST models'''

import copy

import numpy as np
import torch.nn as nn

import faiss
from lib.faiss_utils import *


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


class DKNN(object):

    def __init__(self, model, x_train, y_train, x_cal, y_cal, layers, k=75,
                 num_classes=10):
        self.model = copy.deepcopy(model)
        self.y_train = y_train
        self.layers = layers
        self.k = k
        self.num_classes = num_classes
        self.indices = []
        self.activations = {}

        # register hook to get representations
        layer_count = 0
        for name, module in self.model.named_children():
            if name in layers:
                module.register_forward_hook(self._get_activation(name))
                layer_count += 1
        assert layer_count == len(layers)
        reps = self.get_activations(x_train)

        for layer in layers:
            rep = reps[layer].view(x_train.size(0), -1)
            # normalize activations so inner product is cosine similarity
            index = self._build_index(torch.renorm(rep, 2, 0, 1))
            self.indices.append(index)

        # set up calibration for credibility score
        y_pred = self.classify(x_cal)
        self.A = np.zeros((x_cal.size(0), )) + self.k * len(self.layers)
        for i, (y_c, y_p) in enumerate(zip(y_cal, y_pred)):
            self.A[i] -= y_p[y_c]

    def _get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook

    def _build_index(self, xb):

        d = xb.size(-1)
        # res = faiss.StandardGpuResources()
        # index = faiss.GpuIndexFlatIP(res, d)

        # brute-force
        # index = faiss.IndexFlatIP(d)

        # quantizer = faiss.IndexFlatL2(d)
        # index = faiss.IndexIVFFlat(quantizer, d, 100)
        # index.train(xb.cpu().numpy())

        # locality-sensitive hash
        index = faiss.IndexLSH(d, 256)

        index.add(xb.cpu().numpy())
        return index

    def get_activations(self, x):
        _ = self.model(x)
        return self.activations

    def get_neighbors(self, x):
        output = []
        reps = self.get_activations(x)
        for layer, index in zip(self.layers, self.indices):
            rep = reps[layer].cpu().numpy().reshape(x.size(0), -1)
            D, I = index.search(rep, self.k)
            # D, I = search_index_pytorch(index, reps[layer], k)
            # uncomment when using GPU
            # res.syncDefaultStreamCurrentDevice()
            output.append((D, I))
        return output

    def classify(self, x):
        nb = self.get_neighbors(x)
        class_counts = np.zeros((x.size(0), self.num_classes))
        for (_, I) in nb:
            y_pred = self.y_train.cpu().numpy()[I]
            for i in range(x.size(0)):
                class_counts[i] += np.bincount(y_pred[i], minlength=10)
        return class_counts

    def credibility(self, class_counts):
        alpha = self.k * len(self.layers) - np.max(class_counts, 1)
        cred = np.zeros_like(alpha)
        for i, a in enumerate(alpha):
            cred[i] = np.sum(self.A >= a)
        return cred / self.A.shape[0]
