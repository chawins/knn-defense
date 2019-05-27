
import numpy as np
import torch.nn.functional as F

import faiss
from lib.faiss_utils import *


class DKNN(object):

    def __init__(self, model, x_train, y_train, x_cal, y_cal, layers, k=75,
                 num_classes=10, device='cuda'):
        """
        device: device that model is on
        """
        # self.model = copy.deepcopy(model)
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.layers = layers
        self.k = k
        self.num_classes = num_classes
        self.device = device
        self.indices = []
        self.activations = {}

        # register hook to get representations
        layer_count = 0
        for name, module in self.model.named_children():
            if name in layers:
                module.register_forward_hook(self._get_activation(name))
                layer_count += 1
        assert layer_count == len(layers)
        reps = self.get_activations(x_train, requires_grad=False)

        for layer in layers:
            rep = F.normalize(reps[layer].cpu().view(
                x_train.size(0), -1), 2, 1)
            # normalize activations so inner product is cosine similarity
            index = self._build_index(rep)
            self.indices.append(index)

        # set up calibration for credibility score
        y_pred = self.classify(x_cal)
        self.A = np.zeros((x_cal.size(0), )) + self.k * len(self.layers)
        for i, (y_c, y_p) in enumerate(zip(y_cal, y_pred)):
            self.A[i] -= y_p[y_c]

    def _get_activation(self, name):
        def hook(model, input, output):
            # TODO: detach() is removed to get gradients
            self.activations[name] = output
        return hook

    def _build_index(self, xb):

        d = xb.size(-1)
        # res = faiss.StandardGpuResources()
        # index = faiss.GpuIndexFlatIP(res, d)

        # brute-force
        index = faiss.IndexFlatIP(d)

        # quantizer = faiss.IndexFlatL2(d)
        # index = faiss.IndexIVFFlat(quantizer, d, 100)
        # index.train(xb.cpu().numpy())

        # locality-sensitive hash
        # index = faiss.IndexLSH(d, 256)

        index.add(xb.detach().cpu().numpy())
        return index

    def get_activations(self, x, batch_size=500, requires_grad=True,
                        device=None):

        if device is None:
            device = self.device

        with torch.no_grad():
            num_total = x.size(0)
            num_batches = int(np.ceil(num_total / batch_size))
            activations = {}
            self.model(x[0:1].to(self.device))
            for layer in self.layers:
                size = self.activations[layer].size()
                activations[layer] = torch.empty((num_total, ) + size[1:],
                                                 dtype=torch.float32,
                                                 device=self.device,
                                                 requires_grad=False)
        with torch.set_grad_enabled(requires_grad):
            for i in range(num_batches):
                begin, end = i * batch_size, (i + 1) * batch_size
                self.model(x[begin:end].to(self.device))
                for layer in self.layers:
                    activations[layer][begin:end] = self.activations[layer]

        return activations

    def get_neighbors(self, x, k=None, layers=None):
        if k is None:
            k = self.k
        if layers is None:
            layers = self.layers
        output = []
        reps = self.get_activations(x, requires_grad=False)
        for layer, index in zip(self.layers, self.indices):
            if layer in layers:
                rep = F.normalize(reps[layer].view(x.size(0), -1), 2, 1)
                rep = rep.detach().cpu().numpy()
                D, I = index.search(rep, k)
                # D, I = search_index_pytorch(index, reps[layer], k)
                # uncomment when using GPU
                # res.syncDefaultStreamCurrentDevice()
                output.append((D, I))
        return output

    def classify(self, x):
        """return number of k-nearest neighbors in each class"""
        nb = self.get_neighbors(x)
        class_counts = np.zeros((x.size(0), self.num_classes))
        for (_, I) in nb:
            y_pred = self.y_train.cpu().numpy()[I]
            for i in range(x.size(0)):
                class_counts[i] += np.bincount(y_pred[i], minlength=10)
        return class_counts

    def classify_soft(self, x, layer=None, k=None):

        temp = 2e-2
        if layer is None:
            layer = self.layers[-1]
        if k is None:
            k = self.k
        with torch.no_grad():
            train_reps = self.get_activations(self.x_train)[layer]
            train_reps = F.normalize(
                train_reps.view(self.x_train.size(0), -1), 2, 1)
            reps = self.get_activations(x)[layer]
            reps = F.normalize(reps.view(x.size(0), -1), 2, 1)
            logits = torch.empty((x.size(0), self.num_classes))
            for i, rep in enumerate(reps):
                cos = ((rep.unsqueeze(0) * train_reps).sum(1) / temp).exp()
                # cos = (rep.unsqueeze(0) * train_reps).sum(1)
                for label in range(self.num_classes):
                    logits[i, label] = cos[self.y_train == label].mean()
                    # ind = self.y_train == label
                    # logits[i, label] = cos[ind].topk(k)[0].mean()
            return logits

    def credibility(self, class_counts):
        """compute credibility of samples given their class_counts"""
        alpha = self.k * len(self.layers) - np.max(class_counts, 1)
        cred = np.zeros_like(alpha)
        for i, a in enumerate(alpha):
            cred[i] = np.sum(self.A >= a)
        return cred / self.A.shape[0]


# =============================================================================


class DKNNL2(object):

    def __init__(self, model, x_train, y_train, x_cal, y_cal, layers, k=75,
                 num_classes=10, device='cuda'):
        """
        device: device that model is on
        """
        # self.model = copy.deepcopy(model)
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.layers = layers
        self.k = k
        self.num_classes = num_classes
        self.device = device
        self.indices = []
        self.activations = {}

        # register hook to get representations
        layer_count = 0
        for name, module in self.model.named_children():
            if name in layers:
                module.register_forward_hook(self._get_activation(name))
                layer_count += 1
        assert layer_count == len(layers)
        reps = self.get_activations(x_train, requires_grad=False)

        for layer in layers:
            rep = reps[layer].cpu().view(x_train.size(0), -1)
            # normalize activations so inner product is cosine similarity
            index = self._build_index(rep)
            self.indices.append(index)

        # set up calibration for credibility score
        y_pred = self.classify(x_cal)
        self.A = np.zeros((x_cal.size(0), )) + self.k * len(self.layers)
        for i, (y_c, y_p) in enumerate(zip(y_cal, y_pred)):
            self.A[i] -= y_p[y_c]

    def _get_activation(self, name):
        def hook(model, input, output):
            # TODO: detach() is removed to get gradients
            self.activations[name] = output
        return hook

    def _build_index(self, xb):

        d = xb.size(-1)
        # res = faiss.StandardGpuResources()
        # index = faiss.GpuIndexFlatIP(res, d)

        # brute-force
        index = faiss.IndexFlatL2(d)

        index.add(xb.detach().cpu().numpy())
        return index

    def get_activations(self, x, batch_size=500, requires_grad=True,
                        device=None):

        if device is None:
            device = self.device

        with torch.no_grad():
            num_total = x.size(0)
            num_batches = int(np.ceil(num_total / batch_size))
            activations = {}
            self.model(x[0:1].to(self.device))
            for layer in self.layers:
                size = self.activations[layer].size()
                activations[layer] = torch.empty((num_total, ) + size[1:],
                                                 dtype=torch.float32,
                                                 device=device,
                                                 requires_grad=False)

        with torch.set_grad_enabled(requires_grad):
            for i in range(num_batches):
                begin, end = i * batch_size, (i + 1) * batch_size
                self.model(x[begin:end].to(self.device))
                for layer in self.layers:
                    activations[layer][begin:end] = self.activations[layer]
            return activations

    def get_neighbors(self, x, k=None, layers=None):
        if k is None:
            k = self.k
        if layers is None:
            layers = self.layers
        output = []
        reps = self.get_activations(x, requires_grad=False)
        for layer, index in zip(self.layers, self.indices):
            if layer in layers:
                rep = reps[layer].view(x.size(0), -1)
                rep = rep.detach().cpu().numpy()
                D, I = index.search(rep, k)
                # D, I = search_index_pytorch(index, reps[layer], k)
                # uncomment when using GPU
                # res.syncDefaultStreamCurrentDevice()
                output.append((D, I))
        return output

    def classify(self, x):
        """return number of k-nearest neighbors in each class"""
        nb = self.get_neighbors(x)
        class_counts = np.zeros((x.size(0), self.num_classes))
        for (_, I) in nb:
            y_pred = self.y_train.cpu().numpy()[I]
            for i in range(x.size(0)):
                class_counts[i] += np.bincount(y_pred[i], minlength=10)
        return class_counts

    def predict(self, x):
        """predict label of single sample x"""
        return self.classify(x.unsqueeze(0))[0].argmax()

    def classify_soft(self, x, layer=None, k=None):

        temp = 2e-2
        if layer is None:
            layer = self.layers[-1]
        if k is None:
            k = self.k
        with torch.no_grad():
            train_reps = self.get_activations(self.x_train)[layer]
            train_reps = train_reps.view(self.x_train.size(0), -1)
            reps = self.get_activations(x)[layer]
            reps = reps.view(x.size(0), -1)
            logits = torch.empty((x.size(0), self.num_classes))
            for i, rep in enumerate(reps):
                dist = (((rep.view(1, -1) - train_reps)**2).sum(1) / temp).exp()
                # cos = ((rep.unsqueeze(0) * train_reps).sum(1) / temp).exp()
                # cos = (rep.unsqueeze(0) * train_reps).sum(1)
                for label in range(self.num_classes):
                    logits[i, label] = dist[self.y_train == label].mean()
                    # ind = self.y_train == label
                    # logits[i, label] = cos[ind].topk(k)[0].mean()
            return logits

    def credibility(self, class_counts):
        """compute credibility of samples given their class_counts"""
        alpha = self.k * len(self.layers) - np.max(class_counts, 1)
        cred = np.zeros_like(alpha)
        for i, a in enumerate(alpha):
            cred[i] = np.sum(self.A >= a)
        return cred / self.A.shape[0]

    def find_nn_diff_class(self, x, label):
        """find nearest neighbors of different class to x"""
        nn = np.zeros(x.size(0))
        for i in range(x.size(0)):
            found_diff_class = False
            k = 1e2
            # find k nearest neighbors at a time, keep increasing k until at
            # least one sample of a different class is found
            while not found_diff_class:
                _, I = self.get_neighbors(x[i].unsqueeze(0), k=int(k))[0]
                I = I[0]
                ind = np.where(label[i] != self.y_train[I])[0]
                if len(ind) != 0:
                    nn[i] = I[ind[0]]
                    found_diff_class = True
                else:
                    k *= 10

        return nn


# =============================================================================


class DKNNL2Approx(object):
    """
    Use HNSW index instead of an exact calculation. Only use this with k = 1,
    and do not use the distance.
    """

    def __init__(self, model, x_train, y_train, x_cal, y_cal, layers, k=75,
                 num_classes=10, device='cuda'):
        """
        device: device that model is on
        """
        # self.model = copy.deepcopy(model)
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.layers = layers
        self.k = k
        self.num_classes = num_classes
        self.device = device
        self.indices = []
        self.activations = {}

        # register hook to get representations
        layer_count = 0
        for name, module in self.model.named_children():
            if name in layers:
                module.register_forward_hook(self._get_activation(name))
                layer_count += 1
        assert layer_count == len(layers)
        reps = self.get_activations(x_train, requires_grad=False)

        for layer in layers:
            rep = reps[layer].cpu().view(x_train.size(0), -1)
            # normalize activations so inner product is cosine similarity
            index = self._build_index(rep)
            self.indices.append(index)

        # set up calibration for credibility score
        y_pred = self.classify(x_cal)
        self.A = np.zeros((x_cal.size(0), )) + self.k * len(self.layers)
        for i, (y_c, y_p) in enumerate(zip(y_cal, y_pred)):
            self.A[i] -= y_p[y_c]

    def _get_activation(self, name):
        def hook(model, input, output):
            # TODO: detach() is removed to get gradients
            self.activations[name] = output
        return hook

    def _build_index(self, xb):

        d = xb.size(-1)
        index = faiss.IndexHNSWFlat(d, 32)
        index.add(xb.detach().cpu().numpy())
        return index

    def get_activations(self, x, batch_size=500, requires_grad=True,
                        device=None):

        if device is None:
            device = self.device

        with torch.no_grad():
            num_total = x.size(0)
            num_batches = int(np.ceil(num_total / batch_size))
            activations = {}
            self.model(x[0:1].to(self.device))
            for layer in self.layers:
                size = self.activations[layer].size()
                activations[layer] = torch.empty((num_total, ) + size[1:],
                                                 dtype=torch.float32,
                                                 device=device,
                                                 requires_grad=False)

        with torch.set_grad_enabled(requires_grad):
            for i in range(num_batches):
                begin, end = i * batch_size, (i + 1) * batch_size
                self.model(x[begin:end].to(self.device))
                for layer in self.layers:
                    activations[layer][begin:end] = self.activations[layer]
            return activations

    def get_neighbors(self, x, k=None, layers=None):
        if k is None:
            k = self.k
        if layers is None:
            layers = self.layers
        output = []
        reps = self.get_activations(x, requires_grad=False)
        for layer, index in zip(self.layers, self.indices):
            if layer in layers:
                rep = reps[layer].view(x.size(0), -1)
                rep = rep.detach().cpu().numpy()
                D, I = index.search(rep, k)
                # D, I = search_index_pytorch(index, reps[layer], k)
                # uncomment when using GPU
                # res.syncDefaultStreamCurrentDevice()
                output.append((D, I))
        return output

    def classify(self, x):
        """return number of k-nearest neighbors in each class"""
        nb = self.get_neighbors(x)
        class_counts = np.zeros((x.size(0), self.num_classes))
        for (_, I) in nb:
            y_pred = self.y_train.cpu().numpy()[I]
            for i in range(x.size(0)):
                class_counts[i] += np.bincount(y_pred[i], minlength=10)
        return class_counts

    def classify_soft(self, x, layer=None, k=None):

        temp = 2e-2
        if layer is None:
            layer = self.layers[-1]
        if k is None:
            k = self.k
        with torch.no_grad():
            train_reps = self.get_activations(self.x_train)[layer]
            train_reps = train_reps.view(self.x_train.size(0), -1)
            reps = self.get_activations(x)[layer]
            reps = reps.view(x.size(0), -1)
            logits = torch.empty((x.size(0), self.num_classes))
            for i, rep in enumerate(reps):
                dist = (((rep.view(1, -1) - train_reps)**2).sum(1) / temp).exp()
                # cos = ((rep.unsqueeze(0) * train_reps).sum(1) / temp).exp()
                # cos = (rep.unsqueeze(0) * train_reps).sum(1)
                for label in range(self.num_classes):
                    logits[i, label] = dist[self.y_train == label].mean()
                    # ind = self.y_train == label
                    # logits[i, label] = cos[ind].topk(k)[0].mean()
            return logits

    def credibility(self, class_counts):
        """compute credibility of samples given their class_counts"""
        alpha = self.k * len(self.layers) - np.max(class_counts, 1)
        cred = np.zeros_like(alpha)
        for i, a in enumerate(alpha):
            cred[i] = np.sum(self.A >= a)
        return cred / self.A.shape[0]
