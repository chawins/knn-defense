import numpy as np
import torch

import faiss


class KNNL2(object):

    def __init__(self, x_train, y_train, x_cal, y_cal, k=75, num_classes=10):
        """
        """
        # self.model = copy.deepcopy(model)
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self.num_classes = num_classes

        self.index = self._build_index(x_train.view(x_train.size(0), -1))

        # set up calibration for credibility score
        y_pred = self.classify(x_cal)
        self.A = np.zeros((x_cal.size(0), )) + self.k
        for i, (y_c, y_p) in enumerate(zip(y_cal, y_pred)):
            self.A[i] -= y_p[y_c]

    def _build_index(self, xb):

        d = xb.size(-1)
        # res = faiss.StandardGpuResources()
        # index = faiss.GpuIndexFlatIP(res, d)

        # brute-force
        index = faiss.IndexFlatL2(d)

        index.add(xb.detach().cpu().numpy())
        return index

    def get_neighbors(self, x, k=None):
        if k is None:
            k = self.k
        # D, I = search_index_pytorch(index, reps[layer], k)
        # uncomment when using GPU
        # res.syncDefaultStreamCurrentDevice()
        return self.index.search(x.view(x.size(0), -1).numpy(), k)

    def classify(self, x):
        """return number of k-nearest neighbors in each class"""
        _, I = self.get_neighbors(x)
        class_counts = np.zeros((x.size(0), self.num_classes))
        y_pred = self.y_train.cpu().numpy()[I]
        for i in range(x.size(0)):
            class_counts[i] = np.bincount(
                y_pred[i], minlength=self.num_classes)
        return class_counts

    def credibility(self, class_counts):
        """compute credibility of samples given their class_counts"""
        alpha = self.k - np.max(class_counts, 1)
        cred = np.zeros_like(alpha)
        for i, a in enumerate(alpha):
            cred[i] = np.sum(self.A >= a)
        return cred / self.A.shape[0]

    def find_nn_diff_class(self, x, label):
        """find nearest neighbors of different class to x"""
        x_nn = torch.zeros_like(x)
        for i in range(x.size(0)):
            found_diff_class = False
            k = 1e2
            # find k nearest neighbors at a time, keep increasing k until at
            # least one sample of a different class is found
            while not found_diff_class:
                _, I = self.get_neighbors(x[i].unsqueeze(0), k=int(k))
                I = I[0]
                ind = np.where(label[i] != self.y_train[I])[0]
                if len(ind) != 0:
                    x_nn[i] = self.x_train[I[ind[0]]]
                    found_diff_class = True
                else:
                    k *= 10

        return x_nn

    def get_min_dist(self, x, label, x_target, iterations=10):
        """find attack on 1-NN with minimal distance by doing a line search"""
        x_adv = x.clone()
        diff = x_target - x
        upper_bound = torch.zeros((x.size(0), ) + (x.dim() - 1) * (1, )) + 1
        lower_bound = torch.zeros_like(upper_bound)
        for _ in range(iterations):
            mid = (upper_bound + lower_bound) / 2
            x_cur = x + diff * mid
            y_pred = self.classify(x_cur).argmax(1)
            for i in range(x.size(0)):
                if y_pred[i] == label[i]:
                    lower_bound[i] = mid[i]
                else:
                    x_adv[i] = x_cur[i]
                    upper_bound[i] = mid[i]

        return x_adv
