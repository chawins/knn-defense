import numpy as np
import torch

import faiss
from lib.faiss_utils import *


class KNNL2(object):

    def __init__(self, x_train, y_train, x_cal, y_cal, k=75, num_classes=10):
        """
        """
        # self.model = copy.deepcopy(model)
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self.num_classes = num_classes

        # self.index = self._build_index(x_train.view(x_train.size(0), -1))

        # set up calibration for credibility score
        # y_pred = self.classify(x_cal)
        # self.A = np.zeros((x_cal.size(0), )) + self.k
        # for i, (y_c, y_p) in enumerate(zip(y_cal, y_pred)):
        #     self.A[i] -= y_p[y_c]

    def _build_index(self, xb):

        d = xb.size(-1)
        self.res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(self.res, d)

        # brute-force
        # index = faiss.IndexFlatL2(d)

        index.add(xb.detach().cpu().numpy())
        # index.add(xb)
        return index

    # def get_neighbors(self, x, k=None):
    #     if k is None:
    #         k = self.k
    #     # return self.index.search(x.view(x.size(0), -1).numpy(), k)
    #     D, I = search_index_pytorch(self.index, x.view(x.size(0), -1), k)
    #     # uncomment when using GPU
    #     self.res.syncDefaultStreamCurrentDevice()
    #     return D, I

    def get_neighbors(self, x, k=None):
        if k is None:
            k = self.k
        with torch.no_grad():
            D = torch.zeros((x.size(0), k))
            I = torch.zeros((x.size(0), k), dtype=torch.int64)
            for i, xx in enumerate(x):
                d = (self.x_train - xx).view(self.x_train.size(0), -1).norm(2, 1)
                ind = d.topk(k, largest=False)
                I[i] = ind[1]
                D[i] = ind[0]
            return D, I

    def classify(self, x):
        """return number of k-nearest neighbors in each class"""
        _, I = self.get_neighbors(x)
        class_counts = np.zeros((x.size(0), self.num_classes))
        # this line is for CPU faiss
        # y_pred = self.y_train.cpu().numpy()[I]
        # this line is for GPU faiss
        y_pred = self.y_train[I]
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

    def opt_attack(self, x, label, iterations=10):
        """
        Find optimal attack for 1-NN. Worst case complexity is O(kN^2)
        """

        x_adv = x.clone()
        for i, xx in enumerate(x):
            # get index of current nearest neighbor
            _, cur_ind = self.get_neighbors(xx.unsqueeze(0), k=1)
            cur_ind = cur_ind[0, 0]
            # skip if cur_ind is already misclassified
            if self.y_train[cur_ind] != label[i]:
                continue
            # iterate through all possible target indices
            tar_ind_all = np.where(self.y_train.numpy() != label.numpy()[i])[0]
            # initialize best_dist to be distance to nearest sample of
            # different class
            best_dist = (xx - self.x_train[tar_ind_all]).view(
                len(tar_ind_all), -1).norm(2, 1).min()

            for tar_ind in tar_ind_all:
                out = self.find_edge(xx, cur_ind, tar_ind, best_dist,
                                     iterations=iterations)
                if out is not None:
                    x_adv[i] = out[0]
                    best_dist = out[1]
        return x_adv

    def find_edge(self, x, cur_ind, tar_ind, best_dist, iterations=10):
        """
        Move x towards x_target in the direction orthogonal to its Voronoi edge
        which is the same direction from x_cur to x_target.
        """

        x_cur, x_tar = self.x_train[cur_ind], self.x_train[tar_ind]
        og_dir = (x_tar - x_cur) / (x_tar - x_cur).view(-1).norm()

        # Try with best_dist first, terminate if fails
        x_new = x + best_dist * og_dir
        _, nn = self.get_neighbors(x_new.unsqueeze(0), k=1)
        if nn[0, 0] != tar_ind:
            return None

        x_adv = x_new
        upper_bound = best_dist
        lower_bound = 0
        for _ in range(iterations):
            mid = (upper_bound + lower_bound) / 2
            x_new = x + og_dir * mid
            _, nn = self.get_neighbors(x_new.unsqueeze(0), k=1)
            if nn[0, 0] == tar_ind:
                upper_bound = mid
                best_dist = mid
                x_adv = x_new
            elif nn[0, 0] == cur_ind:
                lower_bound = mid
            else:
                # terminates if the third index (not cur_ind and not tar_ind)
                # shows up. This means there is a closer cell.
                return None

        return x_adv, best_dist

    def get_margin_bound(self, x, label, x_target):

        _, nn = self.get_neighbors(x, k=1)
        nn = nn.view(x.size(0))
        # nn = nn.reshape(x.size(0))
        ind = np.where(self.y_train[nn].numpy() == label.numpy())[0]
        x_nn = self.x_train[nn]
        dist_nn = (x - x_nn).view(x.size(0), -1).norm(2, 1)
        dist_target = (x - x_target).view(x.size(0), -1).norm(2, 1)
        return dist_target - dist_nn, ind

    def get_min_dist(self, x, label, x_target, iterations=10):
        """
        (DEPRECATED)
        find attack on 1-NN with minimal distance by doing a line search

        WARNING: this is deprecated and incorrect for computing the optimal
        adversarial examples for 1-NN. See opt_attack instead.
        """
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


# ============================================================================ #


class KNNL2NP(object):

    def __init__(self, x_train, y_train, x_cal, y_cal, k=75, num_classes=10):
        """
        This is Numpy version of KNNL2.

        x_train, y_train, x_cal, y_cal must be Numpy arrays.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self.num_classes = num_classes

        self.index = self._build_index(x_train.reshape(x_train.shape[0], -1))

        # set up calibration for credibility score
        # y_pred = self.classify(x_cal)
        # self.A = np.zeros((x_cal.shape[0], )) + self.k
        # for i, (y_c, y_p) in enumerate(zip(y_cal, y_pred)):
        #     self.A[i] -= y_p[y_c]

    def _build_index(self, xb):

        d = xb.shape[-1]
        # res = faiss.StandardGpuResources()
        # index = faiss.GpuIndexFlatIP(res, d)

        # brute-force
        # index = faiss.IndexFlatL2(d)

        # HNSW
        index = faiss.IndexHNSWFlat(d, 32)

        index.add(xb)
        return index

    def get_neighbors(self, x, k=None):
        if k is None:
            k = self.k
        # D, I = search_index_pytorch(index, reps[layer], k)
        # uncomment when using GPU
        # res.syncDefaultStreamCurrentDevice()
        return self.index.search(x.reshape(x.shape[0], -1), k)

    def classify(self, x):
        """return number of k-nearest neighbors in each class"""
        _, I = self.get_neighbors(x)
        class_counts = np.zeros((x.shape[0], self.num_classes))
        y_pred = self.y_train[I]
        for i in range(x.shape[0]):
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
        # x_nn = torch.zeros_like(x)
        x_nn = np.zeros_like(x)
        for i in range(x.shape[0]):
            found_diff_class = False
            k = 1e2
            # find k nearest neighbors at a time, keep increasing k until at
            # least one sample of a different class is found
            while not found_diff_class:
                _, I = self.get_neighbors(x[i][np.newaxis], k=int(k))
                I = I[0]
                ind = np.where(label[i] != self.y_train[I])[0]
                if len(ind) != 0:
                    x_nn[i] = self.x_train[I[ind[0]]]
                    found_diff_class = True
                else:
                    k *= 10

        return x_nn

    def opt_attack(self, x, label, pert_bound=2, iterations=10):
        """
        Find optimal attack for 1-NN. Worst case complexity is O(kN^2)
        """

        x_adv = x.copy()
        cur_D, cur_I = self.get_neighbors(x, k=1)

        for i, xx in enumerate(x):
            print(i)
            # get index of current nearest neighbor
            cur_d, cur_i = cur_D[i, 0], cur_I[i, 0]
            # skip if cur_ind is already misclassified
            if self.y_train[cur_i] != label[i]:
                continue
            # iterate through all possible target indices
            tar_ind_all = np.where(self.y_train != label[i])[0]
            # calculate distance to all targets
            dist = np.linalg.norm(
                (xx - self.x_train[tar_ind_all]).reshape(len(tar_ind_all), -1),
                axis=1)
            # only check samples with d <= 2 * pert_bound + cur_d
            filter_ind = np.where(dist <= 2 * pert_bound + cur_d)[0]
            # initialize best_dist to be distance to nearest sample of
            # different class
            best_dist = dist[filter_ind].min()
            print('len %d' % len(tar_ind_all[filter_ind]))

            for tar_ind in tar_ind_all[filter_ind]:
                out = self.find_edge(xx, cur_i, tar_ind, best_dist,
                                     iterations=iterations)
                if out is not None:
                    x_adv[i] = out[0]
                    best_dist = out[1]
        return x_adv

    def find_edge(self, x, cur_ind, tar_ind, best_dist, iterations=10):
        """
        Move x towards x_target in the direction orthogonal to its Voronoi edge
        which is the same direction from x_cur to x_target.
        """

        x_cur, x_tar = self.x_train[cur_ind], self.x_train[tar_ind]
        # this direction is only optimal if the cells are adjacent
        og_dir = (x_tar - x_cur) / np.linalg.norm(x_tar - x_cur)

        # Try with best_dist first, terminate if fails
        x_new = x + best_dist * og_dir
        _, nn = self.get_neighbors(x_new[np.newaxis], k=1)
        if nn[0, 0] == cur_ind:
            return None

        x_adv = x_new
        upper_bound = best_dist
        lower_bound = 0
        for _ in range(iterations):
            mid = (upper_bound + lower_bound) / 2
            x_new = x + og_dir * mid
            _, nn = self.get_neighbors(x_new[np.newaxis], k=1)
            if nn[0, 0] == tar_ind:
                upper_bound = mid
                best_dist = mid
                x_adv = x_new
            else:
                lower_bound = mid

        return x_adv, best_dist

    def get_margin_bound(self, x, label, x_target):

        _, nn = self.get_neighbors(x, k=1)
        nn = nn.reshape(x.shape[0])
        # nn = nn.reshape(x.size(0))
        ind = np.where(self.y_train[nn].numpy() == label.numpy())[0]
        x_nn = self.x_train[nn]
        dist_nn = np.linalg.norm((x - x_nn).reshape(x.shape[0], -1), axis=1)
        dist_target = np.linalg.norm(
            (x - x_target).reshape(x.shape[0], -1), axis=1)
        return dist_target - dist_nn, ind

    def get_min_dist(self, x, label, x_target, iterations=10):
        """
        find attack on 1-NN with minimal distance by doing a line search

        WARNING: this is deprecated and incorrect for computing the optimal
        adversarial examples for 1-NN. See opt_attack instead.
        """
        x_adv = np.copy(x)
        diff = x_target - x
        upper_bound = np.zeros(
            (x.shape[0], ) + (x.ndim - 1) * (1, ), dtype=np.float32) + 1
        lower_bound = np.zeros_like(upper_bound)
        for _ in range(iterations):
            mid = (upper_bound + lower_bound) / 2
            x_cur = x + diff * mid
            y_pred = self.classify(x_cur).argmax(1)
            for i in range(x.shape[0]):
                if y_pred[i] == label[i]:
                    lower_bound[i] = mid[i]
                else:
                    x_adv[i] = x_cur[i]
                    upper_bound[i] = mid[i]

        return x_adv
