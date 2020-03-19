'''Implement gradient-based attack on DkNN and kNN (version 2)'''

import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

INFTY = 1e20


class DKNNAttackV2(object):
    """
    Implement gradient-based attack on k-Nearest Neigbhor and its neural
    network based variants.

    Reference:
    Minimum-Norm Adversarial Examples on KNN and KNN-Based Models
    (Chawin Sitawarin, David Wagner)
    https://arxiv.org/abs/2003.06559
    """

    def __init__(self, dknn):
        self.dknn = dknn
        self.device = dknn.device
        self.layers = dknn.layers
        self.guide_reps = {}
        self.thres = None
        self.coeff = None

        # classify x_train in dknn (leave-one-out)
        out = self.dknn.classify(dknn.x_train, k=(dknn.k + 1))
        eye = np.eye(dknn.num_classes)
        labels = eye[dknn.y_train]
        self.y_pred = (out - labels).argmax(1)

    def __call__(self, x_orig, label, norm, guide_layer=['relu1'], m=100,
                 init_mode=1, init_mode_k=1, binary_search_steps=5,
                 max_iterations=500, learning_rate=1e-2, initial_const=1,
                 max_linf=None, random_start=False, thres_steps=100,
                 check_adv_steps=100, verbose=True):
        """
        Parameters
        ----------
        dknn : DKNN object
            DkNN (defined in lib/dknn.py) that we want to attack.
        x_orig : torch.tensor
            tensor of the original samples to attack. Does not need to require
            gradients, shape is (num_samples, ) + input_shape.
        label : torch.tensor
            tensor of the label corresponding to x_orig.
        norm : (2 or np.inf)
            norm of adversarial perturbation.
        guide_layer : str, optional
            layer name in which we want to find guide samples. Default is
            'relu1'.
        m : int, optional
            number of guide samples. Default is 100
        init_mode : (1 or 2), optional
            1 : initialize attack at <x_orig>.
            2 : initialize attack at k-th neighbor of <x_orig> that is
                not classified as <label>. k is specified by <init_mode_k>.
            Default is 1.
        init_mode_k : int, optional
            specify k when init_mode is set to 2. Default is 1.
        binary_search_step : int, optional
            number of steps for binary search on the norm penalty constant.
            Default is 5.
        max_iterations : int, optional
            number of optimization steps (per one binary search). Default is
            500.
        learning_rate : float , optional
            step size or learning rate for the optimizer. Default is 1e-2.
        initial_const : float, optional
            a number the norm penalty constant should be initialized to.
            Default is 1.
        max_linf : float, optional
            use to bound the L-inf norm of the attacks (addition to L-2 norm
            penalty). Set to None to not use this option. Default is None.
        random_start : bool, optional
            whether or not to initialize the perturbation with small isotropic
            Gaussian noise. Default is False.
        thres_steps : int, optional
            specify number of optimization steps to dynamically recalculate
            threshold and guide samples. Picking a small number makes the
            attack slower but more accurate. Default is 100.
        check_adv_steps : int, optional
            specify number of optimization steps to check if the perturbed
            samples are misclassified and save them if they also have the
            smallest perturbation seen so far. Default is 100.
        verbose : bool, optional
            whether or not to print progress. Default is True.

        Returns
        -------
        x_adv : torch.tensor
            adversarial examples found. If adversarial examples for some inputs
            are not found, return those inputs.
        """

        # min_, max_ = x_orig.min(), x_orig.max()
        min_ = torch.tensor(0., device=self.device)
        max_ = torch.tensor(1., device=self.device)
        if max_linf is not None:
            min_ = torch.max(x_orig - max_linf, min_)
            max_ = torch.min(x_orig + max_linf, max_)
        batch_size = x_orig.size(0)
        x_adv = x_orig.clone()
        label = label.cpu().numpy()
        input_shape = x_orig.detach().cpu().numpy().shape
        # initialize coeff for guide samples
        self.coeff = torch.zeros((x_orig.size(0), m), device=self.device)
        self.coeff[:, :m // 2] -= 1
        self.coeff[:, m // 2:] += 1

        def to_attack_space(x):
            # map from [min_, max_] to [-1, +1]
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            x = (x - a) / b
            # from [-1, +1] to approx. (-1, +1)
            x = x * 0.999999
            # from (-1, +1) to (-inf, +inf)
            return self.atanh(x)

        def to_model_space(x):
            """Transforms an input from the attack space to the model space.
            This transformation and the returned gradient are elementwise."""
            # from (-inf, +inf) to (-1, +1)
            x = torch.tanh(x)
            # map from (-1, +1) to (min_, max_)
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            x = x * b + a
            return x

        # variables representing inputs in attack space will be prefixed with z
        z_orig = to_attack_space(x_orig)
        x_recon = to_model_space(z_orig)

        # declare tensors that keep track of constants and binary search
        const = torch.zeros((batch_size, ), device=self.device)
        const += initial_const
        lower_bound = torch.zeros_like(const)
        upper_bound = torch.zeros_like(const) + INFTY
        best_dist = torch.zeros_like(const) + INFTY

        if init_mode == 1:
            if verbose:
                print('Using init_mode 1: initialize at original input <x_orig>.')
        elif init_mode == 2:
            if verbose:
                print('Using init_mode 2: initialize at k-th neighbor of ' +
                      'input <x_orig> that is not classified as <label>.')
            with torch.no_grad():
                # search for nearest neighbor of incorrect class
                x_init = self.find_kth_neighbor_diff_class(
                    x_orig, label, init_mode_k)
                z_init = to_attack_space(x_init.to('cuda')) - z_orig

        # make a list of number of guide samples that linearly decreases
        start = (self.dknn.k + 1) // 2
        end = max(m // 2, start + 1)
        m_list = np.arange(start, end, (end - start) / binary_search_steps)

        for binary_search_step in range(binary_search_steps):

            # reduce number of guide samples for successful attacks
            idx_m = binary_search_steps - binary_search_step - 1
            m_new = np.ceil(m_list[idx_m]).astype(np.int32)

            # initialize perturbation in transformed space
            if not random_start:
                z_delta = torch.zeros_like(z_orig, requires_grad=True)
            else:
                rand = np.random.randn(*input_shape) * 1e-2
                z_delta = torch.tensor(
                    rand, dtype=torch.float32, requires_grad=True,
                    device=self.device)
            with torch.no_grad():
                if init_mode == 2:
                    z_delta += z_init

            # create a new optimizer
            optimizer = optim.RMSprop([z_delta], lr=learning_rate)

            for iteration in range(max_iterations):
                optimizer.zero_grad()
                x = to_model_space(z_orig + z_delta)

                # adaptively choose threshold and guide samples every
                # <thres_steps> iterations
                with torch.no_grad():
                    if iteration % thres_steps == 0:
                        # thres = self.dknn.get_neighbors(x)[0][0][:, -1]
                        # self.thres = torch.tensor(thres).to(self.device).view(
                        #     batch_size, 1)
                        self.thres = []
                        thres = self.dknn.get_neighbors(x)
                        for i in range(len(self.layers)):
                            t = torch.tensor(thres[i][0][:, -1]).to(
                                self.device).unsqueeze(-1)
                            self.thres.append(t)
                        self.find_guide_samples(
                            x, label, m=m, layers=guide_layer)

                reps = self.dknn.get_activations(x, requires_grad=True)
                loss, dist = self.loss_function(x, reps, const, x_recon, norm)
                loss.backward()
                optimizer.step()

                if (verbose and iteration %
                        (np.ceil(max_iterations / 10)) == 0):
                    print('    step: %d; loss: %.3f; dist: %.3f' %
                          (iteration, loss.cpu().detach().numpy(),
                           dist.mean().cpu().detach().numpy()))

                # every <check_adv_steps>, save adversarial samples
                # with minimal perturbation
                if ((iteration + 1) % check_adv_steps == 0 or
                        iteration == max_iterations):
                    is_adv = self.check_adv(x, label)
                    for i in range(batch_size):
                        if is_adv[i] and best_dist[i] > dist[i]:
                            x_adv[i] = x[i]
                            best_dist[i] = dist[i]

            # check how many attacks have succeeded
            with torch.no_grad():
                is_adv = self.check_adv(x, label)
                if verbose:
                    print('binary step: %d; num successful adv: %d/%d' %
                          (binary_search_step, is_adv.sum(), batch_size))

            for i in range(batch_size):
                # set new upper and lower bounds
                if is_adv[i]:
                    upper_bound[i] = const[i]
                    self.coeff[i, m_new:m // 2] = 0
                    self.coeff[i, m // 2 + m_new:] = 0
                else:
                    lower_bound[i] = const[i]
                # set new const
                if upper_bound[i] == INFTY:
                    # exponential search if adv has not been found
                    const[i] *= 10
                elif lower_bound[i] == 0:
                    const[i] /= 10
                else:
                    # binary search if adv has been found
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2
                # only keep adv with smallest l2dist
                if is_adv[i] and best_dist[i] > dist[i]:
                    x_adv[i] = x[i]
                    best_dist[i] = dist[i]

            # check the current attack success rate (combined with previous
            # binary search steps)
            if verbose:
                with torch.no_grad():
                    is_adv = self.check_adv(x_adv, label)
                    print('binary step: %d; num successful adv so far: %d/%d' %
                          (binary_search_step, is_adv.sum(), batch_size))

        return x_adv

    def check_adv(self, x, label):
        """Check if label of <x> predicted by <dknn> matches with <label>"""
        y_pred = self.dknn.classify(x).argmax(1)
        # y_pred = self.dknn.classify_soft(x).argmax(1)
        return torch.tensor((y_pred != label).astype(np.float32)).to(self.device)
        # class_counts = self.dknn.classify(x)
        # y_pred = class_counts.argmax(1)
        # eye = np.eye(len(class_counts[0]))
        # pred_oh = eye[y_pred]
        # max_counts = (pred_oh * class_counts).sum(1)
        # return (max_counts >= 295) & (y_pred != label)

    def loss_function(self, x, reps, const, x_recon, norm):
        """Returns the loss averaged over the batch (first dimension of x) and
        norm squared of the perturbation
        """

        batch_size = x.size(0)
        # in case you want to compute the loss on all layers, use code below
        adv_loss = torch.zeros(
            (batch_size, len(self.layers)), device=self.device)
        # find squared L-2 distance between original samples and their
        # adversarial examples at each layer
        for l, layer in enumerate(self.layers):
            rep = reps[layer].view(batch_size, 1, -1)
            dist = ((rep - self.guide_reps[layer]) ** 2).sum(2)
            fx = dist - self.thres[l]
            adv_loss[:, l] = F.relu(
                self.coeff.to(self.device) * fx + 1e-5).sum(1)
        adv_loss = adv_loss.sum(1)

        # find L-2 norm squared of perturbation
        if norm == 2:
            dist = ((x - x_recon).view(batch_size, -1) ** 2).sum(1)
            # total_loss is sum of perturbation norm and squared distance
            # of representations, multiplied by constant
            total_loss = dist + const * adv_loss
            return total_loss.mean(), dist.sqrt()
        elif norm == np.inf:
            # (1) penalize l-inf directly
            dist = (x - x_recon).view(batch_size, -1).abs().max(1)[0]
            total_loss = dist + const * adv_loss
            return total_loss.mean(), dist
        else:
            raise ValueError('Norm not implemented (only l2 and l-inf)')

    def find_guide_samples(self, x, label, m=100, layers=['relu1']):
        """Find k nearest neighbors to <x> that all have the same class but not
        equal to <label>
        """
        num_classes = self.dknn.num_classes
        x_train = self.dknn.x_train
        y_train = self.dknn.y_train
        batch_size = x.size(0)
        nn = torch.zeros((m, ) + x.size())
        nb = self.dknn.get_neighbors(x, k=x_train.size(0), layers=layers)

        # find guide samples from the first layer
        D, I = nb[0]
        for i, (d, ind) in enumerate(zip(D, I)):
            mean_dist = np.zeros((num_classes, ))
            for j in range(num_classes):
                mean_dist[j] = np.mean(
                    d[np.where(y_train[ind] == j)[0]][:m // 2])
            mean_dist[label[i]] += INFTY
            nearest_label = mean_dist.argmin()
            nn_ind = np.where(y_train[ind] == nearest_label)[0][:m // 2]
            nn[m // 2:, i] = x_train[ind[nn_ind]]
            nn_ind = np.where(y_train[ind] == label[i])[0][:m // 2]
            nn[:m // 2, i] = x_train[ind[nn_ind]]

        # initialize self.guide_reps if empty
        if not self.guide_reps:
            guide_rep = self.dknn.get_activations(
                nn[:, 0], requires_grad=False)
            for layer in layers:
                # set a zero tensor before filling it
                size = (batch_size, ) + guide_rep[layer].size()
                self.guide_reps[layer] = torch.zeros(size, device=self.device)

        # fill self.guide_reps
        for i in range(batch_size):
            guide_rep = self.dknn.get_activations(
                nn[:, i], requires_grad=False)
            for layer in layers:
                self.guide_reps[layer][i] = guide_rep[layer].detach()

    def find_kth_neighbor_diff_class(self, x, label, k):

        nn = torch.zeros((x.size(0), ), dtype=torch.long)

        for i in range(x.size(0)):
            dist = ((x[i].cpu() - self.dknn.x_train).view(
                self.dknn.x_train.size(0), -1) ** 2).sum(1)
            # we want to exclude samples that are classified to the
            # same label as x_orig
            ind = np.where(self.y_pred == label[i])[0]
            dist[ind] += INFTY
            topk = torch.topk(dist, k, largest=False)[1]
            nn[i] = dist[topk[-1]]

        return self.dknn.x_train[nn]

    @staticmethod
    def atanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    @staticmethod
    def sigmoid(x, a=1):
        return 1 / (1 + torch.exp(-a * x))
