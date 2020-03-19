'''
Implement CVPR 2019 Paper:
"Defense Against Adversarial Images using Web-Scale Nearest-Neighbor Search"
'''
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import faiss
from lib.faiss_utils import *
from sklearn.decomposition import PCA

INFTY = 1e20


class CVPR_Defense(object):
    """
    An object that we use to create and store a deep k-nearest neighbor (knn)
    that uses Euclidean distance as a metric
    """

    def __init__(self, model, x_train, y_train, layers, k=75, num_classes=10,
                 device='cuda'):
        """
        Parameters
        ----------
        model : torch.nn.Module
            neural network model that extracts the representations
        x_train : torch.tensor
            a tensor of training samples with shape (num_train_samples, ) +
            input_shape
        y_train : torch.tensor
            a tensor of labels corresponding to samples in x_train with shape
            (num_train_samples, )
        layers : list of str
            a list of layer names that are used in knn
        k : int, optional
            the number of neighbors to consider, i.e. k in the kNN part
            (default is 75)
        num_classes : int, optional
            the number of classes (default is 10)
        device : str, optional
            name of the device model is on (default is 'cuda')
        """
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.layers = layers
        self.k = k
        self.num_classes = num_classes
        self.device = device
        self.indices = []
        self.activations = {}
        self.index = None

        self.train_logits = torch.zeros((x_train.size(0), num_classes))
        batch_size = 200
        num_batches = int(np.ceil(x_train.size(0) / batch_size))
        for i in range(num_batches):
            begin, end = i * batch_size, (i + 1) * batch_size
            with torch.no_grad():
                self.train_logits[begin:end] = self.model(
                    x_train[begin:end].to(device)).cpu()

        # register hook to get representations
        layer_count = 0
        for name, module in self.model.named_children():
            # if layer name is one of the names specified in self.layers,
            # register a hook to extract the activation at every forward pass
            if name in self.layers:
                module.register_forward_hook(self._get_activation(name))
                layer_count += 1
        assert layer_count == len(layers)

        reps = self.get_activations(
            self.x_train, pca=False, requires_grad=False).cpu()
        pca = PCA(n_components=64)
        pca.fit(reps.cpu().numpy())
        self.pca = torch.tensor(pca.components_.T).float().to(device)
        self.mean = torch.tensor(pca.mean_).float().to(device)
        xb = (reps - self.mean.cpu()) @ self.pca.cpu()
        self._build_index(xb)

    def _get_activation(self, name):
        """Hook used to get activation from specified layer name

        Parameters
        ----------
        name : str
            name of the layer to collect the activations

        Returns
        -------
        hook
            the hook function
        """
        def hook(model, input, output):
            self.activations[name] = output
        return hook

    def _build_index(self, xb):
        """Build faiss index from a given set of samples

        Parameters
        ----------
        xb : torch.tensor
            tensor of samples to build the search index, shape is
            (num_samples, dim)

        Returns
        -------
        index
            faiss index built on the given samples
        """

        d = xb.size(-1)
        # brute-force search on GPU (GPU generally doesn't have enough memory)
        # res = faiss.StandardGpuResources()
        # index = faiss.GpuIndexFlatIP(res, d)

        # brute-force search on CPU
        self.index = faiss.IndexFlatL2(d)
        self.index.add(xb.detach().cpu().numpy())

    def get_activations(self, x, pca=True, batch_size=500, requires_grad=True,
                        device=None):
        """Get activations at each layer in self.layers

        Parameters
        ----------
        x : torch.tensor
            tensor of input samples, shape = (num_samples, ) + input_shape
        batch_size : int, optional
            batch size (Default is 500)
        requires_grad : bool, optional
            whether or not to require gradients on the activations
            (Default is False)
        device : str
            name of the device the model is on (Default is None)

        Returns
        -------
        activations : dict
            dict of torch.tensor containing activations
        """
        if device is None:
            device = self.device

        # first run through to set an empty tensor of an appropriate size
        with torch.no_grad():
            num_total = x.size(0)
            num_batches = int(np.ceil(num_total / batch_size))
            activations = []
            self.model(x[0:1].to(device))
            for layer in self.layers:
                # size = self.activations[layer].size()
                if layer == 'conv1':
                    size = F.avg_pool2d(
                        self.activations[layer], 4, stride=2, padding=0).size()
                elif layer == 'conv2':
                    size = F.avg_pool2d(
                        self.activations[layer], 4, stride=1, padding=0).size()
                elif layer == 'conv3':
                    size = F.avg_pool2d(
                        self.activations[layer], 3, stride=1, padding=0).size()
                size = torch.tensor(size)[1:].prod().item()
                activations.append(torch.empty((num_total, size),
                                               dtype=torch.float32,
                                               device=device,
                                               requires_grad=False))

        with torch.set_grad_enabled(requires_grad):
            for i in range(num_batches):
                begin, end = i * batch_size, (i + 1) * batch_size
                # run a forward pass, the attribute self.activations get set
                # to activations of the current batch
                self.model(x[begin:end].to(device))
                # copy the extracted activations to the dictionary of
                # tensor allocated earlier
                for j, layer in enumerate(self.layers):
                    if j == 0:
                        feat = F.avg_pool2d(
                            self.activations[layer], 4, stride=2, padding=0)
                    elif j == 1:
                        feat = F.avg_pool2d(
                            self.activations[layer], 4, stride=1, padding=0)
                    elif j == 2:
                        feat = F.avg_pool2d(
                            self.activations[layer], 3, stride=1, padding=0)
                    activations[j][begin:end] = feat.view(feat.size(0), -1)

            acts = torch.cat(activations, dim=1)
            if pca:
                acts = (acts - self.mean) @ self.pca
            return acts

    def get_neighbors(self, x, k=None):
        """Find k neighbors of x at specified layers

        Parameters
        ----------
        x : torch.tensor
            samples to query, shape (num_samples, ) + input_shape
        k : int, optional
            number of neighbors (Default is self.k)

        Returns
        -------
        output : list
            list of len(layers) tuples of distances and indices of k neighbors
        """
        if k is None:
            k = self.k

        reps = self.get_activations(x, requires_grad=False).cpu()
        D, I = self.index.search(reps.numpy(), k)
        return D, I

    def get_output(self, x, k=None):
        """Find number of k-nearest neighbors in each class

        Arguments
        ---------
        x : torch.tensor
            samples to query, shape is (num_samples, ) + input_shape
        k : int, optional
            number of neighbors to check (Default is None)

        Returns
        -------
        class_counts : np.array
            array of numbers of neighbors in each class, shape is
            (num_samples, self.num_classes)
        """
        _, nb = self.get_neighbors(x, k=k)
        output = torch.zeros((x.size(0), self.num_classes))
        for i in range(x.size(0)):
            output[i] = self.train_logits[nb[i]].mean(0)
        return output.numpy()


# ============================================================================ #


class CVPR_Attack(object):
    """
    Implement gradient-based attack on (Deep) k-Nearest Neigbhor
    """

    def __init__(self, knn):
        self.knn = knn
        self.device = knn.device
        self.guide_reps = None
        self.thres = None
        self.coeff = None

        # classify x_train in knn (leave-one-out)
        out = self.knn.get_output(knn.x_train, k=(knn.k + 1))
        self.y_pred = out.argmax(1)

    def __call__(self, x_orig, label, norm, m=100,
                 init_mode=1, init_mode_k=1, binary_search_steps=5,
                 max_iterations=500, learning_rate=1e-2, initial_const=1,
                 max_linf=None, random_start=False, thres_steps=100,
                 check_adv_steps=100, verbose=True):
        """
        Parameters
        ----------
        knn : knn object
            knn (defined in lib/knn.py) that we want to attack
        x_orig : torch.tensor
            tensor of the original samples to attack. Does not need to require
            gradients, shape is (num_samples, ) + input_shape
        label : torch.tensor
            tensor of the label corresponding to x_orig
        guide_layer : str. optional
            layer name in which we want to find guide samples. Default is
            'relu1'
        m : int, optional
            number of guide samples. Default is 100
        binary_search_step : int, optional
            number of steps for binary search on the norm penalty constant.
            Default is 5
        max_iterations : int, optional
            number of optimization steps (per one binary search). Default is
            500
        learning_rate : float , optional
            step size or learning rate for the optimizer. Default is 1e-2
        initial_const : float, optional
            a number the norm penalty constant should be initialized to.
            Default is 1
        abort_early : bool, optional
            whether or not to abort the optimization early (before reaching
            max_iterations) if the objective does not improve from the past
            (max_iterations // 10) steps. Default is True
        max_linf : float, optional
            use to bound the L-inf norm of the attacks (addition to L-2 norm
            penalty). Set to None to not use this option. Default is None
        random_start : bool, optional
            whether or not to initialize the perturbation with small isotropic
            Gaussian noise. Default is False

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
        # self.coeff[:, :m // 2] += 1
        self.coeff[:, m // 2:] -= 1

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

        if init_mode == 2:
            with torch.no_grad():
                # search for nearest neighbor of incorrect class
                x_init = self.find_kth_neighbor_diff_class(
                    x_orig, label, init_mode_k)
                z_init = to_attack_space(x_init.to('cuda')) - z_orig

        # make a list of number of guide samples that linearly decreases
        start = (self.knn.k + 1) // 2
        end = max(m // 2, start + 1)
        m_list = np.arange(start, end, (end - start) / binary_search_steps)

        for binary_search_step in range(binary_search_steps):

            # reduce number of guide samples for successful attacks
            idx_m = binary_search_steps - binary_search_step - 1
            # m_new = np.ceil(m_list[idx_m]).astype(np.int32)
            m_new = m // 2

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
                        thres = self.knn.get_neighbors(x)[0][:, -1]
                        self.thres = torch.tensor(thres).to(self.device).view(
                            batch_size, 1)
                        self.find_guide_samples(x, label, m=m)

                reps = self.knn.get_activations(x, requires_grad=True)
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
        """Check if label of <x> predicted by <knn> matches with <label>"""
        y_pred = self.knn.get_output(x).argmax(1)
        # y_pred = self.knn.classify_soft(x).argmax(1)
        return torch.tensor((y_pred != label).astype(np.float32)).to(self.device)

    def loss_function(self, x, reps, const, x_recon, norm):
        """Returns the loss averaged over the batch (first dimension of x) and
        L-2 norm squared of the perturbation
        """

        batch_size = x.size(0)

        # compute loss on the first guide layer
        rep = reps.view(batch_size, 1, -1)
        dist = ((rep - self.guide_reps) ** 2).sum(2)
        fx = self.thres - dist
        adv_loss = F.relu(self.coeff.to(self.device) * fx + 1e-5).sum(1)

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

    def find_guide_samples(self, x, label, m=100):
        """Find k nearest neighbors to <x> that all have the same class but not
        equal to <label>
        """
        num_classes = self.knn.num_classes
        x_train = self.knn.x_train
        y_train = self.knn.y_train
        batch_size = x.size(0)
        nn = torch.zeros((m, ) + x.size()).transpose(0, 1)
        D, I = self.knn.get_neighbors(x, k=x_train.size(0))

        for i, (d, ind) in enumerate(zip(D, I)):
            mean_dist = np.zeros((num_classes, ))
            for j in range(num_classes):
                mean_dist[j] = np.mean(
                    d[np.where(y_train[ind] == j)[0]][:m // 2])
            mean_dist[label[i]] += INFTY
            nearest_label = mean_dist.argmin()
            nn_ind = np.where(y_train[ind] == nearest_label)[0][:m // 2]
            nn[i, m // 2:] = x_train[ind[nn_ind]]
            nn_ind = np.where(y_train[ind] == label[i])[0][:m // 2]
            nn[i, :m // 2] = x_train[ind[nn_ind]]

        # initialize self.guide_reps if empty
        if self.guide_reps is None:
            guide_rep = self.knn.get_activations(nn[0], requires_grad=False)
            # set a zero tensor before filling it
            size = (batch_size, ) + guide_rep.view(m, -1).size()
            self.guide_reps = torch.zeros(size, device=self.device)

        # fill self.guide_reps
        for i in range(batch_size):
            guide_rep = self.knn.get_activations(nn[i], requires_grad=False)
            self.guide_reps[i] = guide_rep.view(m, -1).detach()

    def find_kth_neighbor_diff_class(self, x, label, k):

        nn = torch.zeros((x.size(0), ), dtype=torch.long)

        for i in range(x.size(0)):
            dist = ((x[i].cpu() - self.knn.x_train).view(
                self.knn.x_train.size(0), -1) ** 2).sum(1)
            # we want to exclude samples that are classified to the
            # same label as x_orig
            ind = np.where(self.y_pred == label[i])[0]
            dist[ind] += INFTY
            topk = torch.topk(dist, k, largest=False)[1]
            nn[i] = dist[topk[-1]]

        return self.knn.x_train[nn]

    @staticmethod
    def atanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    @staticmethod
    def sigmoid(x, a=1):
        return 1 / (1 + torch.exp(-a * x))
