'''Implement gradient-based attack on DkNN with L-2 constraint'''

import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

INFTY = 1e20


class DKNNExpAttack(object):
    """
    Implement gradient-based attack on Deep k-Nearest Neigbhor that uses
    L-2 distance as a metric
    """

    def __init__(self, dknn):
        self.dknn = dknn
        self.device = dknn.device
        self.layers = dknn.layers
        self.guide_reps = {}
        self.thres = None
        self.coeff = None

    def __call__(self, x_orig, label, guide_layer='relu1', m=100,
                 binary_search_steps=5, max_iterations=500,
                 learning_rate=1e-2, initial_const=1, max_linf=None,
                 random_start=False, thres_steps=100, check_adv_steps=100,
                 verbose=True):
        """
        Parameters
        ----------
        dknn : DKNN object
            DkNN (defined in lib/dknn.py) that we want to attack
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

        min_, max_ = x_orig.min(), x_orig.max()
        if max_linf is not None:
            min_ = torch.max(x_orig - max_linf, min_)
            max_ = torch.min(x_orig + max_linf, max_)
        batch_size = x_orig.size(0)
        x_adv = x_orig.clone()
        label = label.cpu().numpy()
        input_shape = x_orig.detach().cpu().numpy().shape
        # initialize coeff for guide samples
        self.coeff = torch.zeros((x_orig.size(0), m))
        self.coeff[:, :m // 2] += 1
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
            """Transforms an input from the attack space
            to the model space. This transformation and
            the returned gradient are elementwise."""

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
        best_l2dist = torch.zeros_like(const) + INFTY

        for binary_search_step in range(binary_search_steps):
            if (binary_search_step == binary_search_steps - 1 and
                    binary_search_steps >= 10):
                    # in the last binary search step, use the upper_bound instead
                    # to ensure that unsuccessful attacks use the largest
                    # possible constant
                const = upper_bound

            # initialize perturbation in transformed space
            if not random_start:
                z_delta = torch.zeros_like(z_orig, requires_grad=True)
            else:
                rand = np.random.randn(*input_shape) * 1e-2
                z_delta = torch.tensor(
                    rand, dtype=torch.float32, requires_grad=True,
                    device=self.device)
            # loss_at_previous_check = torch.zeros(1, device=self.device) + INFTY

            # create a new optimizer
            optimizer = optim.RMSprop([z_delta], lr=learning_rate)

            # add learning rate scheduler
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=150,
                threshold=0.01, threshold_mode='rel')

            for iteration in range(max_iterations):
                optimizer.zero_grad()
                x = to_model_space(z_orig + z_delta)

                # adaptively choose threshold and guide samples every
                # <thres_steps> iterations
                with torch.no_grad():
                    if iteration % thres_steps == 0:
                        # order = (self.dknn.k + 1) // 2 - 1
                        # thres = self.dknn.get_neighbors(x)[0][0][:, order]
                        thres = self.dknn.get_neighbors(x)[0][0][:, -1]
                        self.thres = torch.tensor(thres).to(self.device).view(
                            batch_size, 1)
                        self.find_guide_samples(
                            x, label, m=m, layer=guide_layer)

                reps = self.dknn.get_activations(x, requires_grad=True)
                loss, l2dist = self.loss_function(
                    x, reps, const, x_recon)
                loss.backward()
                optimizer.step()
                # lr_scheduler.step(loss)

                if (verbose and iteration %
                        (np.ceil(max_iterations / 10)) == 0):
                    print('    step: %d; loss: %.3f; l2dist: %.3f' %
                          (iteration, loss.cpu().detach().numpy(),
                           l2dist.mean().cpu().detach().numpy()))

                # every <check_adv_steps>, save adversarial samples
                # with minimal perturbation
                if ((iteration + 1) % check_adv_steps == 0 or
                        iteration == max_iterations):
                    is_adv = self.check_adv(x, label)
                    for i in range(batch_size):
                        if is_adv[i] and best_l2dist[i] > l2dist[i]:
                            x_adv[i] = x[i]
                            best_l2dist[i] = l2dist[i]

            # check how many attacks have succeeded
            with torch.no_grad():
                is_adv = self.check_adv(x, label)
                if verbose:
                    print(is_adv.sum())

            for i in range(batch_size):
                # set new upper and lower bounds
                if is_adv[i]:
                    upper_bound[i] = const[i]
                else:
                    lower_bound[i] = const[i]
                # set new const
                if upper_bound[i] == INFTY:
                    # exponential search if adv has not been found
                    const[i] *= 10
                else:
                    # binary search if adv has been found
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2
                # only keep adv with smallest l2dist
                if is_adv[i] and best_l2dist[i] > l2dist[i]:
                    x_adv[i] = x[i]
                    best_l2dist[i] = l2dist[i]

            # check the current attack success rate (combined with previous
            # binary search steps)
            if verbose:
                with torch.no_grad():
                    is_adv = self.check_adv(x_adv, label)
                    print('binary step: %d; number of successful adv: %d/%d' %
                          (binary_search_step, is_adv.sum(), batch_size))

        return x_adv

    def check_adv(self, x, label):
        """Check if label of <x> predicted by <dknn> matches with <label>"""
        y_pred = self.dknn.classify(x).argmax(1)
        return torch.tensor((y_pred != label).astype(np.float32)).to(self.device)

    def loss_function(self, x, reps, const, x_recon):
        """Returns the loss averaged over the batch (first dimension of x) and
        L-2 norm squared of the perturbation
        """

        batch_size = x.size(0)
        adv_loss = torch.zeros(
            (batch_size, len(self.layers)), device=self.device)
        # find squared L-2 distance between original samples and their
        # adversarial examples at each layer
        for l, layer in enumerate(self.layers):
            rep = reps[layer].view(batch_size, 1, -1)
            dist = ((rep - self.guide_reps[layer])**2).sum(2)
            # fx = self.sigmoid((self.thres - dist).clamp(-80 / self.a, 80 / self.a), a=self.a)
            # fx = -dist
            fx = self.thres - dist
            # Fx = (coeff.to(device) * fx).sum(1)
            Fx = torch.max(torch.tensor(0., device=self.device),
                           self.coeff.to(self.device) * fx).sum(1)
            # adv_loss[:, l] = torch.max(torch.tensor(-1., device=device), Fx)
            adv_loss[:, l] = Fx
        # find L-2 norm squared of perturbation
        l2dist = torch.norm((x - x_recon).view(batch_size, -1), dim=1)**2
        # total_loss is sum of squared perturbation norm and squared distance
        # of representations, multiplied by constant
        total_loss = l2dist + const * adv_loss.mean(1)

        return total_loss.mean(), l2dist.sqrt()

    def find_guide_samples(self, x, label, m=100, layer='relu1'):
        """Find k nearest neighbors to <x> that all have the same class but not
        equal to <label>
        """
        num_classes = self.dknn.num_classes
        x_train = self.dknn.x_train
        y_train = self.dknn.y_train
        batch_size = x.size(0)
        nn = torch.zeros((m, ) + x.size()).transpose(0, 1)
        D, I = self.dknn.get_neighbors(
            x, k=x_train.size(0), layers=[layer])[0]

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
        if not self.guide_reps:
            guide_rep = self.dknn.get_activations(
                nn[0], requires_grad=False)
            for l in self.layers:
                # set a zero tensor before filling it
                size = (batch_size, ) + guide_rep[l].view(m, -1).size()
                self.guide_reps[l] = torch.zeros(size, device=self.device)

        # fill self.guide_reps
        for i in range(batch_size):
            guide_rep = self.dknn.get_activations(
                nn[i], requires_grad=False)
            self.guide_reps[layer][i] = guide_rep[layer].view(
                m, -1).detach()

    @staticmethod
    def atanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    @staticmethod
    def sigmoid(x, a=1):
        return 1 / (1 + torch.exp(-a * x))
