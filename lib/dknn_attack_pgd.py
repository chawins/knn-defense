import logging

import numpy as np
import torch
import torch.optim as optim

INFTY = 1e20


class DKNN_PGD(object):
    """
    Implement gradient-based attack on DkNN with L-inf norm constraint.
    The loss function is the same as the L-2 attack, but it uses PGD as an
    optimizer.
    """

    def __init__(self, dknn):
        self.dknn = dknn
        self.device = dknn.device
        self.layers = dknn.layers
        self.guide_reps = {}
        self.thres = None
        self.coeff = None

    def __call__(self, x_orig, label, guide_layer, m, epsilon=0.1,
                 max_epsilon=0.3, max_iterations=1000, num_restart=1,
                 rand_start=True, thres_steps=100, check_adv_steps=100,
                 verbose=True):

        # make sure we run at least once
        if num_restart < 1:
            num_restart = 1

        # if not using randomized start, no point in doing more than one start
        if not rand_start:
            num_restart = 1

        label = label.cpu().numpy()
        batch_size = x_orig.size(0)
        min_, max_ = x_orig.min(), x_orig.max()
        # initialize adv to the original
        x_adv = x_orig.detach()
        best_num_nn = np.zeros((batch_size, ))

        # set coefficient of guide samples
        self.coeff = torch.zeros((x_orig.size(0), m))
        self.coeff[:, :m // 2] += 1
        self.coeff[:, m // 2:] -= 1

        for i in range(num_restart):

            # initialize perturbation
            delta = torch.zeros_like(x_adv)
            if rand_start:
                delta.uniform_(- max_epsilon, max_epsilon)
            delta.requires_grad_()

            for iteration in range(max_iterations):
                x = torch.clamp(x_orig + delta, min_, max_)

                # adaptively choose threshold and guide samples every
                # <thres_steps> iterations
                with torch.no_grad():
                    if iteration % thres_steps == 0:
                        thres = self.dknn.get_neighbors(x)[0][0][:, -1]
                        self.thres = torch.tensor(thres).to(self.device).view(
                            batch_size, 1)
                        self.find_guide_samples(
                            x, label, m=m, layer=guide_layer)

                reps = self.dknn.get_activations(x, requires_grad=True)
                loss = self.loss_function(reps)
                loss.backward()
                # perform update on delta
                with torch.no_grad():
                    delta -= epsilon * delta.grad.detach().sign()
                    delta.clamp_(- max_epsilon, max_epsilon)

                if (verbose and iteration % (np.ceil(max_iterations / 10)) == 0):
                    print('    step: %d; loss: %.3f' %
                          (iteration, loss.cpu().detach().numpy()))

                if ((iteration + 1) % check_adv_steps == 0 or
                        iteration == max_iterations):
                    with torch.no_grad():
                        # check if x are adversarial. Only store adversarial
                        # examples if they have a larger number of wrong
                        # neighbors than orevious
                        is_adv, num_nn = self.check_adv(x, label)
                        for j in range(batch_size):
                            if is_adv[j] and num_nn[j] > best_num_nn[j]:
                                x_adv[j] = x[j]
                                best_num_nn[j] = num_nn[j]

            with torch.no_grad():
                is_adv, _ = self.check_adv(x_adv, label)
            if verbose:
                print('number of successful adv: %d/%d' %
                      (is_adv.sum(), batch_size))

        return x_adv

    def check_adv(self, x, label):
        """Check if label of <x> predicted by <dknn> matches with <label>"""
        output = self.dknn.classify(x)
        num_nn = output.max(1)
        y_pred = output.argmax(1)
        is_adv = (y_pred != label).astype(np.float32)
        return is_adv, num_nn

    def loss_function(self, reps):
        """Returns the loss averaged over the batch (first dimension of x) and
        L-2 norm squared of the perturbation
        """

        batch_size = reps[self.layers[0]].size(0)
        adv_loss = torch.zeros(
            (batch_size, len(self.layers)), device=self.device)
        # find squared L-2 distance between original samples and their
        # adversarial examples at each layer
        for l, layer in enumerate(self.layers):
            rep = reps[layer].view(batch_size, 1, -1)
            dist = ((rep - self.guide_reps[layer])**2).sum(2)
            fx = self.thres - dist
            Fx = torch.max(torch.tensor(0., device=self.device),
                           self.coeff.to(self.device) * fx).sum(1)
            adv_loss[:, l] = Fx

        return adv_loss.mean()

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
