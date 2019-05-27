import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


class DKNNAttack(object):
    """
    """

    def __call__(self, dknn, x_orig, label, guide_layer='relu1', m=100,
                 binary_search_steps=5, max_iterations=500,
                 learning_rate=1e-2, initial_const=1, abort_early=True,
                 max_linf=None):
        """
        x_orig is tensor (requires_grad=False)
        """

        min_, max_ = x_orig.min(), x_orig.max()
        if max_linf is not None:
            min_ = torch.max(x_orig - max_linf, min_)
            max_ = torch.min(x_orig + max_linf, max_)
        batch_size = x_orig.size(0)
        x_adv = x_orig.clone()
        label = label.cpu().numpy()
        device = dknn.device

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
        const = torch.zeros((batch_size, ), device=device)
        const += initial_const
        lower_bound = torch.zeros_like(const)
        upper_bound = torch.zeros_like(const) + 1e9
        best_l2dist = torch.zeros_like(const) + 1e9

        # find a set of guide neighbors
        # guide_id = dknn.get_neighbors(x_orig, k=300, layers=guide_layer)[0][1]
        # guide_label = torch.ne(dknn.y_train[guide_id],
        # guide_sample=dknn.x_train[guide_id]
        # with torch.no_grad():
        #     guide_reps = dknn.get_activations(guide_sample)

        with torch.no_grad():
            # choose guide samples and get their representations
            x_guide = self.find_guide_samples(
                dknn, x_orig, label, k=m, layer=guide_layer)
            guide_reps = {}
            for i in range(batch_size):
                guide_rep = dknn.get_activations(x_guide[i])
                for layer in dknn.layers:
                    if i == 0:
                        # set a zero tensor before filling it
                        size = (batch_size, ) + \
                            guide_rep[layer].view(m, -1).size()
                        guide_reps[layer] = torch.zeros(size).to(device)
                    guide_reps[layer][i] = F.normalize(
                        guide_rep[layer].view(m, -1), 2, 1)

        for binary_search_step in range(binary_search_steps):
            if (binary_search_step == binary_search_steps - 1 and
                    binary_search_steps >= 10):
                    # in the last binary search step, use the upper_bound instead
                    # TODO: find out why... it's not obvious why this is useful
                const = upper_bound

            z_delta = torch.zeros_like(z_orig, requires_grad=True)
            loss_at_previous_check = torch.zeros(1, device=device) + 1e9

            # create a new optimizer
            optimizer = optim.Adam([z_delta], lr=learning_rate)

            for iteration in range(max_iterations):
                optimizer.zero_grad()
                x = to_model_space(z_orig + z_delta)
                reps = dknn.get_activations(x)
                loss, l2dist = self.loss_function(
                    x, reps, guide_reps, dknn.layers, const, x_recon, device)
                loss.backward()
                optimizer.step()

                if iteration % (np.ceil(max_iterations / 10)) == 0:
                    print('    step: %d; loss: %.3f; l2dist: %.3f' %
                          (iteration, loss.cpu().detach().numpy(),
                           l2dist.mean().cpu().detach().numpy()))

                if abort_early and iteration % (np.ceil(max_iterations / 10)) == 0:
                    # after each tenth of the iterations, check progress
                    if torch.gt(loss, .9999 * loss_at_previous_check):
                        break  # stop Adam if there has not been progress
                    loss_at_previous_check = loss

            with torch.no_grad():
                is_adv = self.check_adv(dknn, x, label)
                print(is_adv.sum())

            for i in range(batch_size):
                # set new upper and lower bounds
                if is_adv[i]:
                    upper_bound[i] = const[i]
                else:
                    lower_bound[i] = const[i]
                # set new const
                if upper_bound[i] == 1e9:
                    # exponential search if adv has not been found
                    const[i] *= 10
                else:
                    # binary search if adv has been found
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2
                # only keep adv with smallest l2dist
                if is_adv[i] and best_l2dist[i] > l2dist[i]:
                    x_adv[i] = x[i]
                    best_l2dist[i] = l2dist[i]

            with torch.no_grad():
                is_adv = self.check_adv(dknn, x_adv, label)
            print('binary step: %d; number of successful adv: %d/%d' %
                  (binary_search_step, is_adv.sum(), batch_size))

        return x_adv

    @classmethod
    def check_adv(cls, dknn, x, label):
        y_pred = dknn.classify(x).argmax(1)
        return torch.tensor((y_pred != label).astype(np.float32)).to(dknn.device)
        # y_pred = dknn.classify_soft(x).argmax(1)
        # return (y_pred != torch.tensor(label)).to(dknn.device)

    @classmethod
    def loss_function(cls, x, reps, guide_reps, layers, const, x_recon, device):
        """Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x)."""

        batch_size = x.size(0)
        adv_loss = torch.zeros((batch_size, len(layers)), device=device)
        for l, layer in enumerate(layers):
            # cosine distance
            rep = F.normalize(reps[layer].view(
                batch_size, -1), 2, 1).unsqueeze(1)
            # (1) directly minimize loss
            # adv_loss[:, l] = (rep * guide_reps[layer]).sum((1, 2))

            # (2) use sigmoid
            # this threshold is calculated with k = 75 on basic model
            thres = torch.tensor(
                [0.7260, 0.6874, 0.7105, 0.9484], device=device)
            # thres = torch.tensor([0.7105], device=device)
            a = 4
            adv_loss[:, l] = cls.sigmoid(
                (rep * guide_reps[layer]).sum(2) - thres[l], a=a).sum(1)

            # (3) use soft version
            # width = 1
            # # adv_loss[:, l] = torch.log(torch.exp(
            # #     -2 * (1 - (rep * guide_reps[layer]).sum(2)) / width**2).sum(1))
            # adv_loss[:, l] = torch.log(torch.exp(
            #     (rep * guide_reps[layer]).sum(2) / width**2).sum(1))

            # (4) use max instead of sigmoid
            # thres = torch.tensor(
            #     [0.7260, 0.6874, 0.7105, 0.9484], device=device)
            # thres = torch.tensor([0.7105], device=device)
            # zero = torch.tensor(0.).to(device)
            # adv_loss[:, l] = - torch.max(
            #     thres[l] - (rep * guide_reps[layer]).sum(2), zero).sum(1)

        l2dist = torch.norm((x - x_recon).view(batch_size, -1), dim=1)**2
        total_loss = l2dist - const * adv_loss.mean(1)

        return total_loss.mean(), l2dist.sqrt()

    @staticmethod
    def find_guide_samples(dknn, x, label, k=100, layer='relu1'):
        """
        find k nearest neighbors of the same class (not equal to y_Q) but
        closest to Q
        """
        num_classes = dknn.num_classes
        nn = torch.zeros((k, ) + x.size()).permute(1, 0, 2, 3, 4)
        D, I = dknn.get_neighbors(
            x, k=dknn.x_train.size(0), layers=[layer])[0]

        for i, (d, ind) in enumerate(zip(D, I)):
            mean_dist = np.zeros((num_classes, ))
            for j in range(num_classes):
                mean_dist[j] = np.mean(
                    d[np.where(dknn.y_train[ind] == j)[0]][:k])
            # TODO: this may depend on the index used
            # mean_dist[label[i]] += 1e9
            # nearest_label = mean_dist.argmin()
            mean_dist[label[i]] -= 1e9
            nearest_label = mean_dist.argmax()
            nn_ind = np.where(dknn.y_train[ind] == nearest_label)[0][:k]
            nn[i] = dknn.x_train[ind[nn_ind]]

        return nn

    @staticmethod
    def atanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    @staticmethod
    def sigmoid(x, a=1):
        return 1 / (1 + torch.exp(-a * x))


# ============================================================================

class SoftDKNNAttack(object):
    """
    """

    def __call__(self, dknn, x_orig, label, layer='relu1', m=100,
                 binary_search_steps=5, max_iterations=500,
                 learning_rate=1e-2, initial_const=1, abort_early=True,
                 max_linf=None):
        """
        x_orig is tensor (requires_grad=False)
        """

        min_, max_ = x_orig.min(), x_orig.max()
        if max_linf is not None:
            min_ = torch.max(x_orig - max_linf, min_)
            max_ = torch.min(x_orig + max_linf, max_)
        batch_size = x_orig.size(0)
        x_adv = x_orig.clone()
        # label = label.cpu().numpy()
        label = label.view(-1, 1)
        device = dknn.device

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
        const = torch.zeros((batch_size, ), device=device)
        const += initial_const
        lower_bound = torch.zeros_like(const)
        upper_bound = torch.zeros_like(const) + 1e9
        best_l2dist = torch.zeros_like(const) + 1e9

        with torch.no_grad():
            train_reps = dknn.get_activations(dknn.x_train)[layer]
            train_reps = F.normalize(
                train_reps.view(dknn.x_train.size(0), -1), 2, 1)

        for binary_search_step in range(binary_search_steps):
            if (binary_search_step == binary_search_steps - 1 and
                    binary_search_steps >= 10):
                    # in the last binary search step, use the upper_bound instead
                    # TODO: find out why... it's not obvious why this is useful
                const = upper_bound

            z_delta = torch.zeros_like(z_orig, requires_grad=True)
            loss_at_previous_check = torch.zeros(1, device=device) + 1e9

            # create a new optimizer
            optimizer = optim.Adam([z_delta], lr=learning_rate)

            for iteration in range(max_iterations):
                optimizer.zero_grad()
                x = to_model_space(z_orig + z_delta)
                logits = self.get_logits(dknn, x, train_reps, layer)
                loss, l2dist = self.loss_function(
                    logits, x, label, const, x_recon)
                loss.backward()
                optimizer.step()

                if iteration % (np.ceil(max_iterations / 10)) == 0:
                    print('    step: %d; loss: %.3f; l2dist: %.3f' %
                          (iteration, loss.cpu().detach().numpy(),
                           l2dist.mean().cpu().detach().numpy()))

                if abort_early and iteration % (np.ceil(max_iterations / 10)) == 0:
                    # after each tenth of the iterations, check progress
                    if torch.gt(loss, .9999 * loss_at_previous_check):
                        break  # stop Adam if there has not been progress
                    loss_at_previous_check = loss

            with torch.no_grad():
                is_adv = self.check_adv(logits, label)
                print(is_adv.sum())

            for i in range(batch_size):
                # set new upper and lower bounds
                if is_adv[i]:
                    upper_bound[i] = const[i]
                else:
                    lower_bound[i] = const[i]
                # set new const
                if upper_bound[i] == 1e9:
                    # exponential search if adv has not been found
                    const[i] *= 10
                else:
                    # binary search if adv has been found
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2
                # only keep adv with smallest l2dist
                if is_adv[i] and best_l2dist[i] > l2dist[i]:
                    x_adv[i] = x[i]
                    best_l2dist[i] = l2dist[i]

            with torch.no_grad():
                logits = self.get_logits(dknn, x_adv, train_reps, layer)
                is_adv = self.check_adv(logits, label)
            print('binary step: %d; number of successful adv: %d/%d' %
                  (binary_search_step, is_adv.sum(), batch_size))

        return x_adv

    @classmethod
    def check_adv(cls, logits, label):
        return torch.ne(torch.argmax(logits - 1e-2, 1), label.squeeze(1))

    @classmethod
    def loss_function(cls, logits, x, label, const, x_recon):
        """Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x)."""

        batch_size = x.size(0)
        other = cls.best_other_class(logits, label)
        adv_loss = torch.gather(logits, 1, label).squeeze(1) - other
        adv_loss = torch.max(torch.zeros_like(adv_loss), adv_loss + 1e-2)
        l2dist = torch.norm((x - x_recon).view(batch_size, -1), dim=1)**2
        total_loss = l2dist + const * adv_loss

        return total_loss.mean(), l2dist.sqrt()

    @staticmethod
    def get_logits(dknn, x, train_reps, layer):
        temp = 2e-2
        batch_size = x.size(0)

        # cosine distance
        reps = dknn.get_activations(x)[layer]
        reps = F.normalize(reps.view(batch_size, -1), 2, 1).unsqueeze(1)
        dist_exp = torch.zeros(
            (x.size(0), dknn.num_classes), device=dknn.device)

        for i, rep in enumerate(reps):
            cos = ((rep * train_reps).sum(1) / temp).exp()
            for l in range(dknn.num_classes):
                dist_exp[i, l] = cos[dknn.y_train == l].mean()
        logits = torch.log(dist_exp / dist_exp.sum(1).unsqueeze(1))

        return logits

    @staticmethod
    def best_other_class(logits, exclude):
        """Returns the index of the largest logit, ignoring the class that
        is passed as `exclude`."""
        y_onehot = torch.zeros_like(logits)
        y_onehot.scatter_(1, exclude, 1)
        # make logits that we want to exclude a large negative number
        other_logits = logits - y_onehot * 1e35
        return other_logits.max(1)[0]

    @staticmethod
    def atanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    @staticmethod
    def sigmoid(x, a=1):
        return 1 / (1 + torch.exp(-a * x))
