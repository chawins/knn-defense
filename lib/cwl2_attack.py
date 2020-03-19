import logging

import numpy as np
import torch
import torch.optim as optim

INFTY = 1e20


class CWL2Attack(object):
    """
    """

    def __init__(self, net, x_train=None, y_train=None):
        self.net = net
        self.x_train = x_train
        self.y_train = y_train

        if x_train is not None and y_train is not None:
            num_train = x_train.size(0)
            with torch.no_grad():
                self.y_pred = torch.zeros(
                    (num_train, ), device='cpu', dtype=torch.long)
                batch_size = 200
                num_batches = np.ceil(num_train / batch_size).astype(np.int32)
                for i in range(num_batches):
                    begin = i * batch_size
                    end = (i + 1) * batch_size
                    y_pred = net(x_train[begin:end].to('cuda')).argmax(1).cpu()
                    self.y_pred[begin:end] = y_pred

    def __call__(self, x_orig, label, init_mode, **kwargs):
        """
        init_mode == 0: start with original sample
        init_mode == 1: start with a nearby sample from another class
        init_mode == 2: both
        """
        # check for samples that are already misclassified
        with torch.no_grad():
            y_pred = self.net(x_orig).argmax(1)
            idx = np.where((y_pred == label).cpu().numpy())[0]
        x_adv = x_orig.clone()

        if init_mode != 2:
            x_adv[idx] = self.attack(x_orig[idx], label[idx],
                                     init_mode=init_mode, **kwargs)
        else:
            x0 = self.attack(x_orig[idx], label[idx], init_mode=0, **kwargs)
            x1 = self.attack(x_orig[idx], label[idx], init_mode=1, **kwargs)
            dist0 = (x0 - x_orig[idx]).view(len(idx), -1).norm(2, 1)
            dist1 = (x1 - x_orig[idx]).view(len(idx), -1).norm(2, 1)

            with torch.no_grad():
                # we can omit the adv check for init_mode = 1 since it
                # should always succeed by design
                y_pred = self.net(x0).argmax(1)
                is_adv0 = (y_pred != label[idx]).cpu().numpy()

            for i in range(len(idx)):
                if dist0[i] < dist1[i] and is_adv0[i]:
                    x_adv[idx[i]] = x0[i]
                else:
                    x_adv[idx[i]] = x1[i]
        return x_adv

    def attack(self, x_orig, label, targeted=False, init_mode=0,
               binary_search_steps=10, max_iterations=1000, confidence=0,
               learning_rate=1e-1, initial_const=1, abort_early=True,
               rand_start_std=0.1, check_adv_steps=100):
        """
        x_orig is tensor (requires_grad=False)
        """

        min_, max_ = x_orig.max(), x_orig.min()
        label = label.view(-1, 1)
        batch_size = x_orig.size(0)
        x_adv = x_orig.clone()

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
        const = torch.zeros((batch_size, ), device=x_orig.device)
        const += initial_const
        lower_bound = torch.zeros_like(const)
        upper_bound = torch.zeros_like(const) + INFTY
        best_l2dist = torch.zeros_like(const) + INFTY

        if init_mode == 1:
            with torch.no_grad():
                # search for nearest neighbor of incorrect class
                x_init = self.find_neighbor_diff_class(x_orig, label)
                z_init = to_attack_space(x_init.to('cuda')) - z_orig

        for binary_search_step in range(binary_search_steps):
            if binary_search_step == binary_search_steps - 1 and \
                    binary_search_steps >= 10:
                # in the last binary search step, use the upper_bound instead
                const = upper_bound

            # initialize z_delta
            z_delta = torch.zeros_like(z_orig, requires_grad=True)
            if init_mode == 1:
                with torch.no_grad():
                    z_delta += z_init
            if rand_start_std and binary_search_step > 0:
                with torch.no_grad():
                    z_delta += torch.normal(
                        0., rand_start_std, z_delta.size()).to('cuda')
            loss_at_previous_check = torch.zeros(
                1, device=x_orig.device) + INFTY

            # create a new optimizer
            # optimizer = optim.Adam([z_delta], lr=learning_rate)
            optimizer = optim.RMSprop([z_delta], lr=learning_rate)

            for iteration in range(max_iterations):
                optimizer.zero_grad()
                x = to_model_space(z_orig + z_delta)
                logits = self.net(x)
                loss, l2dist = self.loss_function(
                    x, label, logits, targeted, const, x_recon, confidence)
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

                # every <check_adv_steps>, save adversarial samples
                # with minimal perturbation
                if (iteration % check_adv_steps == 0 or
                        iteration == max_iterations):
                    with torch.no_grad():
                        logits = self.net(x)
                        is_adv = self.check_adv(
                            logits, label, targeted, confidence)
                    for i in range(batch_size):
                        if is_adv[i] and best_l2dist[i] > l2dist[i]:
                            x_adv[i] = x[i]
                            best_l2dist[i] = l2dist[i]

            with torch.no_grad():
                logits = self.net(x)
                is_adv = self.check_adv(logits, label, targeted, confidence)

            for i in range(batch_size):
                if is_adv[i]:
                    # sucessfully find adv
                    upper_bound[i] = const[i]
                else:
                    # fail to find adv
                    lower_bound[i] = const[i]
                if upper_bound[i] == INFTY:
                    # exponential search if adv has not been found
                    const[i] *= 10
                elif lower_bound[i] == 0:
                    const[i] /= 10
                else:
                    # binary search if adv has been found
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2
                # only keep adv with smallest l2dist
                if is_adv[i] and best_l2dist[i] > l2dist[i]:
                    x_adv[i] = x[i]
                    best_l2dist[i] = l2dist[i]

            with torch.no_grad():
                logits = self.net(x_adv)
                is_adv = self.check_adv(logits, label, targeted, confidence)
            print('binary step: %d; number of successful adv: %d/%d' %
                  (binary_search_step, is_adv.sum().cpu().numpy(), batch_size))

        return x_adv

    def find_neighbor_diff_class(self, x, label):

        nn = torch.zeros(x.size(0), dtype=torch.long)

        for i in range(x.size(0)):
            dist = (x[i].cpu() - self.x_train).view(
                self.x_train.size(0), -1).norm(2, 1)
            # we want to exclude samples that are classified to the
            # same label as x_orig
            ind = np.where(self.y_pred == label[i].cpu())[0]
            dist[ind] += INFTY
            nn[i] = dist.argmin()

        return self.x_train[nn]

    @classmethod
    def check_adv(cls, logits, label, targeted, confidence):
        if targeted:
            return torch.eq(torch.argmax(logits - confidence, 1),
                            label.squeeze())
        return torch.ne(torch.argmax(logits - confidence, 1), label.squeeze())

    @classmethod
    def loss_function(cls, x, label, logits, targeted, const, x_recon,
                      confidence):
        """Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x)."""

        other = cls.best_other_class(logits, label)
        if targeted:
            adv_loss = other - torch.gather(logits, 1, label)
        else:
            adv_loss = torch.gather(logits, 1, label) - other
        adv_loss = torch.max(torch.zeros_like(adv_loss), adv_loss + confidence)

        size = x.size(1) * x.size(2) * x.size(3)
        l2dist = torch.norm((x - x_recon).view(-1, size), dim=1)**2
        total_loss = l2dist + const * adv_loss.squeeze()

        return total_loss.mean(), l2dist.sqrt()

    @staticmethod
    def best_other_class(logits, exclude):
        """Returns the index of the largest logit, ignoring the class that
        is passed as `exclude`."""
        y_onehot = torch.zeros_like(logits)
        y_onehot.scatter_(1, exclude, 1)
        # make logits that we want to exclude a large negative number
        other_logits = logits - y_onehot * INFTY
        return other_logits.max(1)[0]

    @staticmethod
    def atanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))

# ============================================================================ #


class CWL2AttackNCA(object):
    """
    """

    def __init__(self, net):
        self.net = net
        x_train, y_train = net.train_data
        self.x_train = x_train
        self.y_train = y_train
        train_len = len(x_train)

        train_rep = net.get_train_rep(requires_grad=False)
        logits = torch.zeros((train_len, net.num_classes))
        mask = []
        for i in range(net.num_classes):
            mask.append((y_train == i).float().cuda())
        for i in range(train_len):
            dist = ((train_rep[i].unsqueeze(0) - train_rep) ** 2).sum(1)
            exp = torch.clamp(- dist * net.log_it.exp(), - 50, 50).exp()
            exp[i] = 0
            exp_sum = exp.sum()
            for j in range(net.num_classes):
                logits[i, j] = (mask[j] * exp).sum() / exp_sum
        self.y_pred = logits.argmax(1)

    def __call__(self, x_orig, label, init_mode, **kwargs):
        """
        init_mode == 0: start with original sample
        init_mode == 1: start with a nearby sample from another class
        init_mode == 2: both
        """
        # check for samples that are already misclassified
        with torch.no_grad():
            y_pred = self.net.compute_logits(x_orig).argmax(1)
            idx = np.where((y_pred == label).cpu().numpy())[0]
        x_adv = x_orig.clone()

        if init_mode != 2:
            x_adv[idx] = self.attack(x_orig[idx], label[idx],
                                     init_mode=init_mode, **kwargs)
        else:
            x0 = self.attack(x_orig[idx], label[idx], init_mode=0, **kwargs)
            x1 = self.attack(x_orig[idx], label[idx], init_mode=1, **kwargs)
            dist0 = (x0 - x_orig[idx]).view(len(idx), -1).norm(2, 1)
            dist1 = (x1 - x_orig[idx]).view(len(idx), -1).norm(2, 1)

            with torch.no_grad():
                # we can omit the adv check for init_mode = 1 since it
                # should always succeed by design
                y_pred = self.net.compute_logits(x0).argmax(1)
                is_adv0 = (y_pred != label[idx]).cpu().numpy()

            for i in range(len(idx)):
                if dist0[i] < dist1[i] and is_adv0[i]:
                    x_adv[idx[i]] = x0[i]
                else:
                    x_adv[idx[i]] = x1[i]
        return x_adv

    def attack(self, x_orig, label, targeted=False, init_mode=0,
               binary_search_steps=10, max_iterations=1000, confidence=0,
               learning_rate=1e-1, initial_const=1, abort_early=True,
               rand_start_std=0.1, check_adv_steps=100):
        """
        x_orig is tensor (requires_grad=False)
        """

        min_, max_ = x_orig.max(), x_orig.min()
        label = label.view(-1, 1)
        batch_size = x_orig.size(0)
        x_adv = x_orig.clone()

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
        const = torch.zeros((batch_size, ), device=x_orig.device)
        const += initial_const
        lower_bound = torch.zeros_like(const)
        upper_bound = torch.zeros_like(const) + INFTY
        best_l2dist = torch.zeros_like(const) + INFTY

        if init_mode == 1:
            with torch.no_grad():
                # search for nearest neighbor of incorrect class
                x_init = self.find_neighbor_diff_class(x_orig, label)
                z_init = to_attack_space(x_init.to('cuda')) - z_orig

        for binary_search_step in range(binary_search_steps):
            if binary_search_step == binary_search_steps - 1 and \
                    binary_search_steps >= 10:
                # in the last binary search step, use the upper_bound instead
                const = upper_bound

            # initialize z_delta
            z_delta = torch.zeros_like(z_orig, requires_grad=True)
            if init_mode == 1:
                with torch.no_grad():
                    z_delta += z_init
            if rand_start_std and binary_search_step > 0:
                with torch.no_grad():
                    z_delta += torch.normal(
                        0., rand_start_std, z_delta.size()).to('cuda')
            loss_at_previous_check = torch.zeros(
                1, device=x_orig.device) + INFTY

            # create a new optimizer
            # optimizer = optim.Adam([z_delta], lr=learning_rate)
            optimizer = optim.RMSprop([z_delta], lr=learning_rate)

            for iteration in range(max_iterations):
                optimizer.zero_grad()
                x = to_model_space(z_orig + z_delta)
                logits = self.net.compute_logits(x, requires_grad=True)
                loss, l2dist = self.loss_function(
                    x, label, logits, targeted, const, x_recon, confidence)
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

                # every <check_adv_steps>, save adversarial samples
                # with minimal perturbation
                if (iteration % check_adv_steps == 0 or
                        iteration == max_iterations):
                    with torch.no_grad():
                        logits = self.net.compute_logits(x)
                        is_adv = self.check_adv(
                            logits, label, targeted, confidence)
                    for i in range(batch_size):
                        if is_adv[i] and best_l2dist[i] > l2dist[i]:
                            x_adv[i] = x[i]
                            best_l2dist[i] = l2dist[i]

            with torch.no_grad():
                logits = self.net.compute_logits(x)
                is_adv = self.check_adv(logits, label, targeted, confidence)

            for i in range(batch_size):
                if is_adv[i]:
                    # sucessfully find adv
                    upper_bound[i] = const[i]
                else:
                    # fail to find adv
                    lower_bound[i] = const[i]
                if upper_bound[i] == INFTY:
                    # exponential search if adv has not been found
                    const[i] *= 10
                elif lower_bound[i] == 0:
                    const[i] /= 10
                else:
                    # binary search if adv has been found
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2
                # only keep adv with smallest l2dist
                if is_adv[i] and best_l2dist[i] > l2dist[i]:
                    x_adv[i] = x[i]
                    best_l2dist[i] = l2dist[i]

            with torch.no_grad():
                logits = self.net.compute_logits(x_adv)
                is_adv = self.check_adv(logits, label, targeted, confidence)
            print('binary step: %d; number of successful adv: %d/%d' %
                  (binary_search_step, is_adv.sum().cpu().numpy(), batch_size))

        return x_adv

    def find_neighbor_diff_class(self, x, label):

        nn = torch.zeros(x.size(0), dtype=torch.long)

        for i in range(x.size(0)):
            dist = (x[i].cpu() - self.x_train).view(
                self.x_train.size(0), -1).norm(2, 1)
            # we want to exclude samples that are classified to the
            # same label as x_orig
            ind = np.where(self.y_pred == label[i].cpu())[0]
            dist[ind] += INFTY
            nn[i] = dist.argmin()

        return self.x_train[nn]

    @classmethod
    def check_adv(cls, logits, label, targeted, confidence):
        if targeted:
            return torch.eq(torch.argmax(logits - confidence, 1),
                            label.squeeze())
        return torch.ne(torch.argmax(logits - confidence, 1), label.squeeze())

    @classmethod
    def loss_function(cls, x, label, logits, targeted, const, x_recon,
                      confidence):
        """Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x)."""

        other = cls.best_other_class(logits, label)
        if targeted:
            adv_loss = other - torch.gather(logits, 1, label)
        else:
            adv_loss = torch.gather(logits, 1, label) - other
        adv_loss = torch.max(torch.zeros_like(adv_loss), adv_loss + confidence)

        size = x.size(1) * x.size(2) * x.size(3)
        l2dist = torch.norm((x - x_recon).view(-1, size), dim=1)**2
        total_loss = l2dist + const * adv_loss.squeeze()

        return total_loss.mean(), l2dist.sqrt()

    @staticmethod
    def best_other_class(logits, exclude):
        """Returns the index of the largest logit, ignoring the class that
        is passed as `exclude`."""
        y_onehot = torch.zeros_like(logits)
        y_onehot.scatter_(1, exclude, 1)
        # make logits that we want to exclude a large negative number
        other_logits = logits - y_onehot * INFTY
        return other_logits.max(1)[0]

    @staticmethod
    def atanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))
