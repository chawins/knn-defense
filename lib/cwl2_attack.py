import logging

import numpy as np
import torch
import torch.optim as optim


class CWL2Attack(object):
    """
    """

    def __call__(self, net, x_orig, label, targeted=False,
                 binary_search_steps=10, max_iterations=1000,
                 confidence=0, learning_rate=1e-1,
                 initial_const=1, abort_early=True):
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
        upper_bound = torch.zeros_like(const) + 1e9
        best_l2dist = torch.zeros_like(const) + 1e9

        for binary_search_step in range(binary_search_steps):
            if binary_search_step == binary_search_steps - 1 and \
                    binary_search_steps >= 10:
                # in the last binary search step, use the upper_bound instead
                # TODO: find out why... it's not obvious why this is useful
                const = upper_bound

            z_delta = torch.zeros_like(z_orig, requires_grad=True)
            loss_at_previous_check = torch.zeros(1, device=x_orig.device) + 1e9

            # create a new optimizer
            optimizer = optim.Adam([z_delta], lr=learning_rate)

            for iteration in range(max_iterations):
                optimizer.zero_grad()
                x = to_model_space(z_orig + z_delta)
                logits = net(x)
                loss, l2dist = self.loss_function(
                    x, label, logits, targeted, const, x_recon, confidence)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    is_adv = self.check_adv(
                        logits, label, targeted, confidence)

                if iteration % (np.ceil(max_iterations / 10)) == 0:
                    print('    step: %d; loss: %.3f; l2dist: %.3f' %
                          (iteration, loss.cpu().detach().numpy(),
                           l2dist.mean().cpu().detach().numpy()))

                if abort_early and iteration % (np.ceil(max_iterations / 10)) == 0:
                    # after each tenth of the iterations, check progress
                    if torch.gt(loss, .9999 * loss_at_previous_check):
                        break  # stop Adam if there has not been progress
                    loss_at_previous_check = loss

                for i in range(batch_size):
                    if is_adv[i]:
                        # sucessfully find adv
                        upper_bound[i] = const[i]
                    else:
                        # fail to find adv
                        lower_bound[i] = const[i]

            for i in range(batch_size):
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
                logits = net(x_adv)
                is_adv = self.check_adv(logits, label, targeted, confidence)
            print('binary step: %d; number of successful adv: %d/%d' %
                  (binary_search_step, is_adv.sum().cpu().numpy(), batch_size))

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
        other_logits = logits - y_onehot * 1e9
        return other_logits.max(1)[0]

    @staticmethod
    def atanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))
