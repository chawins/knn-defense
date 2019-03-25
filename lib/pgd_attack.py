import logging

import numpy as np
import torch
import torch.optim as optim


class PGDAttack(object):
    """
    """

    def __call__(self, net, x_orig, label, targeted=False,
                 epsilon=0.1, max_epsilon=0.3, max_iterations=1000,
                 random_restart=1):
        """
        x_orig is tensor (requires_grad=False)
        """

        label = label.view(-1, 1)
        batch_size = x_orig.size(0)
        min_, max_ = x_orig.min(), x_orig.max()
        x_adv = x_orig.clone()

        for i in range(random_restart):

            delta = torch.zeros_like(x_adv, requires_grad=True)
            best_confidence = torch.zeros(
                (batch_size, ), device=x_orig.device) - 1e9

            for iteration in range(max_iterations):
                x = torch.clamp(x_orig + delta, min_, max_)
                logits = net(x)
                loss = self.loss_function(logits, label, targeted)
                loss.backward()
                # perform update on delta
                with torch.no_grad():
                    delta -= epsilon * delta.grad.sign()
                    delta.clamp_(- max_epsilon, max_epsilon)

            with torch.no_grad():
                is_adv = self.check_adv(logits, label, targeted)

            # calculate confidence (difference between target class and the
            # class with the second highest score)
            real = torch.gather(logits, 1, label)
            other = self.best_other_class(logits, label)
            if targeted:
                confidence = real - other
            else:
                confidence = other - other

            for i in range(batch_size):
                # only keep adv with highest confidence
                if is_adv[i] and best_confidence[i] < confidence[i]:
                    x_adv[i] = x[i]
                    best_confidence[i] = confidence[i]

        with torch.no_grad():
            logits = net(x_adv)
            is_adv = self.check_adv(logits, label, targeted)
        print('number of successful adv: %d/%d' %
              (is_adv.sum().cpu().numpy(), batch_size))

    @classmethod
    def check_adv(cls, logits, label, targeted):
        if targeted:
            return torch.eq(torch.argmax(logits, 1), label.squeeze())
        return torch.ne(torch.argmax(logits, 1), label.squeeze())

    @classmethod
    def loss_function(cls, logits, label, targeted):
        """Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x)."""

        if targeted:
            adv_loss = - torch.gather(logits, 1, label)
        else:
            adv_loss = torch.gather(logits, 1, label)

        return adv_loss.mean()

    @staticmethod
    def best_other_class(logits, exclude):
        """Returns the index of the largest logit, ignoring the class that
        is passed as `exclude`."""
        y_onehot = torch.zeros_like(logits)
        y_onehot.scatter_(1, exclude, 1)
        # make logits that we want to exclude a large negative number
        other_logits = logits - y_onehot * 1e9
        return other_logits.max(1)[0]
