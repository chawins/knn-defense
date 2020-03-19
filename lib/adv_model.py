import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDModel(nn.Module):
    """
    code adapted from
    https://github.com/karandwivedi42/adversarial/blob/master/main.py
    """

    def __init__(self, basic_net, config):
        super(PGDModel, self).__init__()
        self.basic_net = basic_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        assert config['loss_func'] == 'xent', 'Only xent supported for now.'

    def forward(self, inputs, targets, attack=False):
        if not attack:
            return self.basic_net(inputs)

        x = inputs.clone()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for _ in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_net(x)
                loss = F.cross_entropy(logits, targets, reduction='sum')
            grad = torch.autograd.grad(loss, x)[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs.detach() - self.epsilon),
                          inputs.detach() + self.epsilon)
            x = torch.clamp(x, 0, 1)

        return self.basic_net(x)


class PGDL2Model(nn.Module):
    """
    code adapted from
    https://github.com/karandwivedi42/adversarial/blob/master/main.py
    """

    def __init__(self, basic_net, config):
        super(PGDL2Model, self).__init__()
        self.basic_net = basic_net
        self.epsilon = config['epsilon']
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.num_steps = config['num_steps']
        assert config['loss_func'] == 'xent', 'Only xent supported for now.'

    def forward(self, inputs, targets, attack=False):
        if not attack:
            return self.basic_net(inputs)

        x = inputs.clone()
        if self.rand:
            x = x + torch.zeros_like(x).normal_(0, self.step_size)

        for _ in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_net(x)
                loss = F.cross_entropy(logits, targets, reduction='sum')
            grad = torch.autograd.grad(loss, x)[0].detach()
            grad_norm = grad.view(x.size(0), -1).norm(2, 1)
            delta = self.step_size * grad / grad_norm.view(x.size(0), 1, 1, 1)
            x = x.detach() + delta
            diff = (x - inputs).view(x.size(0), -1).renorm(2, 0, self.epsilon)
            x = diff.view(x.size()) + inputs
            x.clamp_(0, 1)

        return self.basic_net(x)
