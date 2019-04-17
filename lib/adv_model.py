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

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for _ in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_net(x)
                loss = F.cross_entropy(logits, targets, reduction='sum')
            grad = torch.autograd.grad(loss, x)[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon),
                          inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)

        return self.basic_net(x)
