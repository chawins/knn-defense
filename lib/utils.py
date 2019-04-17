import numpy as np
import torch


def compute_lid(x, x_train, k, exclude_self=False):
    """
    Calculate LID using the estimation from [1]

    [1] Ma et al., "Characterizing Adversarial Subspaces Using
        Local Intrinsic Dimensionality," ICLR 2018.
    """

    with torch.no_grad():
        x = x.view((x.size(0), -1))
        x_train = x_train.view((x_train.size(0), -1))
        lid = torch.zeros((x.size(0), ))

        for i, x_cur in enumerate(x):
            dist = (x_cur.view(1, -1) - x_train).norm(2, 1)
            # `largest` should be True when using cosine distance
            if exclude_self:
                topk_dist = dist.topk(k + 1, largest=False)[0][1:]
            else:
                topk_dist = dist.topk(k, largest=False)[0]
            mean_log = torch.log(topk_dist / topk_dist[-1]).mean()
            lid[i] = -1 / mean_log
        return lid


# def cal_class_lid(x, x_train, k, exclude_self=False):
#     """
#     Calculate LID on sample using the estimation from [1]

#     [1] Ma et al., "Characterizing Adversarial Subspaces Using
#         Local Intrinsic Dimensionality," ICLR 2018.
#     """

#     x = x.view((x.size(0), -1))
#     x_train = x_train.view((x_train.size(0), -1))
#     lid = torch.zeros((x.size(0), ))

#     for i, x_cur in enumerate(x):
#         dist = (x_cur.view(1, -1) - x_train).norm(2, 1)
#         # `largest` should be True when using cosine distance
#         if exclude_self:
#             topk_dist = dist.topk(k + 1, largest=False)[0][1:]
#         else:
#             topk_dist = dist.topk(k, largest=False)[0]
#         mean_log = torch.log(topk_dist / topk_dist[-1]).mean()
#         lid[i] = -1 / mean_log
#     return lid


def compute_spnorm(inputs, dknn, layers, batch_size=200):

    assert inputs.requires_grad

    num_total = inputs.size(0)
    norm = np.zeros((num_total, len(layers)))
    num_batches = int(np.ceil(num_total / batch_size))

    for i in range(num_batches):
        begin, end = i * batch_size, (i + 1) * batch_size
        x = inputs[begin:end]
        reps = dknn.get_activations(x)
        for l, layer in enumerate(layers):
            y = reps[layer]
            norm[begin:end, l] = compute_spnorm_batch(x, y)

    return norm


def compute_spnorm_batch(inputs, output):
    """
    :param inputs: (batch_size, input_size)
    :param output: (batch_size, output_size)
    :return: jacobian: (batch_size, output_size, input_size)
    """

    batch_size, input_dim = inputs.view(inputs.size(0), -1).size()
    output = output.view(batch_size, -1)
    jacobian = torch.zeros((batch_size, output.size(1), input_dim))
    for i in range(output.size(1)):
        grad = torch.autograd.grad(
            output[:, i].sum(), inputs, retain_graph=True)[0]
        jacobian[:, i, :] = grad.view(batch_size, input_dim)

    norm = np.zeros((batch_size, ))
    for i in range(batch_size):
        norm[i] = np.linalg.norm(jacobian[i].detach().cpu().numpy(), 2)

    return norm
