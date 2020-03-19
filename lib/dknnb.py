import torch

from lib.dataset_utils import load_mnist_all


class DkNNB(torch.Modules):

    def __init__(self, base_net, dknnb_net):
        self.base_net = base_net
        self.dknnb_net = dknnb_net

    def forward(self, x):
        u = self.base_net(x)
        u = self.dknnb_net(x)
        return u


def load_mnist(base_net, batch_size, data_dir='./data', val_size=0.1,
               shuffle=True, seed=1):
    """Load MNIST data into train/val/test data loader along with the soft
    labels obtained from DkNN"""

    num_workers = 4

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist_all(
        data_dir=data_dir, val_size=val_size, shuffle=shuffle, seed=seed)

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    validset = torch.utils.data.TensorDataset(x_valid, y_valid)
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader
