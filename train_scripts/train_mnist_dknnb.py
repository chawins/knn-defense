'''Train DkNNB on MNIST'''
from __future__ import print_function

import logging
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from lib.adv_model import *
from lib.dataset_utils import *
from lib.mnist_model import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def loss_function(y_pred, y_train, y_knn):
    pass


def evaluate(net, dataloader, device):

    dknnb.dknnb_net.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs, targets, attack=adv, clip=True)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    return val_loss / val_total, val_correct / val_total


def train(dknnb, trainloader, validloader, optimizer, epoch, device, log,
          save_best_only=True, best_loss=1e9, model_path='./model.pt'):

    dknnb.dknnb_net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, targets, y_knn)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

    adv_loss, adv_acc = evaluate(net, validloader, criterion, device, adv=True)
    val_loss, val_acc = evaluate(
        net, validloader, criterion, device, adv=False)

    log.info(' %5d | %.4f, %.4f | %.4f, %.4f | %.4f, %.4f | ', epoch,
             train_loss / train_total, train_correct / train_total,
             adv_loss, adv_acc, val_loss, val_acc)

    # Save model weights
    if not save_best_only:
        log.info('Saving model...')
        torch.save(net.state_dict(), model_path + '_epoch%d.h5' % epoch)
    elif save_best_only and adv_loss < best_loss:
        log.info('Saving model...')
        torch.save(net.state_dict(), model_path + '.h5')
        best_loss = adv_loss
    return best_loss


def main():

    # Set experiment id
    exp_id = 0
    dknnb_name = 'dknnb_mnist_exp%d' % exp_id

    # Define baet network for DkNN
    basenet_name = 'adv_mnist_exp20'
    layers = ['fc']     # supports only one layer
    k = 5

    # Training parameters
    batch_size = 128
    epochs = 70
    data_augmentation = False
    learning_rate = 1e-3
    l1_reg = 0
    l2_reg = 0

    # Set all random seeds
    seed = 2019
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up model directory
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, dknnb_name)
    basenet_path = os.path.join(save_dir, basenet_name)

    # Get logger
    log_file = dknnb_name + '.log'
    log = logging.getLogger('train_mnist')
    log.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s %(name)s] %(message)s')
    # Create file handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    log.info(log_file)
    log.info(('MNIST | exp_id: {}, seed: {}, init_learning_rate: {}, ' +
              'batch_size: {}, l2_reg: {}, l1_reg: {}, epochs: {}, ' +
              'data_augmentation: {}').format(
                  exp_id, seed, learning_rate, batch_size, l2_reg, l1_reg,
                  epochs, data_augmentation))

    log.info('Loading base network...')
    # Load the base network that produces embeddings
    base_net = BasicModel()
    base_net.load_state_dict(torch.load(basenet_path))
    base_net = base_net.to(device)
    base_net.eval()

    log.info('Preparing data...')
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_all(
        data_dir='/data', val_size=0.1, shuffle=True, seed=seed)
    dknn = DkNNL2(base_net, x_train, y_train, x_val[:500], y_val[:500], layers,
                  k=k, num_classes=10, device='cuda')

    y_train_dknn = dknn.classify(x_train)
    y_val_dknn = dknn.classify(x_val)
    y_test_dknn = dknn.classify(x_test)

    trainset = torch.utils.data.TensorDataset(x_train, (y_train, y_train_dknn))
    validset = torch.utils.data.TensorDataset(x_val, (y_val, y_val_dknn))
    testset = torch.utils.data.TensorDataset(x_test, (y_test, y_test_dknn))
    num_workers = 4
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    log.info('Building model...')
    dknnb_net = DkNNBModel()
    dknnb_net = dknnb_net.to(device)
    dknnb = DkNNB(base_net, dknnb_net)

    optimizer = optim.Adam(dknnb_net.parameters(), lr=learning_rate)

    log.info(' epoch | loss  , acc    | val_l , val_a  |')
    best_loss = 1e9

    for epoch in range(epochs):

        best_loss = train(dknnb, trainloader, validloader, optimizer, epoch,
                          device, log, save_best_only=True,
                          best_loss=best_loss, model_path=model_path)

    test_loss, test_acc = evaluate(dknnb, testloader, device)
    log.info('Test loss: %.4f, Test acc: %.4f', test_loss, test_acc)


if __name__ == '__main__':
    main()
