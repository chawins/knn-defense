'''Train MNIST model'''
from __future__ import print_function

import logging
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from lib.dataset_utils import *
from lib.dknn import DKNNL2
from lib.lip_model import *
from lib.mnist_model import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def evaluate(net, dataloader, device, config, adv=False):

    net.eval()
    val_loss = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if adv:
                outputs, outputs_adv = net.forward_adv(inputs, targets,
                                                       config['step_size'],
                                                       config['num_steps'],
                                                       config['random_start'])
                loss = net.loss_function(outputs_adv, targets, orig=outputs)
            else:
                outputs = net(inputs)
                loss = net.loss_function(outputs, targets)
            val_loss += loss.item()
            val_total += 1

    return val_loss / val_total


def train(net, trainloader, validloader, optimizer, epoch, device, log, config,
          save_best_only=True, best_loss=0, model_path='./model.pt'):

    net.train()
    train_loss = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, outputs_adv = net.forward_adv(inputs, targets,
                                               config['step_size'],
                                               config['num_steps'],
                                               config['random_start'])
        loss = net.loss_function(outputs_adv, targets, orig=outputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_total += 1

    adv_loss = evaluate(net, validloader, device, config, adv=True)
    val_loss = evaluate(net, validloader, device, config, adv=False)

    log.info(' %5d | %.4f | %8.4f | %8.4f', epoch,
             train_loss / train_total, adv_loss, val_loss)

    # Save model weights
    log.info('Saving model...')
    if (save_best_only and adv_loss < best_loss):
        torch.save(net.state_dict(), model_path + '.h5')
        best_loss = adv_loss
    else:
        torch.save(net.state_dict(), model_path + '_epoch{}.h5'.format(epoch))
    return best_loss


def predict(net, data, log):
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data
    layers = ['fc']
    dknn = DKNNL2(net, x_train, y_train, x_valid, y_valid, layers,
                  k=5, num_classes=10)
    with torch.no_grad():
        y_pred = dknn.classify(x_test)
        ind = np.where(y_pred.argmax(1) == y_test.numpy())[0]
        acc = (y_pred.argmax(1) == y_test.numpy()).sum() / y_test.size(0)
        log.info('accuracy: %.4f', acc)

    from lib.dknn_attack_exp import DKNNExpAttack
    attack = DKNNExpAttack(dknn)

    def attack_batch(x, y, batch_size):
        x_adv = torch.zeros_like(x)
        total_num = x.size(0)
        num_batches = total_num // batch_size
        for i in range(num_batches):
            begin = i * batch_size
            end = (i + 1) * batch_size
            x_adv[begin:end] = attack(
                x[begin:end], y[begin:end],
                guide_layer=layers[0], m=30, binary_search_steps=10,
                max_iterations=1000, learning_rate=1e-1,
                initial_const=1e0, random_start=False,
                thres_steps=200, check_adv_steps=200, verbose=False,
                max_linf=None)
        return x_adv

    num = 100
    x_adv = attack_batch(x_test[ind][:num].cuda(), y_test[ind][:num], 100)
    with torch.no_grad():
        y_pred = dknn.classify(x_adv)
        ind_adv = np.where(y_pred.argmax(1) != y_test[ind][:num].numpy())[0]
        adv_acc = (y_pred.argmax(1) == y_test[ind][:num].numpy()).sum() \
            / y_pred.shape[0]
        dist = (x_adv.cpu() - x_test[ind][:num]).view(
            num, -1).norm(2, 1)[ind_adv].mean()
        log.info('adv accuracy: %.4f, mean dist: %.4f', adv_acc, dist)


def main():

    # Set experiment id
    exp_id = 20
    model_name = 'adv_mnist_nca_exp%d' % exp_id

    # Training parameters
    batch_size = 128
    epochs = 200
    data_augmentation = False
    learning_rate = 1e-4
    l1_reg = 0
    l2_reg = 0

    output_dim = 100
    init_it = 1
    train_it = False

    # Adversarial training parameters
    config = {'num_steps': 30,
              'step_size': 0.1,
              'random_start': True}

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = False

    # Set all random seeds
    seed = 2019
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up model directory
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    # Get logger
    log_file = model_name + '.log'
    log = logging.getLogger('adv_mnist_nca')
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
              'data_augmentation: {}, subtract_pixel_mean: {}').format(
                  exp_id, seed, learning_rate, batch_size, l2_reg, l1_reg,
                  epochs, data_augmentation, subtract_pixel_mean))

    log.info('Preparing data...')
    trainloader, validloader, testloader = load_mnist(
        batch_size, data_dir='/data', val_size=0.1, shuffle=True, seed=seed)

    log.info('Building model...')
    net = NCAModel(output_dim=output_dim,
                   init_it=init_it, train_it=train_it)
    net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    optimizer = optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=l2_reg)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    data = load_mnist_all('/data', val_size=0.1, seed=seed)

    log.info(' epoch |   loss | adv_loss | val_loss')
    best_loss = 1e5
    for epoch in range(epochs):

        config = {'num_steps': epoch,
                  'step_size': 0.1,
                  'random_start': True}

        best_loss = train(net, trainloader, validloader, optimizer,
                          epoch, device, log, config, save_best_only=False,
                          best_loss=best_loss, model_path=model_path)
        predict(net, data, log)

    test_loss = evaluate(net, testloader, device, config, adv=True)
    log.info('Adv Test loss: %.4f', test_loss)
    test_loss = evaluate(net, testloader, device, config, adv=False)
    log.info('Test loss: %.4f', test_loss)


if __name__ == '__main__':
    main()
