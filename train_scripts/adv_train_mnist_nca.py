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
from lib.dknn_attack_v2 import DKNNAttackV2
from lib.lip_model import *
from lib.mnist_model import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def evaluate(net, dataloader, device, params, adv=False):

    net.eval()
    val_loss = 0
    val_total = 0
    val_cor = 0

    net.recompute_train_rep()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if adv:
                _, outputs = net.forward_adv(inputs, targets, params)
            else:
                outputs = net(inputs)
            probs = net.compute_logits(outputs, from_outputs=True).cpu()
            prob = (probs * torch.eye(net.num_classes)[targets]).sum(1)
            val_loss -= prob.log().sum()
            val_cor += (probs.argmax(1) == targets.cpu()).float().sum().item()
            val_total += len(inputs)

    return val_loss / val_total, val_cor / val_total


def train(net, trainloader, validloader, optimizer, epoch, device, log, params,
          save_best_only=True, best_loss=0, model_path='./model.pt'):

    net.train()
    train_loss = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, outputs_adv = net.forward_adv(inputs, targets, params)
        loss = net.loss_function(outputs_adv, targets, orig=outputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_total += 1

    adv_loss, adv_acc = evaluate(net, validloader, device, params, adv=True)
    val_loss, val_acc = evaluate(net, validloader, device, params, adv=False)

    log.info(' %5d | %.4f | %.4f | %.4f | %.4f | %.4f',
             epoch, train_loss / train_total, adv_loss, adv_acc, val_loss,
             val_acc)

    # Save model weights
    if not save_best_only:
        log.info('Saving model...')
        torch.save(net.state_dict(), model_path + '_epoch{}.h5'.format(epoch))
    elif adv_loss < best_loss:
        log.info('Saving model...')
        torch.save(net.state_dict(), model_path + '.h5')
        best_loss = adv_loss
    return best_loss


def attack_batch(attack, x, y, init_mode, init_mode_k, batch_size):
    x_adv = torch.zeros_like(x)
    total_num = x.size(0)
    num_batches = total_num // batch_size
    for i in range(num_batches):
        begin = i * batch_size
        end = (i + 1) * batch_size
        x_adv[begin:end] = attack(
            x[begin:end], y[begin:end], 2, guide_layer='fc', m=40,
            init_mode=init_mode, init_mode_k=init_mode_k,
            binary_search_steps=10, max_iterations=1000, learning_rate=1e-1,
            initial_const=1e0, max_linf=None, random_start=False,
            thres_steps=200, check_adv_steps=200, verbose=False)
    return x_adv


def predict(net, data, log):
    (x_train, y_train), (x_valid, y_valid), (_, _) = data
    layers = ['fc']
    dknn = DKNNL2(net, x_train, y_train, x_valid, y_valid, layers,
                  k=5, num_classes=10)
    with torch.no_grad():
        y_pred = dknn.classify(x_valid)
        ind = np.where(y_pred.argmax(1) == y_valid.numpy())[0]
        acc = (y_pred.argmax(1) == y_valid.numpy()).sum() / y_valid.size(0)
        log.info('accuracy: %.4f', acc)

    # attack = DKNNAttackV2(dknn)
    # num = 100
    # x_adv = attack_batch(
    #     attack, x_valid[ind][:num].cuda(), y_valid[ind][:num], 1, 1, 100)
    # with torch.no_grad():
    #     y_pred = dknn.classify(x_adv)
    #     ind_adv = np.where(y_pred.argmax(1) != y_valid[ind][:num].numpy())[0]
    #     adv_acc = (y_pred.argmax(1) == y_valid[ind][:num].numpy()).sum() \
    #         / y_pred.shape[0]
    #     dist = (x_adv.cpu() - x_valid[ind][:num]).view(
    #         num, -1).norm(2, 1)[ind_adv].mean()
    #     log.info('adv accuracy: %.4f, mean dist: %.4f', adv_acc, dist)


def main():

    # Set experiment id
    exp_id = 81
    model_name = 'adv_mnist_nca_exp%d' % exp_id

    # Training parameters
    batch_size = 128
    epochs = 100
    data_augmentation = False
    learning_rate = 1e-4
    l1_reg = 0
    l2_reg = 0

    output_dim = 20
    init_it = 5
    train_it = True

    eps_final = 3
    eps_init = 0
    eps_warmup_epoch = 0

    # Adversarial training parameters
    params = {'epsilon': eps_final,
              'num_steps': 40,
              'step_size': 0.2,
              'random_start': True}

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = False

    # Set all random seeds
    seed = 2020
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
    train_data, _, _ = load_mnist_all(
        '/data', val_size=0.1, shuffle=True, seed=seed)

    log.info('Building model...')
    net = NCAModelV3(
        normalize=False, output_dim=output_dim, init_it=init_it,
        train_it=train_it, train_data=train_data)
    # net = SoftLabelNCA(
    #     ys_train=torch.zeros((len(train_data[0]), 10), device=device),
    #     normalize=False, output_dim=output_dim, init_it=init_it,
    #     train_it=train_it, train_data=train_data)
    net = net.to(device)
    # net.recompute_ys_train(100)
    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=l2_reg)
    # optimizer = optim.SGD(
    #     net.parameters(), lr=learning_rate, weight_decay=l2_reg)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [80, 90], gamma=0.1)

    log.info(' epoch |   loss |  adv_l |  adv_a |  val_l |  val_a')
    best_loss = 1e5
    for epoch in range(epochs):

        # if epoch < eps_warmup_epoch:
        #     eps = (eps_final - eps_init) * epoch / eps_warmup_epoch + eps_init
        # else:
        #     eps = eps_final
        # params['epsilon'] = eps

        # if epoch < 20:
        #     params['num_steps'] = 0
        # else:
        #     params['num_steps'] = 0
        # params['epsilon'] = 0
        # elif epoch < 50:
        #     params['epsilon'] = (eps_final - eps_init) * epoch / eps_warmup_epoch + eps_init

        best_loss = train(net, trainloader, validloader, optimizer,
                          epoch, device, log, params, save_best_only=True,
                          best_loss=best_loss, model_path=model_path)
        lr_scheduler.step()
        # predict(net, data, log)

    test_loss, test_acc = evaluate(net, testloader, device, params, adv=True)
    log.info('Adv test loss: %.4f, Adv test acc: %.4f', test_loss, test_acc)
    test_loss, test_acc = evaluate(net, testloader, device, params, adv=False)
    log.info('Test loss: %.4f, Test acc: %.4f', test_loss, test_acc)


if __name__ == '__main__':
    main()
