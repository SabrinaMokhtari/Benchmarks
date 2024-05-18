import argparse
import os
from sklearn.model_selection import ParameterGrid

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from sklearn.linear_model import LogisticRegression

from train_utils import CheXpert_test, CheXpert_train, get_device, train, test
from data import get_data, get_scatter_transform, get_scattered_loader
from models import ScatterLinear, get_num_params
from dp_utils import ORDERS, get_privacy_spent, get_renyi_divergence, scatter_normalization
from log import Logger

from torch_ema import ExponentialMovingAverage
import math
import CONFIG

def non_private_main(dataset, augment=False, use_scattering=False, size=None,
         mini_batch_size=256, sample_batches=False,
         lr=1, optim="Adam", momentum=0.9, nesterov=False,
         noise_multiplier=1, max_grad_norm=0.1, epochs=100,
         input_norm=None, num_groups=None, bn_noise_multiplier=None,
         max_epsilon=None, logdir=None, early_stop=True,
         max_physical_batch_size=128, weight_decay=1e-5, privacy=True,
         weight_standardization=True, ema_flag=True, write_file=None, 
         grad_sample_mode=None):

    logger = Logger(logdir)
    device = get_device()

    train_data, test_data = get_data(dataset, augment=augment)
    scattering, K, (h, w) = None, 243, (56, 56)

    n = len(train_data)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=mini_batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=mini_batch_size, shuffle=False, num_workers=0, pin_memory=False)

    rdp_norm = 0
    if input_norm == "BN":
        # compute noisy data statistics or load from disk if pre-computed
        save_dir = f"bn_stats/{dataset}"
        os.makedirs(save_dir, exist_ok=True)
        bn_stats, rdp_norm = scatter_normalization(train_loader,
                                                   scattering,
                                                   K,
                                                   device,
                                                   len(train_data),
                                                   len(train_data),
                                                   noise_multiplier=bn_noise_multiplier,
                                                   orders=ORDERS,
                                                   save_dir=save_dir)
        model = ScatterLinear(K, (h, w), input_norm="BN", classes = 5, bn_stats=bn_stats)
    else:
        model = ScatterLinear(K, (h, w), input_norm=input_norm, classes=5, num_groups=num_groups)

    model.to(device)

    # baseline Logistic Regression without privacy
    if optim == "LR":
        assert not augment
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for data, target in train_loader:
            with torch.no_grad():
                data = data.to(device)
                X_train.append(data.cpu().numpy().reshape(len(data), -1))
                y_train.extend(target.cpu().numpy())

        for data, target in test_loader:
            with torch.no_grad():
                data = data.to(device)
                X_test.append(data.cpu().numpy().reshape(len(data), -1))
                y_test.extend(target.cpu().numpy())

        import numpy as np
        X_train = np.concatenate(X_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        for idx, C in enumerate([0.01, 0.1, 1.0, 10, 100]):
            clf = LogisticRegression(C=C, fit_intercept=True)
            clf.fit(X_train, y_train)

            train_acc = 100 * clf.score(X_train, y_train)
            test_acc = 100 * clf.score(X_test, y_test)
            print(f"C={C}, "
                  f"Acc train = {train_acc: .2f}, "
                  f"Acc test = {test_acc: .2f}")

            logger.log_epoch(idx, 0, train_acc, 0, test_acc, None)
        return

    print(f"model has {get_num_params(model)} parameters")

    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    nesterov=nesterov)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        

    best_acc = 0
    flat_count = 0

    if ema_flag:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    else:
        ema = None

    write_file=f"/u2/s4mokhta/outputs/linear-non-private/{dataset}_epochs{epochs}_privacy{privacy}_Optim{optim}_LR{lr}_WS{weight_standardization}_EMA{ema_flag}_BatchSize{mini_batch_size}_input_norm{input_norm}_num_groups{num_groups}.txt"
    result_file = open(write_file, 'a')
    result_file.write(f'Minibatch size: {mini_batch_size}\nLearning Rate: {lr}\nOptimizer: {optim}\n')
    result_file.write(f'Epochs: {epochs}\nPrivacy: {privacy}\nMaximum Epsilon: {max_epsilon}\n'
                      f'Max Physical Batch Size: {max_physical_batch_size}\n')


    for epoch in range(0, epochs):
        print(f"\nEpoch: {epoch}")

        if dataset == 'chexpert_tensors' or dataset == 'chexpert_tensors_augmented':
            train_loss, train_acc, train_val_auc_mean = CheXpert_train(model, train_loader, optimizer, ema, max_physical_batch_size=max_physical_batch_size, grad_sample_mode=grad_sample_mode)
            test_loss, test_acc, test_val_auc_mean, ema_test_loss, ema_test_acc, ema_test_val_auc_mean = CheXpert_test(model, test_loader, ema)
        elif dataset == 'eyepacs_complete_tensors' or dataset == 'eyepacs_complete_tensors_augmented':
            train_loss, train_acc, train_val_auc_mean = train(model, train_loader, optimizer, ema, max_physical_batch_size=max_physical_batch_size, grad_sample_mode=grad_sample_mode)
            test_loss, test_acc, test_val_auc_mean, ema_test_loss, ema_test_acc, ema_test_val_auc_mean = test(model, test_loader, ema)

        result_file.write(f'Train set: Average loss: {train_loss}, Accuracy: {train_acc}, AUC: {train_val_auc_mean}\n')
        result_file.write(f'Test set: Average loss: {test_loss}, Accuracy: {test_acc}, AUC: {test_val_auc_mean}\n')
        result_file.write(f'EMA Test set: Average loss: {ema_test_loss}, Accuracy: {ema_test_acc}, AUC: {ema_test_val_auc_mean}\n')

def main(dataset, augment=False, use_scattering=False, size=None,
         mini_batch_size=256, sample_batches=False,
         lr=1, optim="Adam", momentum=0.9, nesterov=False,
         noise_multiplier=1, max_grad_norm=0.1, epochs=100,
         input_norm=None, num_groups=None, bn_noise_multiplier=None,
         max_epsilon=None, logdir=None, early_stop=True,
         max_physical_batch_size=128, weight_decay=1e-5, privacy=True,
         weight_standardization=True, ema_flag=True, write_file=None,
         grad_sample_mode=None):

    logger = Logger(logdir)
    device = get_device()

    train_data, test_data = get_data(dataset, augment=augment)
    scattering, K, (h, w) = None, 243, (56, 56)

    n = len(train_data)
    DELTA = 10 ** -(math.ceil(math.log10(n)))
    print("DELTA: ", DELTA)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=mini_batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=mini_batch_size, shuffle=False, num_workers=0, pin_memory=False)

    rdp_norm = 0
    if input_norm == "BN":
        # compute noisy data statistics or load from disk if pre-computed
        save_dir = f"bn_stats/{dataset}"
        os.makedirs(save_dir, exist_ok=True)
        bn_stats, rdp_norm = scatter_normalization(train_loader,
                                                   scattering,
                                                   K,
                                                   device,
                                                   len(train_data),
                                                   len(train_data),
                                                   noise_multiplier=bn_noise_multiplier,
                                                   orders=ORDERS,
                                                   save_dir=save_dir)
        model = ScatterLinear(K, (h, w), input_norm="BN", classes = 5, bn_stats=bn_stats)
    else:
        model = ScatterLinear(K, (h, w), input_norm=input_norm, classes=5, num_groups=num_groups)

    model.to(device)

    # baseline Logistic Regression without privacy
    if optim == "LR":
        assert not augment
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for data, target in train_loader:
            with torch.no_grad():
                data = data.to(device)
                X_train.append(data.cpu().numpy().reshape(len(data), -1))
                y_train.extend(target.cpu().numpy())

        for data, target in test_loader:
            with torch.no_grad():
                data = data.to(device)
                X_test.append(data.cpu().numpy().reshape(len(data), -1))
                y_test.extend(target.cpu().numpy())

        import numpy as np
        X_train = np.concatenate(X_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        for idx, C in enumerate([0.01, 0.1, 1.0, 10, 100]):
            clf = LogisticRegression(C=C, fit_intercept=True)
            clf.fit(X_train, y_train)

            train_acc = 100 * clf.score(X_train, y_train)
            test_acc = 100 * clf.score(X_test, y_test)
            print(f"C={C}, "
                  f"Acc train = {train_acc: .2f}, "
                  f"Acc test = {test_acc: .2f}")

            logger.log_epoch(idx, 0, train_acc, 0, test_acc, None)
        return

    print(f"model has {get_num_params(model)} parameters")

    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    nesterov=nesterov)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    privacy_engine = PrivacyEngine(accountant='prv')
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=max_epsilon,
        target_delta=DELTA,
        max_grad_norm=max_grad_norm,
    )

    noise_multiplier = optimizer.noise_multiplier # change it to for private experiments optimizer.noise_multiplier o/w 0
    print(f"Using sigma={optimizer.noise_multiplier} and C={max_grad_norm}")

    best_acc = 0
    flat_count = 0

    if ema_flag:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    else:
        ema = None

    write_file=f"/u2/s4mokhta/outputs/linear/{dataset}_epochs{epochs}_privacy{privacy}_max_epsilon{max_epsilon}_DELTA{DELTA}_max_grad_norm{max_grad_norm}_Optim{optim}_LR{lr}_WS{weight_standardization}_EMA{ema_flag}_BatchSize{mini_batch_size}_input_norm{input_norm}_num_groups{num_groups}.txt"
    result_file = open(write_file, 'a')
    result_file.write(f'Minibatch size: {mini_batch_size}\nLearning Rate: {lr}\nOptimizer: {optim}\n')
    result_file.write(f'Epochs: {epochs}\nPrivacy: {privacy}\nMaximum Epsilon: {max_epsilon}\nDelta: {DELTA}\n'
                      f'Clip norm: {max_grad_norm}\n, Noise Multiplier: {noise_multiplier}\n, Max Physical Batch Size: {max_physical_batch_size}\n')


    for epoch in range(0, epochs):
        print(f"\nEpoch: {epoch}")

        if dataset == 'chexpert_tensors' or dataset == 'chexpert_tensors_augmented':
            train_loss, train_acc, train_val_auc_mean = CheXpert_train(model, train_loader, optimizer, ema, max_physical_batch_size=max_physical_batch_size, grad_sample_mode=grad_sample_mode)
            test_loss, test_acc, test_val_auc_mean, ema_test_loss, ema_test_acc, ema_test_val_auc_mean = CheXpert_test(model, test_loader, ema)
        elif dataset == 'eyepacs_complete_tensors' or dataset == 'eyepacs_complete_tensors_augmented':
            train_loss, train_acc, train_val_auc_mean = train(model, train_loader, optimizer, ema, max_physical_batch_size=max_physical_batch_size, grad_sample_mode=grad_sample_mode)
            test_loss, test_acc, test_val_auc_mean, ema_test_loss, ema_test_acc, ema_test_val_auc_mean = test(model, test_loader, ema)

        if noise_multiplier > 0:
            
            print(f"privacy engine: {privacy_engine.accountant.history}")
            epsilon = privacy_engine.get_epsilon(delta=DELTA)
            print(f"ε = {epsilon:.2f}")

            if max_epsilon is not None and epsilon >= max_epsilon:
                return
        else:
            epsilon = None

        result_file.write(f'Train set: Average loss: {train_loss}, Accuracy: {train_acc}, AUC: {train_val_auc_mean}\n')
        result_file.write(f'Test set: Average loss: {test_loss}, Accuracy: {test_acc}, AUC: {test_val_auc_mean}\n')
        result_file.write(f'EMA Test set: Average loss: {ema_test_loss}, Accuracy: {ema_test_acc}, AUC: {ema_test_val_auc_mean}\n')
        result_file.write(f'Epsilon without considering BN: {epsilon}\n')

if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['chexpert_tensors', 'eyepacs_complete_tensors'])
    parser.add_argument('--mini_batch_size', type=int, default=1024)
    parser.add_argument('--input_norm', default=None, choices=["GroupNorm", "BN"])
    parser.add_argument('--max_grad_norm', nargs='+', type=float, help='List of floats')
    parser.add_argument('--num_groups', nargs='+', type=int, help='List of integers')
    parser.add_argument('--max_epsilon', type=int, default=3)
    args = parser.parse_args()

    # Access the list argument
    num_groups = args.num_groups
    max_grad_norm = args.max_grad_norm
    input_norm = args.input_norm
    dataset = args.dataset
    mini_batch_size = args.mini_batch_size
    max_epsilon = args.max_epsilon
    
    params = CONFIG.params

    variables_dict = CONFIG.variable_parameters_dict

    variables_dict['num_groups'] = num_groups
    variables_dict['max_grad_norm'] = max_grad_norm 
    variables_dict['input_norm'] = [input_norm]
    variables_dict['dataset'] = [dataset]
    variables_dict['mini_batch_size'] = [mini_batch_size]
    variables_dict['max_epsilon'] = [max_epsilon]

    variables = list(ParameterGrid(variables_dict))
    
    for i in range(3):
        for conf in variables:
            params.update(conf)

            dataset = params['dataset']

            print(params)

            privacy = params['privacy']
            if privacy:
                print("Running private experiment")
                main(**params)
            else:
                print("Running non-private experiment")
                non_private_main(**params)