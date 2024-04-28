import torch

#*****************************************************************************#
params = {
    'dataset': 'eyepacs_complete_tensors_augmented',
    'size': None,
    'augment': False,
    'use_scattering': False,
    'max_physical_batch_size': 256,
    'mini_batch_size': 1024,
    'lr': 0.001,
    'optim': "Adam",
    'momentum': 0.9,
    'nesterov': True,
    'noise_multiplier': 0,
    'max_grad_norm': 0.01,
    'epochs': 20,
    'input_norm': None,
    'num_groups': 0,
    'bn_noise_multiplier': 8,
    'max_epsilon': 3,
    'early_stop': True,
    'sample_batches': False,
    'logdir': None,
    'privacy': True,
    'weight_standardization': False,
    'ema_flag': False,
    'weight_decay': 1e-5,
    'grad_sample_mode': "no_op"
}

variable_parameters_dict = { # This variable dictionary can used to run batch tasks for multiple hyperparameters
    'input_norm' : ["BN"],
    'max_grad_norm' : [0.001, 0.01],
    'max_epsilon' : [3],
}