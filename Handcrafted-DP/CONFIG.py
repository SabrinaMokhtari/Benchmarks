import torch
# File for experiment configuration that is used to setup the for loops for
# mnist_test.py and plot_scripts that work on the results and logs generated
# by running mnist_test.py

# Set RESULTS_DIR to a new folder for a clean experiment
# (or rm -rf the RESULTS_DIR folder, so that previous results don't bash new
# results, since the log files get appended.)

#*****************************************************************************#
params = {
    'dataset': 'chexpert_tensors',
    'size': None,
    'augment': False,
    'use_scattering': False,
    'max_physical_batch_size': 128,
    'mini_batch_size': 1024,
    'lr': 0.001,
    'optim': "Adam",
    'momentum': 0.9,
    'nesterov': True,
    'noise_multiplier': 0,
    'max_grad_norm': 0.01,
    'epochs': 30,
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
    'grad_sample_mode': None,
}

variable_parameters_dict = { # This variable dictionary can used to run batch tasks for multiple hyperparameters
    'input_norm' : [None],
    'num_groups': [0],
    'max_grad_norm': [0.001, 0.01, 0.1, 1, 10],
}
#   'max_epsilon' : [1, 3, 5, 8],
#   'num_groups' : [9, 27, 81],
#   'max_grad_norm' : [0.001, 0.01, 1, 10],
#   'lr' : [0.0001, 0.0005, 0.001, 1, 4],
#   'lr' : [0.0005, 0.001, 0.005, 0.01],