import os
from os.path import join, dirname
import torch

### PATH CONFIGS ###

_root_directory = dirname(__file__)

path_configs = {'root_dir': _root_directory,
                'data_dir': join(_root_directory, 'data'),
                'results_dir': join(_root_directory, 'results')
                }


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


safe_mkdir(join(path_configs['root_dir'], 'data'))
safe_mkdir(join(path_configs['root_dir'], 'results'))

### HARDWARE CONFIGS ###

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    _device = 0
else:
    _device = 'cpu'

# _device = 'cpu'  # TODO: debug CUDA
machine_configs = {'device': _device}


def move(x):
    return x.to(_device)


### EXPERIMENT 1 CONFIGS ###

exp1_hp = {'batch_size': 25,
           'learning_rate': 10 ** (-5),  # default
           'num_epochs': 150,  # default
           'base_loss_fn': torch.nn.CrossEntropyLoss,
           'optimizer': torch.optim.Adam,
           }

### EXPERIMENT 2 CONFIGS ###

exp2_hp = {'batch_size': 25,
           'learning_rate': 10 ** (-5),  # default
           'num_epochs': 100,  # default
           'base_loss_fn': torch.nn.CrossEntropyLoss,
           'optimizer': torch.optim.Adam,
           }
