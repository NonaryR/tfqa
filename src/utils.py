import random
import numpy as np
import torch
import os
import ujson as json
import spacy_udpipe
from base import NUMERIC_RE, NUMERIC_TOKEN



def mklogs(args):
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_printoptions(precision=10)    

    directory = args.logs_dir
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f"{directory}/cli.args", "w") as outfile:
        json.dump(args, outfile, indent=4)

    return directory

def print_grad_stats(model):
    mean = 0
    std = 0
    norm = 1e-5
    for param in model.parameters():
        grad = getattr(param, 'grad', None)
        if grad is not None:
            mean += grad.data.abs().mean()
            std += grad.data.std()
            norm += 1
    mean /= norm
    std /= norm
    print(f'Mean grad {mean}, std {std}, n {norm}')


def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Недопустимый тип данных {}'.format(type(data)))