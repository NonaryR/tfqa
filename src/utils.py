import random
import numpy as np
import torch
import os
import ujson as json


def mklogs(args):
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
    directory = args.logs_dir
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f"{directory}/cli.args", "w") as outfile:
        json.dump(args, outfile, indent=4)

    return directory

