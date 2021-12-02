import os

import RNA
import numpy as np
import torch

from rl_lib.environment import Env_RNA
from utils.config import action_space
from utils.rna_lib import get_subgraph_exist

root = os.path.dirname(os.path.realpath(__file__))

# load data
data_dir = root + '/data/raw/rfam_learn/train/1.rna'
dotB_list = []
f = open(data_dir)
iter_f = iter(f)
for line in iter_f:
    line = line.replace('\n', '')
    dotB_list.append(line)

init_len = 3

do_skip = True

env = Env_RNA(dotB_list=dotB_list, action_space=action_space, h_weight=2, do_skip=do_skip)

env.reset(init_len=init_len)

get_subgraph_exist(env.graphs[0], 3)


