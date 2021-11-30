import torch
import argparse

parser = argparse.ArgumentParser()

# cuda 的 GPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 动作空间
action_space = 4

# backbone 参数
backboneParam_dict = {
    'in_size': 4,
    'out_size': 16,
    'hide_size_list': [8],
    'normalize': True,
    'lr': 0.000001
}

for key, value in backboneParam_dict.items():
    parser.add_argument(key, action='store_const', const=value)

backboneParam = parser.parse_args()

# actor 参数
actorParam_dict = {
    'in_size': 16,
    'out_size': action_space,
    'hide_size_list': [8],
    'n_gcn': 0,
    'normalize': True,
    'lr': 0.000001
}

for key, value in actorParam_dict.items():
    parser.add_argument(key, action='store_const', const=value)

actorParam = parser.parse_args()

# critic 参数
criticParam_dict = {
    'in_size': 16,
    'out_size': 1,
    'hide_size_list': [8, 4],
    'n_gcn': 1,
    'normalize': True,
    'lr': 0.000001
}

for key, value in actorParam_dict.items():
    parser.add_argument(key, action='store_const', const=value)

criticParam = parser.parse_args()