import torch
import torch_geometric
import RNA
from utils.rna_lib import init_graph, get_graph, act
import pathos.multiprocessing as pathos_mp
import gym
from functools import partial
import numpy as np


def fetch_items(list, index):
    """
    fetch the elements from the list with the index of aims
    :param list: original list
    :param index: index of aims
    :return: the aims
    """
    return [list[i] for i in index]


class Env_RNA(gym.Env):
    def __init__(self, dotB_list, action_space, h_weight,pool=None):
        super(Env_RNA, self).__init__()
        if pool is None:
            self.pool = pathos_mp.ProcessPool()
        else:
            self.pool = pool
        self.dotB_list = dotB_list
        self.graphs = []
        self.action_space = action_space
        self.h_weight = h_weight
        self.id_list = list(range(len(dotB_list)))

    def reset(self, init_len=1):
        gen_work = partial(get_graph, h_weight=self.h_weight)
        self.graphs = self.pool.map(gen_work, self.dotB_list)
        init_work = partial(init_graph, init_len=init_len, action_space=self.action_space)
        self.graphs = self.pool.map(init_work, self.graphs)

        return torch_geometric.data.Batch.from_data_list(self.graphs).clone()

    def step(self, actions, ep):
        step_work = partial(act, place=ep, action_space=self.action_space)
        self.graphs = self.pool.map(step_work, self.graphs, actions)
        finished_list = [0] * len(self.graphs)

    def remove_graph(self, finish_id_list):
        finish_index = np.argwhere(np.array(self.id_list) in finish_id_list)
        remove_id_list = finish_id_list
        remove_graph_list = fetch_items(self.graphs, finish_index)
        self.graphs = [self.graphs[i] for i in range(len(self.graphs)) if i not in finish_index]
        self.id_list = [self.id_list[i] for i in range(len(self.id_list)) if i not in finish_index]
        return remove_id_list, remove_graph_list



