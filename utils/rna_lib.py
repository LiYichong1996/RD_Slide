import numpy as np
import torch
import torch_geometric
from tqdm import tqdm
from random import choice
import multiprocessing as mp
import random
import RNA


#############################################################
# 全局常量
#############################################################

base_color_dict = {'A': 'y', 'U': 'b', 'G': 'r', 'C': 'g'}
base_list = ['A', 'U', 'C', 'G']
onehot_list = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])]
base_pair_dict_6 = {'A': ['U'], 'U': ['A', 'G'], 'G': ['U', 'C'], 'C': ['G'] }
base_pair_dict_4 = {'A': ['U'], 'U': ['A'], 'C': ['G'], 'G': ['C']}
base_pair_list_6 = [['A', 'U'], ['U', 'A'], ['U', 'G'], ['G', 'U'], ['G', 'C'], ['C', 'G']]
base_pair_list_4 = [['A', 'U'], ['U', 'A'], ['C', 'G'], ['G', 'C']]


#############################################################
# 数据结构和工具
#############################################################

class Stack(object):
    """栈"""
    def __init__(self):
         self.items = []

    def is_empty(self):
        """判断是否为空"""
        return self.items == []

    def push(self, item):
        """加入元素"""
        self.items.append(item)

    def pop(self):
        """弹出元素"""
        return self.items.pop()

    def peek(self):
        """返回栈顶元素"""
        return self.items[len(self.items)-1]

    def size(self):
        """返回栈的大小"""
        return len(self.items)


def dice(max, min=0):
    """
    骰子工具
    :param max: 最大数+1
    :param min: 最小数，默认为0
    :return: 随机采样数
    """
    return random.randrange(min, max, 1)

#############################################################
# 结构转换
#############################################################

def structure_dotB2Edge(dotB):
    """
    将点括号结构转化为边集
    :param dotB: 点括号结构
    :return: 边集，有向图
    """
    l = len(dotB)
    # 初始化
    u = []
    v = []
    for i in range(l - 1):
        u += [i, i + 1]
        v += [i + 1, i]
    str_list = list(dotB)
    stack = Stack()
    for i in range(l):
        if (str_list[i] == '('):
            stack.push(i)
        elif (str_list[i] == ')'):
            last_place = stack.pop()
            u += [i, last_place]
            v += [last_place, i]
    edge_index = torch.tensor(np.array([u, v]))
    return edge_index


def structure_edge2DotB(edge_index):
    """
    将边集转换为点括号结构
    :param edge_index: 边集，tensor
    :return:
    """
    u = edge_index[0, :]
    v = edge_index[1, :]
    l = torch.max(u) + 1
    dotB = ['.'] * l
    edges = edge_index.t()
    for edge in edges:
        if edge[0] < edge[1] - 1:
            dotB[edge[0]] = '('
            dotB[edge[1]] = ')'
    dotB = ''.join(dotB)
    return dotB


#############################################################
# 序列转换
#############################################################

def base2Onehot(base):
    """
    碱基字符转onehot编码
    :param base:
    :return:
    """
    onehot = torch.tensor(np.zeros((4,)), dtype=torch.long)
    i = 0
    if (base == "A"):
        i = 0
    elif (base == "U"):
        i = 1
    elif (base == "C"):
        i = 2
    else:
        i = 3
    onehot[i] = 1

    return onehot


def seq_base2Onehot(seq_base, max_size=None):
    """
    将碱基对序列编码为onehot形式
    :param: seq_base: 碱基序列
    :param: max_size: 序列最大长度
    :return: 碱基序列的onehot编码，tensor形式
    """
    # pool = mp.Pool()
    l = len(seq_base)
    # seq_onehot = pool.map(base2Onehot, list(seq_base))
    # pool.close()
    # pool.join()
    seq_onehot = map(base2Onehot, list(seq_base))
    seq_onehot = list(seq_onehot)
    if max_size is not None:
        seq_onehot += [torch.tensor([0, 0, 0, 0])] * (max_size - l)
    seq_onehot = torch.stack(seq_onehot, dim=0)
    return seq_onehot


def onehot2Base(onehot):
    """
    onehot编码转碱基字符
    :param onehot:  onehot编码，tensor
    :return: 碱基字符
    """
    base = ['A', 'U', 'C', 'G']
    onehot = onehot.numpy()[0]
    if np.all(onehot == 0):
        return ''
    i = np.where(onehot == 1)
    i = i[0].item()
    return base[i]


def seq_onehot2Base(seq_onehot):
    """
    onehot编码序列转为碱基字符串
    :param seq_onehot: onehot编码序列，tensor
    :return: 碱基字符串
    """
    # seq_onehot = delete_zero_row(seq_onehot)
    # pool = mp.Pool()
    seq = list(torch.split(seq_onehot, 1, dim=0))
    # seq_base = pool.map(onehot2Base, seq)
    # pool.close()
    # pool.join()
    seq_base = map(onehot2Base, seq)
    seq_base = ''.join(list(seq_base))
    return seq_base


#############################################################
# 能量计算
#############################################################

def get_energy_onehot(seq_onehot, dotB):
    """
    由onehot序列计算能量
    :param seq_onehot: onehot序列
    :return: 能量
    """
    seq_base = seq_onehot2Base(seq_onehot)
    energy = RNA.energy_of_struct(seq_base, dotB)
    return energy


def get_energy_base(seq_base, dotB):
    """
    由碱基序列序列计算能量
    :param seq_base: 碱基序列
    :return: 能量
    """
    energy = RNA.energy_of_struct(seq_base, dotB)
    return energy


def get_energy_graph(graph):
    """
    由graph计算能量
    :param graph: 图，pyg.Data
    :return: 能量
    """
    # energy = get_energy_onehot(graph.x, graph.y['dotB'])
    energy = get_energy_base(graph.y['seq_base'], graph.y['dotB'])
    return energy


#############################################################
# 计算距离
#############################################################

def get_distance_from_onehot(seq_onehot ,dotB_Aim):
    """
    由onehot编码序列计算与目标结构的距离
    :param seq_onehot: onehot序列
    :param dotB_Aim: 目标结构的dotB结构
    :return: 距离
    """
    seq_base = seq_onehot2Base(seq_onehot)
    dotB_Real = RNA.fold(seq_base)[0]
    # distance = Levenshtein.distance(dotB_Real, dotB_Aim)
    distance = RNA.hamming_distance(dotB_Real, dotB_Aim)
    return distance


def get_distance_from_base(seq_base ,dotB_Aim):
    """
    由onehot编码序列计算与目标结构的距离
    :param seq_onehot: onehot序列
    :param dotB_Aim: 目标结构的dotB结构
    :return: 距离
    """
    dotB_Real = RNA.fold(seq_base)[0]
    # distance = Levenshtein.distance(dotB_Real, dotB_Aim)
    distance = RNA.hamming_distance(dotB_Real, dotB_Aim)
    return distance


def get_distance_from_base_norm(seq_base ,dotB_Aim):
    """
    由onehot编码序列计算与目标结构的距离
    :param seq_onehot: onehot序列
    :param dotB_Aim: 目标结构的dotB结构
    :return: 距离
    """
    dotB_Real = RNA.fold(seq_base)[0]
    # distance = Levenshtein.distance(dotB_Real, dotB_Aim) / len(dotB_Aim)
    distance = RNA.hamming_distance(dotB_Real, dotB_Aim) / len(dotB_Aim)
    return distance


def get_distance_from_graph(graph):
    """
    由graph计算与目标结构的距离
    :param graph: 图
    :return: 距离
    """
    # distance = get_distance_Levenshtein(graph.x, graph.y['dotB'])
    seq_base = graph.y['seq_base']
    dotB_aim = graph.y['dotB']
    dotB_real = RNA.fold(seq_base)[0]
    # distance = Levenshtein.distance(dotB_real, dotB_aim)
    distance = RNA.hamming_distance(dotB_real, dotB_aim)
    return distance


def get_distance_from_graph_norm(graph):
    """
    由graph计算与目标结构的距离
    :param graph: 图
    :return: 距离
    """
    # distance = get_distance_Levenshtein(graph.x, graph.y['dotB'])
    seq_base = graph.y['seq_base']
    dotB_aim = graph.y['dotB']
    dotB_real = RNA.fold(seq_base)[0]
    # distance = Levenshtein.distance(dotB_real, dotB_aim) / len(dotB_aim)
    distance = RNA.hamming_distance(dotB_real, dotB_aim) / len(dotB_aim)
    return distance


#############################################################
# 建图
#############################################################

def get_graph(dotB, h_weight=2):
    l = len(dotB)
    seq_onehot = torch.zeros((l, 4))
    edge_index = structure_dotB2Edge(dotB)

    edge_attr = []

    for i in range(edge_index.shape[1]):
        place = edge_index[0][i]
        pair_place = edge_index[1][i]
        if place + 1 < pair_place or place > pair_place + 1:
            edge_attr.append(h_weight)
        else:
            edge_attr.append(1)

    edge_attr = torch.tensor(edge_attr).view(-1, 1)

    y = {'dotB': dotB, 'seq_base': ''}

    graph = torch_geometric.data.Data(x=seq_onehot, y=y, edge_index=edge_index, edge_attr=edge_attr)

    return graph


#############################################################
# 图初始化
#############################################################

def init_graph(graph, init_len, action_space):
    """
    图初始化
    :param graph: 原空白图
    :param init_len: 初始化核苷酸位数
    :param action_space: 动作空间数
    :return:
    """
    if action_space == 4:
        base_pair_dict = base_pair_dict_4
        base_pair_list = base_pair_list_4
    elif action_space == 6:
        base_pair_dict = base_pair_dict_6
        base_pair_list = base_pair_list_6
    edges = graph.edge_index.t()
    u = graph.edge_index[0]
    v = graph.edge_index[1]
    seq_base_init = []
    for i in range(init_len):
        pair = base_pair_list[dice(action_space)]
        base = pair[0]
        onehot = base2Onehot(base)
        graph.x[i] = onehot
        index = (graph.edge_index[0,:] == i).nonzero()
        next_places = graph.edge_index[1, index]
        for next_place in next_places:
            if next_place > i + 1:
                next_onehot = base2Onehot(pair[1])
                graph.x[next_place.item()] = next_onehot
    return graph


#############################################################
# 动作
#############################################################

def act(graph, action, place, action_space, do_skip):
    """
    图动作
    :param graph:
    :param place:
    :param action:
    :return:
    """
    if action_space == 4:
        base_pair_dict = base_pair_dict_4
        base_pair_list = base_pair_list_4
    elif action_space == 6:
        base_pair_dict = base_pair_dict_6
        base_pair_list = base_pair_list_6

    skip = 0
    if torch.any(graph.x[place] == 1.) and do_skip:
        skip = 1
    else:
        pair = base_pair_list[action]
        base = pair[0]
        onehot = base2Onehot(base)
        graph.x[place] = onehot
        index = (graph.edge_index[0, :] == place).nonzero()
        next_places = graph.edge_index[1, index]
        for next_place in next_places:
            if (next_place > place + 1) or (next_place < place -1):
                next_base = pair[1]
                next_onehot = base2Onehot(next_base)
                graph.x[next_place.item()] = next_onehot
    return graph, skip


