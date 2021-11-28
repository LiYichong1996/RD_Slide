from torch.autograd import Variable
from torch.distributions import Categorical

from network.rd_net import Backbone, Actor, Critic
import torch
import torch.nn as nn
import torch.nn.functional as F
import pathos.multiprocessing as pathos_mp
from functools import partial


class PPO(nn.Module):
    def __init__(
            self, param_b, param_a, param_c,
            k_epoch=5, batch_size=100, eps_clips=0.2, gamma=0.9, max_grad_norm=1, n_graph=100,
            pool=None, action_space=4, max_loop=1, device=torch.device('cpu')
    ):
        super(PPO, self).__init__()
        if pool is None:
            self.pool = pathos_mp.ProcessPool()
        else:
            self.pool = pool

        self.eps_clips = eps_clips
        self.use_cuda = (True if torch.cuda.is_available() else False)
        self.k_epoch = k_epoch
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.n_chain = n_graph
        self.action_space = action_space
        self.device = device

        self.backbone = Backbone(param_b.in_size, param_b.out_size, param_b.hide_size_list, param_b.normalize)
        self.actor = Actor(param_a.in_size, param_a.out_size, param_a.hide_size_list, param_a.n_gcn, param_a.normalize)
        self.critic = Critic(param_c.in_size, param_c.out_size, param_c.hide_size_list, param_c.n_gcn, param_c.normalize)

        self.batch_size = batch_size

        self.lr_b = param_b.lr
        self.lr_a = param_a.lr
        self.lr_c = param_c.lr

        self.optimizer_b = torch.optim.Adam(self.backbone.parameters(), lr=self.lr_b)
        self.optimizer_a = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.optimizer_c = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.buffer_list = [[]] * self.n_chain
        self.buffer_cnt = 0
        self.buffer_loop = max_loop

    def forward(self, data_batch_, actions, ep, len_list_):
        data_batch = data_batch_.clone.to(self.device)
        x = Variable(data_batch.x.float().to(self.device))
        edge_index = Variable(data_batch.edge_index.to(self.device))
        edge_attr = Variable(data_batch.edge_attr.to(self.device))
        edge_weight = edge_attr.view(-1, ).float()

        feature = self.backbone(x, edge_index, edge_weight)

        values = self.critic(feature, edge_index, edge_weight)

        graphs_probs = self.actor(feature, edge_index, edge_weight)

        len_list = [0] + len_list_[:-1]

        probs_index = torch.tensor([l+ep for l in len_list])

        action_probs = graphs_probs[probs_index]

        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return action_log_probs, values, dist_entropy


