import os.path
from collections import namedtuple

import torch_geometric.data
from torch import no_grad, clamp
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
from network.rd_net import Backbone, Actor, Critic
import torch
import torch.nn as nn
import torch.nn.functional as F
import pathos.multiprocessing as pathos_mp
from functools import partial
from rl_lib.environment import fetch_items


Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'step', 'reward', 'next_state', 'done'])


def get_action_sample(probs):
    return torch.multinomial(probs, 1).item()


def get_action_max(probs):
    return torch.argmax(probs).item()


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

        self.backbone = Backbone(param_b.in_size, param_b.out_size, param_b.hide_size_list, param_b.normalize, param_b.bias)
        self.actor = Actor(param_a.in_size, param_a.out_size, param_a.hide_size_list, param_a.n_gcn, param_a.normalize, param_a.bias)
        self.critic = Critic(param_c.in_size, param_c.out_size, param_c.hide_size_list, param_c.n_gcn, param_c.normalize, param_c.bias)

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

    def forward(self, data_batch_, actions, index):
        data_batch = data_batch_.clone().to(self.device)
        x = Variable(data_batch.x.float().to(self.device))
        edge_index = Variable(data_batch.edge_index.to(self.device))
        edge_attr = Variable(data_batch.edge_attr.to(self.device))
        edge_weight = edge_attr.view(-1, ).float()
        batch = data_batch.batch

        feature = self.backbone(x, edge_index, edge_weight)

        values = self.critic(feature, edge_index, batch, edge_weight)

        graphs_probs = self.actor(feature, edge_index, edge_weight)

        action_probs = graphs_probs[index]

        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return action_log_probs, values, dist_entropy

    def work(self, data_batch_, index, type='sample'):
        data_batch = data_batch_.clone().to(self.device)
        x = Variable(data_batch.x.float().to(self.device))
        edge_index = Variable(data_batch.edge_index.to(self.device))
        edge_attr = Variable(data_batch.edge_attr.to(self.device))
        edge_weight = edge_attr.view(-1, ).float()

        with no_grad():
            feature = self.backbone(x, edge_index, edge_weight)
            graph_probs = self.actor(feature, edge_index, edge_weight).cpu()
            action_probs = graph_probs[index]
            action_probs_list = torch.split(action_probs, 1, dim=0)

        if type == 'sample':
            action_list = self.pool.map(get_action_sample, action_probs_list)
        else:
            action_list = self.pool.map(get_action_max, action_probs_list)

        actions = torch.tensor(list(action_list), dtype=torch.long).view(-1, )
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)

        return actions.detach(), action_log_probs.detach()

    def storeTransition(self, transition, id_chain):
        buffer_tmp = self.buffer_list[id_chain] + [transition]
        self.buffer_list[id_chain] = buffer_tmp

    def trainStep(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        graph_list = []
        action_list = []
        reward_list = []
        old_a_log_probs_list = []
        Gt_list = []
        step_list = []

        for id_chain in range(self.n_chain):
            graph_list_tmp = [t.state for t in self.buffer_list[id_chain]]
            action_list_tmp = [t.action for t in self.buffer_list[id_chain]]
            reward_list_tmp = [t.reward for t in self.buffer_list[id_chain]]
            old_a_log_probs_list_tmp = [t.a_log_prob for t in self.buffer_list[id_chain]]
            step_list_tmp = [t.step for t in self.buffer_list[id_chain]]
            done_list_tmp = [t.done for t in self.buffer_list[id_chain]]

            R = 0
            Gt_list_tmp = []
            for r, done in zip(reward_list_tmp[::-1], done_list_tmp[::-1]):
                if done:
                    R = 0
                R = r + self.gamma * R
                Gt_list_tmp.insert(0, R)

            graph_list += graph_list_tmp
            action_list += action_list_tmp
            Gt_list += Gt_list_tmp
            step_list += step_list_tmp
            old_a_log_probs_list += old_a_log_probs_list_tmp\

        Gt = torch.tensor(Gt_list, dtype=torch.float).to(self.device)
        actions = torch.tensor(action_list).view(-1,).to(self.device)
        steps = torch.tensor(step_list).view(-1,).to(self.device)
        old_a_log_probs = torch.tensor(old_a_log_probs_list, dtype=torch.float).view(-1,).to(self.device)

        loss_a_all = 0
        loss_c_all = 0

        for i in range(1, self.k_epoch+1):
            loss_a = 0
            loss_c = 0
            loss_a_log = 0
            loss_c_log = 0
            n_log = 0

            for index in BatchSampler(SubsetRandomSampler(range(len(graph_list))), batch_size, False):
                Gt_index = Gt[index]
                actions_index = actions[index]
                step_index = steps[index]
                graphs_index = fetch_items(graph_list, index)
                len_index = [len(graph.y['dotB']) for graph in graphs_index]
                len_index = [1] + len_index[:-1]
                node_index = [len_index[i] + step_index[i].item() - 1 for i in range(len(len_index))]
                graphs_index = torch_geometric.data.Batch.from_data_list(graphs_index).to(self.device)

                a_log_probs, values, dist_entropy = self.forward(graphs_index, actions_index, node_index)
                delta = Gt_index - values.view(-1,)
                advantage = delta.view(-1,)

                ratio = torch.exp(a_log_probs - old_a_log_probs[index].detach())
                surr1 = ratio * advantage
                surr2 = clamp(ratio, 1 - self.eps_clips, 1 + self.eps_clips) * advantage

                loss_a = -torch.min(surr1, surr2).mean()
                loss_c = F.mse_loss(Gt_index.view(-1,), values.view(-1,))
                loss_all = loss_a + 0.5 * loss_c - 0.01 * dist_entropy.mean()

                l = len(index)
                n_log += l
                loss_a_log += loss_a.item() * l
                loss_c_log += loss_c.item() * l

                self.optimizer_b.zero_grad()
                self.optimizer_a.zero_grad()
                self.optimizer_c.zero_grad()

                loss_all.backward()
                # nn.utils.clip_grad_norm_(self.backbone.parameters(), self.max_grad_norm)
                # nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                # nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.optimizer_b.step()
                self.optimizer_a.step()
                self.optimizer_c.step()

            loss_a_log = loss_a_log / n_log
            loss_c_log = loss_c_log / n_log
            print("Loss_A: {}, Loss_c: {}".format(loss_a_log, loss_c_log))

        return loss_a_log, loss_c_log

    def clean_buffer(self):
        self.buffer_cnt += 1
        if self.buffer_cnt % self.buffer_loop == 0:
            self.buffer_cnt = 0
            for i in range(len(self.buffer_list)):
                del self.buffer_list[i][:]

    def save(self, model_dir, episode):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.backbone.state_dict(), model_dir + 'backbone_' + str(episode) + '.pth')
        torch.save(self.actor.state_dict(), model_dir + 'actor_' + str(episode) + '.pth')
        torch.save(self.critic.state_dict(), model_dir + 'critic_' + str(episode) + '.pth')

    def load(self, model_dir, episode):
        if not os.path.exists(model_dir):
            raise ValueError('Files not exist!')

        self.backbone.load_state_dict(
            torch.load(model_dir + 'backbone_' + str(episode) + '.pth', map_location=lambda storage, loc: storage))
        self.actor.load_state_dict(
            torch.load(model_dir + 'actor_' + str(episode) + '.pth', map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(
            torch.load(model_dir + 'critic_' + str(episode) + '.pth', map_location=lambda storage, loc: storage))
        print('Load weights from {} with episode {}'.format(model_dir, episode))





