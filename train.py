import cProfile
import math
from rl_lib.environment import Env_RNA
from rl_lib.ppo import PPO, Transition
from network.rd_net import Backbone, Critic, Actor
from torch.autograd import Variable
from utils.config import action_space, device, backboneParam, actorParam, criticParam
import os
import torch
import torch_geometric
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils.rna_lib import get_distance_from_graph_norm, seq_onehot2Base, get_distance_from_base
import pathos.multiprocessing as pathos_mp

def main():
    ################## training ##################

    print("========= train =========")

    ################## global configration ##################

    # root path
    root = os.path.dirname(os.path.realpath(__file__))
    # local time
    local_time = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

    # multi-processing pool
    pool_main = pathos_mp.ProcessPool()
    pool_env = pathos_mp.ProcessPool()
    pool_agent = pathos_mp.ProcessPool()

    ################## environment ##################

    # load data
    data_dir = root + '/data/raw/rfam_learn/train/1.rna'
    dotB_list = []
    f = open(data_dir)
    iter_f = iter(f)
    for line in iter_f:
        dotB_list.append(line.replace('/n', ''))

    # create env
    init_len = 1

    env = Env_RNA(dotB_list=dotB_list, action_space=action_space, h_weight=2, pool=pool_env)

    env.reset(init_len=init_len)

    max_len = max(env.len_list)

    ################## agent ##################

    # ppo parameters
    # training epochs
    k_epochs = 6
    # batch size
    batch_size = 64
    # clip for ppo surr2
    eps_clip = 0.1
    # reward decay
    gamma = 0.9
    # frequency for buffer clean
    buffer_clean_freq = 3

    # learning rate decay
    lr_decay = 0.095

    agent = PPO(
        backboneParam, actorParam, criticParam, k_epochs, batch_size, eps_clip, gamma,
        n_graph=len(env.graphs), pool=pool_agent, action_space=action_space, max_loop=buffer_clean_freq, device=device
    )

    # learning rate schedule
    scheduler_b = torch.optim.lr_scheduler.ExponentialLR(agent.optimizer_b, lr_decay)
    scheduler_c = torch.optim.lr_scheduler.ExponentialLR(agent.optimizer_c, lr_decay)
    scheduler_a = torch.optim.lr_scheduler.ExponentialLR(agent.optimizer_a, lr_decay)

    ################## processing details ##################

    # round times
    round_time = 300

    # step in a round
    max_ep_len = max_len

    # max steps for training
    max_train_timestep = round_time * max_ep_len

    # frequency for training
    update_timestep = max_ep_len

    # frequency for printing state
    print_freq = 4 * max_ep_len

    # frequency for saving the model
    save_model_freq = 20 * max_ep_len

    # frequency for log
    log_freq = 1 * max_ep_len

    action_type = 'sample'

    ################## logging ##################

    # main log folder
    log_dir_root = root + '/logs/' + local_time
    if not os.path.exists(log_dir_root):
        os.makedirs(log_dir_root)

    # log folder
    log_dir = log_dir_root + '/Logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # model folder
    model_dir = log_dir_root + '/Model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # tensorboard folder
    tensor_dir = log_dir_root + '/Tensor/'
    if not os.path.exists(tensor_dir):
        os.makedirs(tensor_dir)

    # create tensorboard
    writer = SummaryWriter(tensor_dir, comment="Train_Log_{}.log".format(local_time))
    writer.add_text('created time', str(local_time))
    print('tensorboard at: ' + tensor_dir)

    # running time files
    time_dir = log_dir_root + '/program.prof'
    cProfile.run('re.compile("ccc")', filename=time_dir)

    # log files
    log_f_name = log_dir + '/log'
    done_f_name = log_dir + '/done.csv'
    print("logging at: " + log_f_name)

    # logging file
    log_f_list = []
    act_log_f_list = []
    for i in range(len(env.dotB_list)):
        log_f_list.append(open(log_f_name + '_' + str(i) + '.csv', "w+"))
        act_log_f_list.append(open(log_f_name + '_act_' + str(i) + '.csv', "w+"))
    done_log_f = open(done_f_name, "w+")
    for log_f in log_f_list:
        log_f.write('episode, timestep, reward, distance, dotB, sequence' + '\n')

    ################## training procedure ##################

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT): ", start_time)
    print("=================================================================================")

    time_step = 0
    i_episode = 0

    # game round
    while time_step < max_train_timestep:
        i_episode += 1
        print("=========================== round " + str(i_episode) + " ===========================")

        # reset the environment for a new round
        state = torch_geometric.data.Batch.from_data_list(env.reset(init_len))
        state.x, state.edge_index = Variable(state.x.float().to(device)), Variable(state.edge_index.to(device))
        state.edge_attr = Variable(state.edge_attr.to(device))

        for t in range(1, max_ep_len+1):
            # agent work
            step_index = env.get_index_ep(place=t-1)
            actions, a_log_probs = agent.work(state, step_index, action_type)

            # environment react
            next_state, reward_list, finished_list = env.step(actions, t-1)

            # put in buffer
            action_list = actions.split(1, dim=0)
            a_log_prob_list = a_log_probs.split(1, dim=0)

            for graph, action, a_log_prob, reward, next_graph, id, done in zip(
                state.clone().to_data_list(), action_list, a_log_prob_list, reward_list, next_state, env.id_list, finished_list
            ):
                trans = Transition(graph, action, a_log_prob, t-1, reward, next_graph, done)
                agent.storeTransition(trans, id)
                if reward == 1:
                    done_log_f.write(
                        'Episode_{} Graph_{} is done! Struct: {} | Sequence: {}'.format(
                            i_episode, id,
                            next_graph.y['dotB'],
                            seq_onehot2Base(next_graph.x)
                        ))
                    done_log_f.write('\n')
                    done_log_f.flush()

            # remove the finished graphs
            finish_index = list(np.nonzero(finished_list == 1)[0])

            if len(finish_index) > 0:
                env.remove_graph(finish_index)

            # if all graphs are done
            if len(env.graphs) == 0:
                break

            time_step += 1

            # renew the state
            state = torch_geometric.data.Batch.from_data_list(env.graphs).clone()
            state.x, state.edge_index = Variable(state.x.float().to(device)), Variable(state.edge_index.to(device))
            state.edge_attr = Variable(state.edge_attr.to(device))

        # log data for a round
        final_graphs = [chain[-1].next_state for chain in agent.buffer]
        final_seqs_onehot = [graph.x[:, :4] for graph in final_graphs]
        final_seqs_base = list(map(seq_onehot2Base, final_seqs_onehot))
        # final_seqs_base = [graph.y['seq_base'] for graph in final_graphs]
        final_dotBs = [graph.y['dotB'] for graph in final_graphs]
        final_distance = pool_main.map(get_distance_from_base, final_seqs_base, final_dotBs)
        final_distance = list(final_distance)
        final_rewards = []
        for i in range(len(agent.buffer)):
            rewards = [g.reward for g in agent.buffer[i]]
            final_rewards.append(np.array(rewards).sum())

        # log for a round
        if time_step % log_freq == 0:
            for i in range(len(log_f_list)):
                log_f_list[i].write(
                    '{},{},{},{},{},{}\n'.format(i_episode, time_step, final_rewards[i],
                                                    final_distance[i], final_dotBs[i],
                                                    final_seqs_base[i]))
                log_f_list[i].flush()

            for i in range(len(act_log_f_list)):
                action_list = [str(t.action) for t in agent.buffer[i]]
                act_str = ','.join(action_list)
                act_log_f_list[i].write(
                    act_str + '\n'
                )
                act_log_f_list[i].flush()

        # train the network
        if time_step % update_timestep == 0:
            loss_a, loss_c = agent.trainStep()
            loss_a = abs(loss_a)
            scheduler_b.step()
            scheduler_c.step()
            scheduler_a.step()

            # 记录到tensorboard
            # loss
            writer.add_scalar('loss_a', loss_a, i_episode)
            writer.add_scalar('loss_c', loss_c, i_episode)

            # 网络参数
            for tag_, value in agent.backbone.named_parameters():
                tag_ = "b." + tag_.replace('.', '/')
                writer.add_histogram(tag_, value, i_episode)
                writer.add_histogram(tag_ + '/grad', value.grad.data.cpu().numpy(), i_episode)
            for tag_, value in agent.actor.named_parameters():
                tag_ = "a." + tag_.replace('.', '/')
                writer.add_histogram(tag_, value, i_episode)
                writer.add_histogram(tag_ + '/grad', value.grad.data.cpu().numpy(), i_episode)
            for tag_, value in agent.critic.named_parameters():
                tag_ = "c." + tag_.replace('.', '/')
                writer.add_histogram(tag_, value, i_episode)
                writer.add_histogram(tag_ + '/grad', value.grad.data.cpu().numpy(), i_episode)

            for i in range(len(env.graphs)):
                writer.add_scalar('reward_' + str(i), final_rewards[i], i_episode)
                writer.add_scalar('distance_' + str(i), final_distance[i], i_episode)
                writer.add_text('sequence_' + str(i), final_seqs_base[i], i_episode)

            agent.clean_buffer()

            # 学习率
            writer.add_histogram('lr_b', agent.optimizer_b.state_dict()['param_groups'][0]['lr'],
                                 i_episode)
            writer.add_histogram('lr_a', agent.optimizer_a.state_dict()['param_groups'][0]['lr'],
                                 i_episode)
            writer.add_histogram('lr_c', agent.optimizer_c.state_dict()['param_groups'][0]['lr'],
                                 i_episode)

        # save model
        if time_step % save_model_freq == 0:
            print(
                "--------------------------------------------------------------------------------------------")
            print("saving model at : " + model_dir)
            agent.save(model_dir, i_episode)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print(
                "--------------------------------------------------------------------------------------------")

    for log_f in log_f_list:
        log_f.close()
    for log_f in act_log_f_list:
        log_f.close()
    # for log_f in place_log_f_list:
    #     log_f.close()
    done_log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    main()