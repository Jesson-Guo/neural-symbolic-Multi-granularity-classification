import numpy as np
import torch
import copy

from gym import Env, spaces
from src.tree import sub_node_classifier, node_hash
from utils.globals import *


class TreeClassifyEnv(Env):
    def __init__(self, obs_len, decay) -> None:
        super().__init__()
        # probability
        self.observation_space = spaces.Box(low=np.zeros(obs_len), high=np.ones(obs_len), dtype=np.float32)
        # 0: stop
        # 1: select children
        # 2: select parent
        self.action_space = spaces.Discrete(3)
        self.obs_len = obs_len
        self.decay = decay

    def reset(self, batch_size, labels):
        self.labels = labels.data.numpy()
        self.layer = [1 for _ in range(batch_size)]
        state = sub_node_classifier[1][0]
        self.state = [state for _ in range(batch_size)]
        self.done = [0 for _ in range(batch_size)]
        return self.state, {}

    def step(self, batch_size, action):
        next_state = self.state
        reward = [0 for _ in range(batch_size)]
        label2id = get_value('label2id')
        id2label = get_value('id2label')
        node_dict = get_value('node_dict')
        lpaths = get_value('lpaths')

        for i in range(batch_size):
            if self.done[i]:
                continue

            lp = lpaths[self.labels[i]]
            lp_len = len(lp)
            target_node = node_dict[id2label[self.labels[i]]]

            if action[i] == 1:
                self.done[i] = 1
                next_state[i] = self.state[i]
                dist = target_node.node_distance(next_state[i], label2id, lpaths)
                reward[i] = self.layer[i] if dist == 0 else -dist

            elif action[i] == 0:
                if self.state[i].is_leaf():
                    next_state[i] = self.state[i]
                    reward[i] = -self.layer[i]
                else:
                    pred = self.state[i].sub_out[i].reshape(1, -1)
                    pred = torch.softmax(pred, dim=1).data.max(1)[1]
                    next_state[i] = self.state[i].children[pred.item()]
                    self.layer[i] += 1

                    if self.layer[i] >= lp_len:
                        reward[i] = -self.layer[i]
                    else:
                        corr_node = node_dict[id2label[lp[self.layer[i]]]]
                        dist = corr_node.node_distance(next_state[i], label2id, lpaths)
                        reward[i] = self.layer[i] if dist == 0 else -dist

            elif action[i] == 2:
                # root has no parent
                if self.state[i].parent == None:
                    next_state[i] = self.state[i]
                    reward[i] = -1
                else:
                    next_state[i] = self.state[i].parent
                    self.layer[i] -= 1
                    
                    if self.layer[i] >= lp_len:
                        reward[i] = -self.layer[i]
                    else:
                        corr_node = node_dict[id2label[lp[self.layer[i]]]]
                        dist_curr = corr_node.node_distance(self.state[i], label2id, lpaths)
                        dist_next = corr_node.node_distance(next_state[i], label2id, lpaths)
                        reward[i] = dist_curr - dist_next

        self.state = next_state
        return next_state, reward, self.done, {}


def rl_process(env, agent, i, labels):
    episodes = 100
    max_reward = 0
    min_reward = 0
    outputs = (['fall11'], 0)
    for _ in range(episodes):
        total_step = 0
        total_reward = 0
        step = 0
        # TODO 固定初状态返回root节点，是否可以随机返回一个node？
        state, _ = env.reset(i, labels)
        done = False
        out = [state.wnid]
        while not done:
            action = agent.select_action(state.sub_out[i])
            next_state, reward, done, _ = env.step(i, action)
            agent.store_transition(state.sub_out[i].cpu().detach().numpy(), action, reward, next_state.sub_out[i].cpu().detach().numpy())
            if agent.memory_counter >= agent.memory_capacity:
                agent.learn()
                # if done:
                #     print("episode: {} , the episode reward is {}".format(i, round(total_reward, 3)))
            state = next_state
            step += 1
            total_reward += reward
            out.append(state.wnid)
        total_step += step + 1
        max_reward = max(max_reward, total_reward)
        min_reward = min(min_reward, total_reward)
        outputs = (out, total_reward) if total_reward > outputs[1] else outputs
        # print("Total Step:{}\t Episode: {}\t Total Reward: {:0.2f}".format(total_step, epoch, total_reward))
    return outputs, max_reward, min_reward
