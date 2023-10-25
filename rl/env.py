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
        # 0: select children
        # 1: stop
        # 2: select sibling
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
        for i in range(batch_size):
            if self.done[i]:
                continue
            lp = get_value('lpaths')[self.labels[i]]
            lp_len = len(lp)

            reward[i] = -self.layer[i]
            if action[i] == 0:
                if self.state[i].is_leaf():
                    self.done[i] = 1
                    next_state[i] = self.state[i]
                    # for test
                    if self.layer[i]-1 < len(lp) and get_value('label2id')[next_state[i].wnid] == lp[self.layer[i]-1]:
                        reward[i] = self.layer[i]
                else:
                    pred = self.state[i].sub_out[i].reshape(1, -1)
                    pred = torch.softmax(pred, dim=1).data.max(1)[1]
                    next_state[i] = self.state[i].children[pred.item()]
                    self.layer[i] += 1
                    if self.layer[i]-1 < len(lp) and get_value('label2id')[next_state[i].wnid] == lp[self.layer[i]-1]:
                        reward[i] = self.layer[i]
            elif action[i] == 1:
                self.done[i] = 1
                next_state[i] = self.state[i]
                if self.layer[i] != 1 and self.layer[i]-1 < len(lp) and get_value('label2id')[next_state[i].wnid] == lp[self.layer[i]-1]:
                    reward[i] = self.layer[i]
            elif action[i] == 2:
                # root
                if self.state[i].parent == None:
                    self.done[i] = 1
                    next_state[i] = self.state[i]
                    # for test
                    reward[i] = -10
                else:
                    parent = self.state[i].parent
                    for k, v in parent.children.items():
                        if v == self.state[i]:
                            break
                    sub_out = copy.copy(parent.sub_out)
                    sub_out[i, k] = 0

                    pred = sub_out[i].reshape(1, -1)
                    pred = torch.softmax(pred, dim=1).data.max(1)[1]
                    next_state[i] = parent.children[pred.item()]

                    if self.layer[i]-1 < len(lp) and get_value('label2id')[next_state[i].wnid] == lp[self.layer[i]-1]:
                        reward[i] = self.layer[i]

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
