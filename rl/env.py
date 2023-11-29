import numpy as np
import torch
import copy
import random

from gym import Env, spaces
from utils.globals import *


label2id = {}
id2label = {}
node_dict = {}
lpaths = {}
tree = None


class TreeClassifyEnv(Env):
    def __init__(self, obs_len, decay) -> None:
        super().__init__()
        # probability
        self.observation_space = spaces.Box(low=np.zeros(obs_len), high=np.ones(obs_len), dtype=np.float32)
        # 0: 当前路径错误
        # 1: 当前路径正确
        # 2: stop
        self.action_space = spaces.Discrete(3)
        self.obs_len = obs_len
        self.decay = decay

        global label2id, id2label, node_dict, lpaths, tree
        label2id = get_value('label2id')
        id2label = get_value('id2label')
        node_dict = get_value('node_dict')
        lpaths = get_value('lpaths')
        tree = get_value('tree')

    def reset(self, batch_size, x, labels, train=True):
        self.labels = labels.cpu().numpy()
        self.done = [0 for _ in range(batch_size)]

        self.state = []
        self.layer = []
        for i in range(batch_size):
            if train:
                rand_node = node_dict[id2label[random.sample(id2label.keys(), 1)[0]]]
                self.state.append(rand_node)
                self.layer.append(rand_node.layer)
            else:
                self.state.append(tree)
                self.layer.append(1)

        init_buffer = []

        for i in range(batch_size):
            lp = lpaths[self.labels[i]]
            for j in range(len(lp)-1):
                node = node_dict[id2label[lp[j]]]
                fc = node.classifier.state_dict()['1.weight'].cpu().numpy()
                fc = fc.mean(0)
                temp = np.concatenate((x[i], fc), axis=None)

                next_node = node_dict[id2label[lp[j+1]]]
                if next_node.is_leaf():
                    fc_next = np.zeros(x[i].shape)
                else:
                    fc_next = next_node.classifier.state_dict()['1.weight'].cpu().numpy()
                    fc_next = fc_next.mean(0)
                next_temp = np.concatenate((x[i], fc_next), axis=None)

                init_buffer.append((temp, 1, 1, next_temp))
            init_buffer.append((next_temp, 1, 1, next_temp))
        return self.state, {'init_buffer': init_buffer}

    def step(self, batch_size, action):
        next_state = copy.copy(self.state)
        reward = [0 for _ in range(batch_size)]

        for i in range(batch_size):
            if self.done[i]:
                continue

            lp = lpaths[self.labels[i]]
            lp_len = len(lp)
            target_node = node_dict[id2label[self.labels[i]]]

            if action[i] == 0:
                # if self.state[i].is_leaf():
                #     next_state[i] = self.state[i]
                #     reward[i] = -1
                #     # reward[i] = -self.layer[i]
                # else:
                #     pred = self.state[i].sub_out[i].reshape(1, -1)
                #     pred = torch.softmax(pred, dim=1).data.max(1)[1]
                #     next_state[i] = self.state[i].children[pred.item()]
                #     self.layer[i] += 1

                #     dist_curr = target_node.node_distance(self.state[i], label2id, lpaths)
                #     dist_next = target_node.node_distance(next_state[i], label2id, lpaths)
                #     # reward[i] = dist_curr - dist_next
                #     reward[i] = self.layer[i] if dist_curr > dist_next else -dist_next
                if self.state[i].parent == None:
                    self.done[i] = 1
                    next_state[i] = self.state[i]
                    reward[i] = -1
                else:
                    for k, v in self.state[i].parent.children.items():
                        if v.wnid == self.state[i].wnid:
                            break
                    self.state[i].parent.sub_out[i, k] = 0
                    next_state[i] = self.state[i].parent

                    if label2id[self.state[i].wnid] in lp:
                        reward[i] = -self.layer[i]
                    else:
                        reward[i] = self.layer[i]
                    # dist_curr = target_node.node_distance(self.state[i], label2id, lpaths)
                    # dist_next = target_node.node_distance(next_state[i], label2id, lpaths)
                    # reward[i] = 1 if dist_curr > dist_next else -1
                    self.layer[i] -= 1

            elif action[i] == 1:
                if self.state[i].is_leaf():
                    self.done[i] = 1
                    next_state[i] = self.state[i]
                    reward[i] = self.layer[i] if target_node.wnid == self.state[i].wnid else -self.layer[i]
                else:
                    pred = self.state[i].sub_out[i].reshape(1, -1)
                    pred = torch.softmax(pred, dim=1).data.max(1)[1]
                    next_state[i] = self.state[i].children[pred.item()]

                    if label2id[self.state[i].wnid] in lp:
                        reward[i] = self.layer[i]
                    else:
                        reward[i] = -self.layer[i]
                    # dist_curr = target_node.node_distance(self.state[i], label2id, lpaths)
                    # dist_next = target_node.node_distance(next_state[i], label2id, lpaths)
                    # reward[i] = 1 if dist_curr > dist_next else -1
                    self.layer[i] += 1

                # dist_curr = target_node.node_distance(self.state[i], label2id, lpaths)
                # dist_next = target_node.node_distance(next_state[i], label2id, lpaths)
                # # reward[i] = dist_curr - dist_next
                # reward[i] = self.layer[i] if dist_curr > dist_next else -dist_next

                # if self.layer[i] >= lp_len:
                #     reward[i] = -self.layer[i]
                # else:
                #     corr_node = node_dict[id2label[lp[self.layer[i]-1]]]
                #     dist_curr = corr_node.node_distance(self.state[i], label2id, lpaths)
                #     dist_next = corr_node.node_distance(next_state[i], label2id, lpaths)
                #     reward[i] = dist_curr - dist_next

            elif action[i] == 2:
                self.done[i] = 1
                next_state[i] = self.state[i]
                reward[i] = self.layer[i] if target_node.wnid == self.state[i].wnid else -self.layer[i]

        self.state = next_state
        return next_state, reward, self.done, {}
