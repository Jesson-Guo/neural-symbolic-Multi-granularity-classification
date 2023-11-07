import torch
import os
import numpy as np
import copy
import queue

import torch.nn as nn
from utils.globals import *


sub_node_classifier = {}
node_hash = {}
tree = None


# 决策树非叶节点用一个全连接层 或者 计算内积
class InferTree(nn.Module):
    def __init__(self, num_classes, dim, criterion, lamb, device, ckpt) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.criterion = criterion
        self.lamb = lamb
        self.device = device

        global tree
        tree = get_value('tree')
        tree.prob = 1.
        tree.path_prob = 1.

        self.depth = 0
        self.depth = self.__get_tree_depth()

        for i in range(self.depth):
            sub_node_classifier[i+1] = []
        self.__build_tree()

        self.num_inner_node = 0
        # inner node 不考虑错误分类节点
        for _, nodes in sub_node_classifier.items():
            for node in nodes:
                node_hash[node] = self.num_inner_node
                self.num_inner_node += len(node.children)

        self.penalty_list = [
            self.lamb * (2 ** (-depth)) for depth in range(0, self.depth)
        ]

    def __get_tree_depth(self):
        def depth(node, layer):
            node.set_layer(layer)
            if node.is_leaf():
                return layer
            tree_depth = 0
            for i in node.children.keys():
                tree_depth = max(tree_depth, depth(node.children[i], layer+1))
            return tree_depth

        return depth(tree, 1)

    def __build_tree(self):
        def build(node, layer):
            if node.is_leaf():
                return 0
            layers = [
                # nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Flatten(),
                nn.Linear(self.dim, node.num_child()+1)
            ]
            # 增加一个类别表示分类错误（需要返回到父节点）？，
            # TODO 训练时batch_size是否可以 >1?
            # if node.num_child() > 1:
            #     layers.append(nn.Linear(self.dim, node.num_child()+1))
            # else:
            #     layers.append(nn.Linear(self.dim, 1))

            node.set_subid(get_value('label2id'))
            classifier = nn.Sequential(*layers).to(self.device)
            node.set_classifier(classifier)
            sub_node_classifier[node.layer].append(node)

            for i in node.children.keys():
                build(node.children[i], layer+1)
            return 0

        node = tree
        if node.is_leaf() or node == None:
            # node error
            return None
        build(node, 1)
        return node

    def forward(self, x, labels):
        penalty = torch.tensor(0.0).to(self.device)
        out = torch.zeros([x.shape[0], self.num_classes+1], dtype=torch.float32).to(self.device)

        for l, nodes in sub_node_classifier.items():
            for node in nodes:
                sub_output = node.classifier(x)
                node.sub_out = copy.copy(sub_output[:, 1:]).data
                sub_labels = []
                for label in labels.cpu().numpy():
                    lp = get_value('lpaths')[label]
                    if len(lp) <= l:
                        sub_labels.append(0)
                    else:
                        sub_labels.append(node.get_subid(lp[l]))
                sub_labels = torch.as_tensor(sub_labels).to(self.device)

                # penalty += self.criterion(sub_output, sub_labels) * self.penalty_list[l]
                penalty += self.criterion(sub_output, sub_labels) / l

                prob = torch.softmax(sub_output, dim=1)
                for i, child in node.children.items():
                    child.prob = sub_output[:, i+1]
                    child.path_prob = prob[:, i+1] * node.path_prob
                    # 有些子节点出现在了不同的分支下
                    if child.is_leaf():
                        idx = get_value('label2id')[child.wnid]
                        out[:, idx] += child.path_prob

        return out[:, 1:], penalty

    def infer(self, x):
        out = torch.zeros([x.shape[0], self.num_classes+1], dtype=torch.float32).to(self.device)

        for l, nodes in sub_node_classifier.items():
            for node in nodes:
                sub_output = node.classifier(x)
                node.sub_out = copy.copy(sub_output[:, 1:]).data
                prob = torch.softmax(sub_output, dim=1)
                for i, child in node.children.items():
                    child.prob = prob[:, i+1]
                    child.path_prob = prob[:, i+1] * node.path_prob
                    # 有些子节点出现在了不同的分支下
                    if child.is_leaf():
                        idx = get_value('label2id')[child.wnid]
                        out[:, idx] += child.path_prob
        return out[:, 1:]
