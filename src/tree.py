import torch
import os
import numpy as np
import copy
import queue

import torch.nn as nn
from src.hierarchy import label2id, lpaths


# 决策树非叶节点用一个全连接层 或者 计算内积
class InferTree(nn.Module):
    def __init__(self, root, num_classes, dim, criterion, lamb, device) -> None:
        super().__init__()
        self.root = root
        self.num_classes = num_classes
        self.dim = dim
        self.criterion = criterion
        self.lamb = lamb
        self.device = device

        self.depth = 0
        self.leaves = []
        self.sub_node_classifier = {}
        self.depth = self.__get_tree_depth()

        self.penalty_list = [
            self.lamb * (2 ** (-depth)) for depth in range(0, self.depth)
        ]

        for i in range(self.depth):
            self.sub_node_classifier[i+1] = []
        
        self.root.prob = 1.
        self.root.path_prob = 1.

    def __get_tree_depth(self):
        def depth(node, layer):
            node.set_layer(layer)
            if node.is_leaf():
                self.leaves.append(node)
                return layer
            tree_depth = 0
            for i in node.children.keys():
                tree_depth = max(tree_depth, depth(node.children[i], layer+1))
            return tree_depth

        node = self.root
        return depth(node, 1)

    def build_tree(self):
        def build(node, layer):
            if node.is_leaf():
                return layer
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

            node.set_subid()
            classifier = nn.Sequential(*layers).to(self.device)
            node.set_classifier(classifier)
            self.sub_node_classifier[node.layer].append(node)

            depth = 0
            for i in node.children.keys():
                depth = max(depth, build(node.children[i], layer+1))
            return depth

        node = self.root
        if node.is_leaf() or node == None:
            # node error
            return None
        self.depth = build(node, 1)
        return node

    def forward(self, x, labels):
        # 使用小分类器进行分类, 可并行
        # TODO 一个高效的训练方法，将所有的小分类器拼成一个大的，最后再作分割，可以不使用for循环

        penalty = torch.tensor(0.0).to(self.device)
        out = torch.zeros([x.shape[0], self.num_classes+1], dtype=torch.float32).to(self.device)

        ll = 0

        for l, nodes in self.sub_node_classifier.items():
            for node in nodes:
                sub_output = node.classifier(x)
                sub_labels = []
                for label in labels.cpu().numpy():
                    lp = lpaths[label]
                    if len(lp) <= l:
                        sub_labels.append(0)
                    else:
                        sub_labels.append(node.get_subid(lp[l]))
                sub_labels = torch.as_tensor(sub_labels).to(self.device)

                # penalty += self.criterion(sub_output, sub_labels) * self.penalty_list[l]
                penalty += self.criterion(sub_output, sub_labels) / l

                for i, child in node.children.items():
                    child.prob = sub_output[:, i]
                    child.path_prob = sub_output[:, i] * node.path_prob

                    # alpha = torch.sum(child.path_prob * child.prob, 0) / torch.sum(child.path_prob, 0)
                    # penalty += torch.log(alpha)

                    # 有些子节点出现在了不同的分支下
                    if child.is_leaf():
                        idx = label2id[child.wnid]
                        out[:, idx] += child.prob
                        ll += 1

                # calculate penalty
                # penalty += penalty * self.penalty_list[l] / len(nc.children)

        # penalty = -penalty

        return out, penalty
