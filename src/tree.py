import torch
import numpy as np
import copy

import torch.nn as nn
from utils.globals import *


inner_nodes = {}
node_dict = {}
label2id = {}
tree = None


class InferTree(nn.Module):
    def __init__(self, arch, num_classes, dim, criterion, lamb, device) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.criterion = criterion
        self.lamb = lamb
        self.device = device

        global tree, node_dict, label2id
        node_dict = get_value('node_dict')
        label2id = get_value('label2id')
        tree = get_value('tree')
        tree.prob = 1.
        tree.path_prob = 1.

        self.depth = 0
        if arch == 'cifar10':
            self.depth = 5
        elif arch == 'cifar100':
            self.depth = 10
        elif arch == 'tiny-imagenet':
            self.depth = 10
        elif arch == 'imagenet':
            self.depth = 13

        # if arch == 'cifar10':
        #     self.depth = 12
        # elif arch == 'cifar100':
        #     self.depth = 14
        # elif arch == 'tiny-imagenet':
        #     self.depth = 13
        # elif arch == 'imagenet':
        #     self.depth = 14

        for i in range(self.depth):
            inner_nodes[i+1] = []
        self.__build_tree()

        self.env_params = {}

    def __build_tree(self):
        def build(node, layer):
            node.set_layer(layer)
            layers = [
                # nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, bias=False),
                nn.Flatten(),
                nn.Linear(self.dim, node.num_child()+1)
            ]
            # 增加一个类别表示分类错误（需要返回到父节点）？，

            node.set_subid(get_value('label2id'))
            classifier = nn.Sequential(*layers).to(self.device)
            node.set_classifier(classifier)
            inner_nodes[node.layer].append(node)

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

        for l, nodes in inner_nodes.items():
            for node in nodes:
                if node.is_leaf():
                    continue
                sub_output = node.classifier(x)
                prob = torch.softmax(sub_output, dim=1)
                # sub_labels = []
                # for label in labels.cpu().numpy():
                #     lp = get_value('lpaths')[label]
                #     if len(lp) <= l:
                #         sub_labels.append(0)
                #     else:
                #         sub_labels.append(node.get_subid(lp[l]))
                # sub_labels = torch.as_tensor(sub_labels).to(self.device)

                # penalty += self.criterion(prob, sub_labels, len(node.children)+1) / l

                node.sub_prob = copy.copy(prob).data
                for i, child in node.children.items():
                    # child.prob = sub_output[:, i+1]
                    child.path_prob = prob[:, i+1] * node.path_prob
                    if child.is_leaf():
                        idx = get_value('label2id')[child.wnid]
                        out[:, idx] += child.path_prob

        return out[:, 1:], penalty

    def infer(self, x):
        out = torch.zeros([x.shape[0], self.num_classes+1], dtype=torch.float32).to(self.device)

        for l, nodes in inner_nodes.items():
            for node in nodes:
                if node.is_leaf():
                    continue
                sub_output = node.classifier(x)
                prob = torch.softmax(sub_output, dim=1)
                node.sub_prob = copy.copy(prob).data
                for i, child in node.children.items():
                    child.prob = prob[:, i+1]
                    child.path_prob = prob[:, i+1] * node.path_prob
                    if child.is_leaf():
                        idx = get_value('label2id')[child.wnid]
                        out[:, idx] += child.path_prob
        return out[:, 1:]
