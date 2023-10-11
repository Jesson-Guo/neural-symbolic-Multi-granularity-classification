import torch
import os
import numpy as np
import copy

import torch.nn as nn


MODEL_FC_KEYS = (
    "fc.weight",
    "linear.weight",
    "module.linear.weight",
    "module.net.linear.weight",
    "output.weight",
    "module.output.weight",
    "output.fc.weight",
    "module.output.fc.weight",
    "classifier.weight",
    "model.last_layer.3.weight",
)


def get_weights_from_state_dict(state_dict):
    fc = None
    for key in MODEL_FC_KEYS:
        if key in state_dict:
            fc = state_dict[key].squeeze()
            break
    if fc is not None:
        return fc.detach()


def get_weights_from_checkpoint(checkpoint):
    data = torch.load(checkpoint, map_location=torch.device("cpu"))

    for key in ("net", "state_dict"):
        try:
            state_dict = data[key]
            break
        except:
            state_dict = data

    fc = get_weights_from_state_dict(state_dict)
    return fc


# 决策树非叶节点用一个全连接层 或者 计算内积
class InferTree(nn.Module):
    def __init__(self, root, label2id, criterion, device) -> None:
        super().__init__()
        self.root = root
        self.label2id = label2id
        self.criterion = criterion
        self.device = device

        self.depth = 0
        self.leaves = {}
        self.sub_node_classifier = []
        self.depth = self.__get_tree_depth()

    def __get_tree_depth(self):
        def depth(node, layer):
            node.set_layer(layer)
            if node.is_leaf():
                self.leaves[node.id] = node
                return layer
            tree_depth = 0
            for i in node.children.keys():
                tree_depth = max(tree_depth, depth(node.children[i], layer+1))
            return tree_depth

        node = self.root
        return depth(node, 1)

    def format_tree(self):
        for _, leaf in self.leaves.items():
            if leaf.layer < self.depth:
                for i in range(leaf.layer+1, self.depth+1):
                    sub = copy.deepcopy(leaf)
                    sub.set_layer(i)
                    leaf.update_child(sub)
                    leaf = sub

    def build_tree(self):
        def build(node, layer):
            if node.is_leaf():
                return layer
            layers = [
                # nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Flatten(),
                # 增加一个类别表示分类错误（需要返回到父节点）？，
                # TODO 训练时batch_size是否可以 >1?
                nn.Linear(512, node.num_child()+1)
            ]
            node.set_subid()
            classifier = nn.Sequential(*layers)
            classifier.to(self.device)
            node.set_classifier(classifier)
            self.sub_node_classifier.append(node)
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

    def infer(self, x, label_paths):
        out = {}
        loss = 0.
        # 使用小分类器进行分类, 可并行
        # TODO 一个高效的训练方法，将所有的小分类器拼成一个大的，最后再作分割，可以不使用for循环
        for nc in self.sub_node_classifier:
            sub_labels = []
            layer = nc.layer
            for lp in label_paths:
                if layer >= len(lp):
                    sub_labels.append(0)
                else:
                    sub_labels.append(nc.get_subid(lp[layer]))
            sub_labels = torch.as_tensor(sub_labels)
            sub_labels = torch.autograd.Variable(sub_labels)
            sub_labels = sub_labels.to(self.device)

            sub_out = nc.classifier(x)
            sub_loss = self.criterion(sub_out, sub_labels)

            out[nc] = sub_out
            loss += sub_loss
        return out, loss
