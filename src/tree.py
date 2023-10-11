import torch
import os
import numpy as np
import copy
import queue

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
    def __init__(self, root, label2id, num_classes, lamb, device) -> None:
        super().__init__()
        self.root = root
        self.label2id = label2id
        self.num_classes = num_classes
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

    # def format_tree(self):
    #     for _, leaf in self.leaves.items():
    #         if leaf.layer < self.depth:
    #             for i in range(leaf.layer+1, self.depth+1):
    #                 sub = copy.deepcopy(leaf)
    #                 sub.set_layer(i)
    #                 leaf.update_child(sub)
    #                 leaf = sub

    def build_tree(self):
        def build(node, layer):
            if node.is_leaf():
                return layer
            layers = [
                # nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Flatten(),
                nn.Linear(512, node.num_child()),
                nn.Sigmoid()
            ]
            # 增加一个类别表示分类错误（需要返回到父节点）？，
            # TODO 训练时batch_size是否可以 >1?
            # if node.num_child() > 1:
            #     layers.append(nn.Linear(512, node.num_child()+1))
            # else:
            #     layers.append(nn.Linear(512, 1))

            node.set_subid()
            classifier = nn.Sequential(*layers)
            classifier.to(self.device)
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

    def forward(self, x):
        loss = 0.
        # 使用小分类器进行分类, 可并行
        # TODO 一个高效的训练方法，将所有的小分类器拼成一个大的，最后再作分割，可以不使用for循环

        penalty = torch.tensor(0.0).to(self.device)
        out = torch.zeros([x.shape[0], self.num_classes], dtype=torch.float64).to(self.device)

        ll = 0

        for l, nodes in self.sub_node_classifier.items():
            for nc in nodes:
                prob = nc.classifier(x)

                i = 0
                for _, child in nc.children.items():
                    child.prob = prob[:, i]
                    child.path_prob = prob[:, i] * nc.path_prob

                    alpha = torch.sum(child.path_prob * child.prob, 0) / torch.sum(child.path_prob, 0)
                    penalty += torch.log(alpha)

                    # 有些子节点出现在了不同的分支下
                    if child.is_leaf():
                        idx = self.label2id[child.id]
                        out[:, idx] += child.path_prob
                        ll += 1

                    i += 1

                # calculate penalty
                penalty += penalty * self.penalty_list[l] / len(nc.children)

        return out, -penalty

    # def infer(self, x):
    #     node = self.root
    #     out = [self.label2id[node.id]]
    #     # TODO 一个高效的训练方法，将所有的小分类器拼成一个大的，最后再作分割，可以不使用for循环
    #     while not node.is_leaf():
    #         pred = node.classifier(x)
    #         _, pred = torch.max(pred.data, 1)
    #         pred = pred.item()
    #         if pred == 0:
    #             break
    #         # 有一个错误类
    #         # TODO 当node只有一个子节点时需要可以不考虑错误类？
    #         pred_index = list(node.children.keys())[pred-1]
    #         pred_node = node.children[pred_index]
    #         out.append(self.label2id[pred_node.id])
    #         node = pred_node
    #     return out


# class FastInferTree(nn.Module):
#     def __init__(self, root, label2id, criterion, device) -> None:
#         super().__init__()
#         self.root = root
#         self.label2id = label2id
#         self.criterion = criterion
#         self.device = device

#         self.depth = 0
#         self.inner_node = {}
#         self.num_inner_node = 0
#         self.leaves = []
#         self.sub_node_classifier = None
#         self.depth = 0
#         self.__init_tree()

#     def __init_tree(self):
#         self.inner_node[1] = [1]
#         self.num_inner_node += 1

#         que = queue.Queue()
#         que.put(self.root)

#         while not que.empty():
#             self.depth += 1
#             que_size = que.size()
#             self.inner_node[self.depth+1] = []

#             for i in range(que_size):
#                 node = que.get()
#                 self.inner_node[self.depth+1].append(len(node.children))
#                 self.num_inner_node += len(node.children)

#                 for _, child in node.children.items():
#                     que.put(child)

#     def build_tree(self):
#         layers = [
#             # nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.Flatten(),
#             nn.Linear(512, self.num_inner_node),
#             nn.Sigmoid()
#         ]
#         self.sub_node_classifier = nn.Sequential(*layers)
#         self.sub_node_classifier = self.sub_node_classifier.to(self.device)

#     def train(self, x, label_paths):
#         out = {}
#         loss = 0.
#         # 使用小分类器进行分类, 可并行
#         # TODO 一个高效的训练方法，将所有的小分类器拼成一个大的，最后再作分割，可以不使用for循环
#         prob = self.sub_node_classifier(x)
#         path_prob = prob[:, 0]
#         begin = 1
#         penalty = torch.tensor(0.0).to(self.device)

#         que = queue.Queue()
#         que.put(self.root)

#         while not que.empty():
#             que_size = que.size()
#             for i in range(que_size):
#                 node = que.get()
#                 if node.is_leaf():
#                     pass
#                 else:
#                     self.label2id[node.id]

#                 for _, child in node.children.items():
#                     que.put(child)

#         for i in range(1, self.depth):
#             num = sum(self.inner_node[i+1])
#             for j in range(begin, begin+num):

#             layer_prob = prob[:, begin:begin+num]


#         node_prob = []
#         path_prob = []



#         for nc in self.sub_node_classifier:
#             sub_labels = []
#             layer = nc.layer
#             for lp in label_paths:
#                 if layer >= len(lp):
#                     sub_labels.append(0)
#                 else:
#                     sub_labels.append(nc.get_subid(lp[layer]))
#             sub_labels = torch.as_tensor(sub_labels)
#             sub_labels = torch.autograd.Variable(sub_labels)
#             sub_labels = sub_labels.to(self.device)

#             sub_out = nc.classifier(x)
#             sub_loss = self.criterion(sub_out, sub_labels)

#             out[nc] = sub_out
#             loss += sub_loss
#         return out, loss

#     def infer(self, x):
#         node = self.root
#         out = [self.label2id[node.id]]
#         # TODO 一个高效的训练方法，将所有的小分类器拼成一个大的，最后再作分割，可以不使用for循环
#         while not node.is_leaf():
#             pred = node.classifier(x)
#             _, pred = torch.max(pred.data, 1)
#             pred = pred.item()
#             if pred == 0:
#                 break
#             # 有一个错误类
#             # TODO 当node只有一个子节点时需要可以不考虑错误类？
#             pred_index = list(node.children.keys())[pred-1]
#             pred_node = node.children[pred_index]
#             out.append(self.label2id[pred_node.id])
#             node = pred_node
#         return out
