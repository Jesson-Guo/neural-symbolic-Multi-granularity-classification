import torch
import os
import numpy as np

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
    def __init__(self, root, label2id) -> None:
        super().__init__()
        self.root = root
        self.label2id = label2id

    def build_tree(self, node, checkpoint):
        def build(node, weights):
            if node.is_leaf():
                # TODO Resnet最后fc层的参数列向量对应一个权重leaf[0].weight = fc[:,0]
                index = self.label2id[node.id]
                node.set_weight(weights[:,index])
                return node.weight

            # 根据聚类方法更新父节点weights
            weight = 0
            for i in node.children.keys():
                weight += build(node.children[i], weights)
            weight /= node.nchild
            node.set_weight(weight)
            return weight

        weights = get_weights_from_checkpoint(checkpoint)
        build(node, weights)
        return node

    def infer(self, x):
        # 计算子节点weight与x的内积
        out = []
        node = self.root
        while not node.is_leaf():
            temp = []
            for k in node.children.keys():
                v = torch.dot(x, node.children[k].weight)
                temp.append(v.data)
            i = np.argmax(np.array(temp))
            out.append((node[i].id, node[i].name))
        return out
