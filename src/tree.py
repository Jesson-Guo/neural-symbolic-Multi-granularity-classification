import torch
import os
import numpy as np

from sklearn.cluster import AgglomerativeClustering


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


def build(node, weights):
    if node.is_leaf():
        # TODO 确定weights中向量与label之间的对应关系
        node.set_weight()
        return node.weight
    
    # 根据聚类方法更新父节点weights
    weight = 0
    for i in node.children.keys():
        weight += build(node.children[i], weights)
    weight /= node.nchild
    node.set_weight(weight)
    return weight


def build_tree(
    node,
    checkpoint,

):
    weights = get_weights_from_checkpoint(checkpoint)
    build(node, weights)
    return node

