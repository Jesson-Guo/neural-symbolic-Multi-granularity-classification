import numpy as np
import copy

import torch.nn as nn


class SymbolicInference(nn.Module):
    def __init__(self, root, miu, sigma, lpaths) -> None:
        self.root = root
        # 或者设置成超参数
        self.miu = miu
        self.sigma = sigma
        self.lpaths = lpaths
        self.depth = self._tree_depth()

    def _tree_depth(self):
        def helper(node):
            if not node:
                return 0
            if not node.children:
                return 1
            depth = 0
            for i in node.children.keys():
                depth = max(depth, helper(node.children[i]))
            return depth + 1

        return helper(self.root)

    def _fuzzy_similarity_degree(self, x, l):
        return np.exp(-np.divide(np.float_power(x-self.miu[l], 2), 2*np.float_power(self.sigma[l], 2)))

    def _infer_child(self, node, feat):
        fuzzy_membership = []
        for i in node.children.keys():
            child = node.children[i]
            fuzzy_membership.append(self._fuzzy_similarity_degree(feat, child.id))
        child_key = np.argmax(np.array(fuzzy_membership))
        max_mem = np.max(fuzzy_membership)

        sec_max_mem = np.sort(fuzzy_membership)[-2]
        return node.children[child_key], max_mem, sec_max_mem

    def infer(self, feat_list, labels):
        batch_size = feat_list[0].shape[0]
        root = copy.deepcopy(self.root)
        out = []
        loss = 0
        for i in range(batch_size):
            infer_path = [root]
            lpath = []
            if self.training:
                lpath = self.lpaths[labels[i]].reverse()
            diffs = []

            node = root
            num_feat = len(feat_list)
            infer_layer = 1
            while not node.is_leaf():
                # feature selection
                feat = feat_list[(infer_layer*num_feat)//self.depth][i]
                node, max_mem, sec_max_mem = self._infer_child(node, feat)
                if self.training:
                    diff = max_mem - sec_max_mem
                    if node.id != lpath[infer_layer]:
                        diff = 0
                    diffs.append(np.power(1-diff, 2)*lpath[infer_layer])
                else:
                    infer_path.append(node.id)
                    diffs.append(max_mem)
                infer_layer += 1
            if self.training:
                loss += np.sum(np.array(diffs))
            else:
                out.append((infer_path, diffs))
        if self.training:
            return loss
        else:
            return out

    def forward(self, feat_list, labels):
        return self.infer(feat_list, labels)
