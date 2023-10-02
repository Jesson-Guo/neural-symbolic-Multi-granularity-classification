import numpy as np
import copy

import torch


class SymbolicInference(object):
    def __init__(self, root, miu, sigma, lpaths, label2id) -> None:
        self.root = root
        # 或者设置成超参数
        self.miu = miu
        self.sigma = sigma
        self.lpaths = lpaths
        self.label2id = label2id
        self.depth = self._tree_depth()

        self.training = False

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

    def _fuzzy_similarity_degree(self, x, l, device):
        mean = torch.as_tensor([self.miu[l]]).to(device)
        std = torch.as_tensor([self.sigma[l]]).to(device)

        return torch.exp(-torch.divide((x-mean).pow(2), 2*std.pow(2)))

    def _infer_child(self, node, feat, device):
        fuzzy_membership = []
        fuzzy_membership_value = []
        for i in node.children.keys():
            child = node.children[i]
            degree = self._fuzzy_similarity_degree(feat, child.id, device)
            fuzzy_membership.append(degree)
            fuzzy_membership_value.append(degree.item())
        max_key = np.argmax(np.array(fuzzy_membership_value))

        max_mem = fuzzy_membership[max_key]
        sec_max_mem = 0.
        if len(fuzzy_membership_value) > 1:
            sec_key = np.argsort(np.array(fuzzy_membership_value))[-2]
            sec_max_mem = fuzzy_membership[sec_key]

        child_key = list(node.children.keys())[max_key]

        return node.children[child_key], max_mem, sec_max_mem

    def infer(self, feat_list, labels=None, device='cpu'):
        batch_size = feat_list[0].shape[0]
        root = copy.deepcopy(self.root)
        out = []
        loss = torch.tensor([0], dtype=torch.float64).to(device)
        for i in range(batch_size):
            infer_path = [self.label2id[root.id]]
            lpath = []
            if self.training:
                lpath = copy.deepcopy(self.lpaths[labels[i].item()])
                lpath.reverse()
            diffs = []

            node = root
            num_feat = len(feat_list)
            infer_layer = 1
            while not node.is_leaf():
                if self.training and infer_layer >= len(lpath):
                    break
                # feature selection
                feat = feat_list[(infer_layer*num_feat)//self.depth][i]
                node, max_mem, sec_max_mem = self._infer_child(node, feat, device)
                if self.training:
                    diff = max_mem - sec_max_mem
                    if self.label2id[node.id] != lpath[infer_layer]:
                        diff *= 0
                    loss += (1-diff).pow(2)*lpath[infer_layer]
                else:
                    infer_path.append(self.label2id[node.id])
                    diffs.append(max_mem.item())
                infer_layer += 1
            del lpath
            if not self.training:
                out.append((infer_path, diffs))
        del root
        if self.training:
            return loss
        else:
            return out

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
