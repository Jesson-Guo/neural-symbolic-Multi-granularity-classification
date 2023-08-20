import numpy as np
import queue


class SymbolicInference(object):
    def __init__(self, root, miu, sigma) -> None:
        self.root = root
        # 或者设置成超参数
        self.miu = miu
        self.sigma = sigma
        self.depth = self._tree_depth()

    def _tree_depth(self):
        def helper(node):
            if not node:
                return 0
            if not node.children:
                self.miu[node.id] = 0
                self.sigma[node.id] = 0
                return 1
            depth = 0
            self.miu[node.id] = 0
            self.sigma[node.id] = 0
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
        return node.children[child_key]

    def infer(self, root, depth, feat_list):
        infer_path = []
        infer_path.append(root)

        node = root
        num_feat = len(feat_list)
        infer_layer = 1
        while not node.is_leaf():
            # feature selection
            feat = feat_list[(infer_layer*num_feat)//depth]
            node = self._infer_child(node, feat)
            infer_path.append(node)
            infer_layer += 1

        return infer_path
            
