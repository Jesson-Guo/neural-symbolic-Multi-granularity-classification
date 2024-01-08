import torch
import copy
import random
import traceback
import queue
import numpy as np

from collections import deque
from src.gpt import GPT
from utils.util import Result


STATUS = {
    2: "sure",
    1: "likely",
    0: "impossible"
}


class Thought(object):
    def __init__(self, labels, feedback=2, parent=None, name="") -> None:
        self.labels = labels
        self.feedback = feedback
        self.parent = parent
        self.plans = {}
        self.name = name

        self.tid = -1
        self.score = None
        self.path_score = None

    def is_valid(self):
        return self.feedback > 0

    def stop(self):
        return len(self.labels) == 1

    def add_child(self, num, t):
        if num not in self.plans:
            self.plans[num] = []
        self.plans[num].append(t)

    def update_parent(self, t):
        self.parent = t

    def to_dict(self):
        label_list = [k for k in self.labels.keys()]
        return {"feedback": self.feedback, "labels": str(label_list), "name": self.name, "plans": {}}


class ToT:
    def __init__(self, num_classes=0, sim_func=None, plan_dict=None, root=None) -> None:
        self.sim_func = sim_func
        self.plan_dict = plan_dict
        self.num_coarses = num_classes
        self.root = root

        self.thought_dict = {}
        self.thought_cache = {}

    def clean(self):
        thoughts = [self.root]
        while len(thoughts):
            t = thoughts.pop()
            t.score = None
            t.path_score = None
            for ts in t.plans.values():
                for i in range(len(ts)):
                    thoughts.append(ts[i])

        self.root.score = 1
        self.root.path_score = 1

    def reset(self):
        temp = {}
        for k in self.plan_dict.keys():
            # 需要判断一下root，root tid = -1
            temp[k] = self.num_coarses
            self.num_coarses += 1

        ts = [self.root]
        while len(ts):
            t = ts.pop()
            if t.stop():
                t.tid = list(t.labels.keys())[0]
                self.thought_dict[t.tid] = t
                continue

            for _ in t.plans.values():
                for child in _:
                    ts.insert(0, child)

            if t.name == "Thing":
                continue

            label_list = list(t.labels.keys())
            label_list.sort()
            label_str = str(label_list)[1:-1]
            t.tid = temp[label_str]
            self.thought_dict[t.tid] = t

        del temp
        for k, plans in self.plan_dict.items():
            self.thought_cache[k] = {}
            for i, ts in plans.items():
                if ts[0].is_valid():
                    self.thought_cache[k][i] = {"tids": [], "coarse_targets": {}, "do_loss": False}
                    for j in range(len(ts)):
                        self.thought_cache[k][i]["tids"].append(ts[j].tid)
                        for l in ts[j].labels.keys():
                            self.thought_cache[k][i]["coarse_targets"][l] = j
                        if not self.thought_cache[k][i]["do_loss"] and not ts[j].stop():
                            self.thought_cache[k][i]["do_loss"] = True

    def dfs(self, idx, x, alpha):
        dq = deque()
        result = Result(self.root.name, STATUS[self.root.feedback], 0)
        dq.append((self.root, result))
        ok = False
        pred = -1
        while not ok and len(dq):
            t, r = dq.pop()
            if t.stop():
                score = x[idx, t.tid]
                r_c = Result(t.name, STATUS[t.feedback], score, r)
                r.add(r_c)
                if score > alpha:
                    ok = True
                    pred = t.tid
                continue

            label_list = list(t.labels.keys())
            label_list.sort()
            label_str = str(label_list)[1:-1]

            for caches in self.thought_cache[label_str].values():
                tids = caches["tids"]
                scores = x[idx, tids]
                out = scores.softmax(dim=1)
                pred = out.data.max(1)[1].item()
                tt = self.thought_dict[tids[pred]]
                res = Result(tt.name, STATUS[tt.feedback], scores[pred], r)
                dq.append((tt, res))
        return pred, result


    def bfs(self, v, node_dict, label_to_wnid, alpha, thought: Thought):
        pass
        # result = Result(thought.name, STATUS[thought.feedback])
        # thoughts = [thought]
        # while len(thoughts):
        #     t = thoughts.pop()
        #     if not t.is_valid():
        #         continue
        #     if t.stop():
        #         r = Result(t.labels[0], STATUS[t.feedback], parent=res)
        #         res.add(r)
        #         break
        #     self.solve_once(v, node_dict, label_to_wnid, t, gpt, res)
        #     for _, child in t.plans.items():
        #         thoughts.append(child)
        # return name

    def solve(self, x, alpha, method='dfs'):
        method = getattr(self, method)
        outputs = torch.LongTensor((x.shape[0]))
        for i in range(x.shape[0]):
            outputs[i], _ = method(i, x, alpha)
        return outputs
