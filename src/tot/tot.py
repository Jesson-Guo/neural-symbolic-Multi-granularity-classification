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
        self.name_cache = {}
        self.leaves_cnt = np.zeros((num_classes))

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

    def reset(self, cfg, samples_per_cls):
        temp = {}
        num_classes = self.num_coarses
        for k in self.plan_dict.keys():
            # 需要判断一下root，root tid = -1
            if len(k.split(",")) == num_classes:
                continue
            temp[k] = self.num_coarses
            self.num_coarses += 1

        ts = [self.root]
        while len(ts):
            t = ts.pop()
            if t.stop():
                t.tid = list(t.labels.keys())[0]
                self.thought_dict[t.tid] = t
                self.leaves_cnt[t.tid] += 1
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
        self.data_caches = {-1: sum(samples_per_cls)}
        for i, t in self.thought_dict.items():
            subset_list = list(t.labels.keys())
            subset_num_sample_list = [samples_per_cls[int(s)] for s in subset_list]
            self.data_caches[i] = sum(subset_num_sample_list)

        for k, plans in self.plan_dict.items():
            self.thought_cache[k] = {}
            for i, ts in plans.items():
                if ts[0].is_valid():
                    self.thought_cache[k][i] = {"tids": [], "samples_per_cls": [], "coarse_targets": {}, "do_loss": False}
                    for j in range(len(ts)):
                        self.thought_cache[k][i]["samples_per_cls"].append(self.data_caches[ts[j].tid])
                        self.thought_cache[k][i]["tids"].append(ts[j].tid)
                        for l in ts[j].labels.keys():
                            self.thought_cache[k][i]["coarse_targets"][l] = j
                        if not self.thought_cache[k][i]["do_loss"] and not ts[j].stop():
                            self.thought_cache[k][i]["do_loss"] = True

        for k in self.plan_dict.keys():
            self.name_cache[k] = {}
            for j, ts in self.plan_dict[k].items():
                self.name_cache[k][j] = []
                for t in ts:
                    self.name_cache[k][j].append(t.name)

    def dfs(self, idx, x, alpha):
        dq = deque()
        result = Result(self.root.name, STATUS[self.root.feedback], 1)
        dq.append((self.root, result))
        pred = -1
        # candidates 可以设置一个阈值
        candidates = {"score": [], "tids": []}
        # candidates = {"score": {}, "tids": {}}
        while len(dq):
            t, r = dq.pop()
            if t.stop():
                if r.status > 0:
                    # if self.judge == "path":
                    #     score = r.score
                    # elif self.judge == "score":
                    #     score = x[idx, t.tid]
                    candidates["score"].append(r.score)
                    candidates["tids"].append(t.tid)
                    # if t.tid not in candidates:
                    #     candidates["score"][t.tid] = r.score
                    #     candidates["tids"][t.tid] = 1
                    # else:
                    #     candidates["score"][t.tid] += r.score
                    #     candidates["tids"][t.tid] += 1
                    if len(candidates["score"]) == alpha:
                        break
                continue

            label_list = list(t.labels.keys())
            label_list.sort()
            label_str = str(label_list)[1:-1]

            for caches in self.thought_cache[label_str].values():
                tids = caches["tids"]
                scores = x[idx, tids].unsqueeze(0)
                out = scores.softmax(dim=1)
                pred = out.data.max(1)[1].item()
                # top2 = out.data.topk(2, 1, True, True)[0].squeeze()
                score = out[0, pred].data.item()
                score = score * r.score
                # if t.tid == -1:
                #     score = scores[0, pred]
                # else:
                #     score = (scores[0, pred] + r.score) / 2
                tt = self.thought_dict[tids[pred]]
                # status = 0 if (top2[0] - top2[1]) / top2[0] < 0.5 else tt.feedback
                status = tt.feedback
                res = Result(tt.name, status, score, r)
                r.add(res)
                dq.append((tt, res))

        if len(candidates["score"]) == 0:
            return random.randint(0, self.leaves_cnt.shape[0]-1), 1
        # candidates_scores, candidates_tids = [], []
        # for k in candidates["score"].keys():
        #     candidates_scores.append(candidates["score"][k] / candidates["tids"][k])
        #     candidates_tids.append(k)
        # a = torch.FloatTensor(candidates_scores).argmax()
        # pred = candidates_tids[a]
        a = torch.FloatTensor(candidates["score"]).argmax()
        pred = candidates["tids"][a]
        return pred, 0
        # return pred, result

    def bfs(self, idx, x, alpha):
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

    def solve(self, x, targets, alpha, method='dfs'):
        method = getattr(self, method)
        pred = torch.LongTensor((x.shape[0])).to(x.device)
        cnt = [0 for _ in range(self.leaves_cnt.shape[0])]
        for i in range(x.shape[0]):
            pred[i], c = method(i, x, alpha)
            cnt[targets[i]] += c
        return pred, cnt
