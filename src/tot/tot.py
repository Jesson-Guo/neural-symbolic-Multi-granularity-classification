import torch
import copy
import random
import traceback
import numpy as np

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
    def __init__(self, sim_func, plan_dict, num_classes, root=None) -> None:
        self.sim_func = sim_func
        self.plan_dict = plan_dict
        self.num_coarses = 0
        self.root = root

        self.num_others = 0

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
        self.thought_dict, temp = {}, {}
        for k in self.plan_dict.keys():
            # 需要判断一下root，root tid = -1
            temp[k] = self.num_coarses
            self.num_coarses += 1

        self.num_others = self.num_coarses
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
            if t.name == "Other":
                t.tid = self.num_others
                self.num_others += 1
                continue

            label_list = list(t.labels.keys())
            label_list.sort()
            label_str = str(label_list)[1:-1]
            t.tid = temp[label_str]
            self.thought_dict[t.tid] = t

    def estimate_clusters(self, v, plan, plan_w, ts, func):
        similarity = func(v, plan_w)
        idx = similarity.argmin().item()
        t = ts[idx]
        score = similarity[idx].item()
        for name, _ in plan_w.items():
            if idx == 0:
                break
            idx -= 1
        name_t = copy.deepcopy(t.name)
        name_t.append(name)
        return name_t, t, score

    def choose_plan(self, thought: Thought, node_dict, label_to_wnid):
        plans = []
        for ts in thought.plans.values():
            # 启发式的选取，likely即可
            if ts[0].is_valid():
                plan, plan_w = {}, {}
                for t in ts:
                    plan[t.name] = []
                    plan_w[t.name] = []
                    for item in t.labels.values():
                        plan[t.name].append(item)
                        plan_w[t.name].append(node_dict[label_to_wnid[item]].weight.data)
                plans.append((plan, plan_w, ts))
        random.shuffle(plans)
        return plans

    def solve_once(self, v, plan, plan_w, ts):
        name, t, score = self.estimate_clusters(v, plan, plan_w, ts, self.sim_func)
        r = Result(name, STATUS[t.feedback], score)
        return t, r

    def dfs(self, v, node_dict, label_to_wnid, alpha, thought: Thought):
        def helper(thought: Thought, res: Result, ok: bool):
            if not thought.is_valid():
                return False, None, 1
            if thought.stop():
                l = list(thought.labels.values())[0]
                plan_w = {thought.name: [node_dict[label_to_wnid[l]].weight.data]}
                similarity = self.sim_func(v, plan_w)
                score = similarity.min().item()
                del similarity

                r = Result(thought.name, STATUS[thought.feedback], score, parent=result)
                res.add(r)
                return True, l, score

            score = 1
            label = None
            plans = self.choose_plan(thought, node_dict, label_to_wnid)
            for plan, plan_w, ts in plans:
                t, r = self.solve_once(v, plan, plan_w, ts)
                res.add(r)
                if r.score > 0:
                    r.status = STATUS[0]
                    continue
                ok, label, score = helper(t, r, ok)
                if ok and score < -alpha:
                    break
                ok = False

            if label == None:
                scores = []
                for i in range(len(res.children)):
                    scores.append(res.children[i].score)
                idx = np.argmax(scores)
                res.children[idx].status = 1

                plan, plan_w, ts = plans[idx]
                t, r = self.solve_once(v, plan, plan_w, ts)
                ok, label, score = helper(t, r, ok)
            return ok, label, score

        result = Result(thought.name, STATUS[thought.feedback], 0)
        _, label, _ = helper(thought, result, False)
        return result, label

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

    def solve(self, v, node_dict, label_to_wnid, alpha, method='dfs'):
        method = getattr(self, method)
        output = method(v, node_dict, label_to_wnid, alpha, self.root)
        return output
