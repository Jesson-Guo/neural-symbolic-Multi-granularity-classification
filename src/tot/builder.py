import json
import copy
import random
import traceback
import numpy as np

from src.gpt import GPT
from utils.util import Result
from src.tot.tot import Thought, ToT


class ToTBuilder:
    def __init__(self, tot: ToT, num_plans, plan_func) -> None:
        self.tot = tot
        self.num_plans = num_plans
        self.plan_func = plan_func

    def estimate_plans(self, plans_w):
        result = []
        for plan_w in plans_w:
            out = self.plan_func(plan_w)
            # out是2/1/0表示sure/likely/impossible，后续也可以比较启发式和最优的差别
            # 这里也可以考虑将index传给gpt让gpt评价（可尝试）
            result.append(out)
        # 选择最好的（最大的），如果有些指标是越小越好需要先处理
        # idx = np.array(result).argmax()
        return result

    def build_on_tree(self, labels, tree, node_children):
        root = Thought(labels, name=['Thing'])
        thoughts = [(root, tree)]
        while len(thoughts):
            t, node = thoughts.pop()
            for node_child in node.children.values():
                name_t = copy.deepcopy(t.name)
                name_t.append(node_child.name)
                t_child = Thought(node_children[node_child.wnid], 2, t, name_t)
                t.add_child(0, t_child)
                thoughts.insert(0, (t_child, node_child))
        return root

    def check_plans(self, plans, labels):
        plans_t = copy.deepcopy(plans)
        for i in range(len(plans)):
            label_set = set()
            for l in labels.values():
                label_set.add(l)
            plan = plans[i]
            names = set()
            for name, ls in plan.items():
                ls_set = []
                names.add(name)
                for j in range(len(ls)):
                    if ls[j] in label_set():
                        label_set.remove(ls[j])
                        ls_set.append(ls[j])
                if len(ls_set):
                    plans_t[i][name] = list(ls_set)

            label_set = list(label_set)
            if len(label_set):
                if "Miscellaneous" not in names:
                    plans_t[i]["Miscellaneous"] = label_set
                else:
                    plans_t[i]["Miscellaneous"].extend(label_set)
        return plans_t

    def check(self):
        if self.root == None:
            return 0
        thoughts = [self.root]
        while len(thoughts):
            thought = thoughts.pop()
            for ts in thought.plans.values():
                for t in ts:
                    assert len(t.labels) < len(thought.labels), "child thought labels must be smaller than parent."
                    thoughts.append(t)

    def get_plans_with_weights(self, plans, weights):
        plans_w = []
        for plan in plans:
            plan_w = {}
            for name, label_ids in plan.items():
                ws = []
                for label_id in label_ids:
                    ws.append(weights[label_id].data)
                if len(ws):
                    plan_w[name] = ws
            if plan_w != {}:
                plans_w.append(plan_w)
        return plans_w

    def build_on_gpt(self, labels, weights, gpt: GPT, save_path, load_path=""):
        try:
            cnt = 0
            plan_dict = {}
            if load_path == "":
                self.root = Thought(labels, 2, name='Thing')
            else:
                self.load(load_path, labels)
            thoughts = [self.root]

            while len(thoughts):
                t = thoughts.pop()
                if (not t.is_valid()) or t.stop():
                    continue
                if len(t.labels) == 2 and len(t.plans) == 0:
                    for k, v in t.labels.keys():
                        thought = Thought({k: v}, 2, t, v)
                        t.add_child(0, thought)
                    t.add_child(0, Thought({}, 0, t, "Other"))
                    continue
                if len(t.plans) > 0:
                    for _ in t.plans.values():
                        for child in _:
                            thoughts.insert(0, child)
                    continue

                label_list = list(t.labels.keys())
                label_list.sort()
                label_str = str(label_list)[1:-1]
                if label_str not in plan_dict:
                    contents = gpt.generate(t.labels, num_plans=self.num_plans, cat_name=t.name, n=1)
                    cnt += 1

                    plans = gpt.gen_plans(contents)
                    plans = self.check_plans(plans, t.labels)
                    plans_w = self.get_plans_with_weights(self, plans, weights)

                    estimate = self.estimate_plans(plans_w)

                    if sum(estimate) == 0:
                        # re-generate 不考虑一直错
                        thoughts.append(t)
                        continue

                    for j in range(len(estimate)):
                        i = len(t.plans)
                        for name, ls in plans[j].items():
                            if len(ls) == 1:
                                name = ls[0]
                            if len(ls) < len(t.labels):
                                l_dict = {l: labels[l] for l in ls}
                                thought = Thought(l_dict, estimate[j], t, name)
                                t.add_child(i, thought)
                                thoughts.insert(0, thought)
                        t.add_child(i, Thought({}, 0, t, "Other"))

                    plan_dict[label_str] = t.plans
                else:
                    if t.feedback != 0 and len(t.labels) < len(t.parent.labels):
                        t.plans = plan_dict[label_str]
                if cnt % 10 == 0:
                    self.save(save_path)
            self.save(save_path)
        except Exception as e:
            self.save(save_path)
            print(e)
            print(traceback.format_exc())

    def save(self, save_path):
        out = self.root.to_dict()
        que = [(self.root, out)]

        while len(que):
            t, t_dict = que.pop()
            if len(t.plans) == 0:
                continue
            for i, plan in t.plans.items():
                t_dict["plans"][i] = []
                for t_c in plan:
                    t_c_dict = t_c.to_dict()
                    t_dict["plans"][i].append(t_c_dict)
                    que.insert(0, (t_c, t_c_dict))

        out = json.dumps(out, indent=4, separators=(',', ': '))
        f = open(save_path, 'w')
        f.write(out)
        f.close()

    def load(self, load_path, labels):
        def load_child(t_dict):
            assert t_dict["labels"].startswith('[') or t_dict["labels"].endswith(']'), "please check your json file."
            if t_dict["labels"] == "[]":
                return Thought(labels={}, feedback=0, parent=None, name=t_dict["name"])
            label_list = t_dict["labels"][1:-1].split(',')
            label_dict = {}
            for l in label_list:
                l = int(l.strip())
                label_dict[l] = labels[l]

            t = Thought(labels=label_dict, feedback=t_dict["feedback"], parent=None, name=t_dict["name"])
            if "plans" not in t_dict:
                t_dict["plans"] = {}
            for i, ts in t_dict["plans"].items():
                for t_child in ts:
                    child = load_child(t_child)
                    if child == None:
                        continue
                    child.parent = t
                    t.add_child(int(i), child)
            return t

        f = open(load_path, 'r')
        tot_data = json.load(f)
        self.root = load_child(tot_data)
