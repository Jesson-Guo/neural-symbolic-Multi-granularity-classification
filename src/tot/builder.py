import json
import copy
import random
import traceback
import numpy as np

from src.gpt import GPT
from utils.util import Result
from src.tot.tot import Thought


class ToTBuilder:
    def __init__(self, plan_func, num_plans=2, num_coarse=10000, num_k=2, max_depth=3) -> None:
        self.plan_func = plan_func
        self.num_plans = num_plans
        self.num_coarse = num_coarse
        self.num_k = num_k
        self.max_depth = max_depth

    def build_on_tree(self, labels, tree, node_children):
        root = Thought(labels, name=['Thing'], layer=0)
        thoughts = [(root, tree)]
        while len(thoughts):
            t, node = thoughts.pop()
            for node_child in node.children.values():
                name_t = copy.deepcopy(t.name)
                name_t.append(node_child.name)
                t_child = Thought(node_children[node_child.wnid], 2, t, name_t, layer=t.layer+1)
                t.add_child(0, t_child)
                thoughts.insert(0, (t_child, node_child))
        return root

    def check_plans(self, plans, labels):
        plans_t = copy.deepcopy(plans)
        for i in range(len(plans)):
            label_set = set()
            for l in labels.keys():
                label_set.add(l)
            plan = plans[i]
            names = set()
            for name, ls in plan.items():
                ls_set = []
                names.add(name)
                for j in range(len(ls)):
                    if ls[j] in label_set:
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

    def check(self, root):
        if root == None:
            return 0
        thoughts = [root]
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

    def build_on_gpt(self, labels, gpt: GPT, save_path, load_path=""):
        # 需要设置中间节点的数量上限！！！！
        # 设置叶子节点，当一个thought中少雨k个时直接分类
        try:
            cnt = 0
            plan_dict = {}
            if load_path == "":
                root = Thought(labels, 2, name='Thing', layer=0)
            else:
                root, _ = self.load(labels, load_path)
            thoughts = [root]

            while len(thoughts) and len(plan_dict) < self.num_coarse:
                t = thoughts.pop()
                label_list = list(t.labels.keys())
                label_list.sort()
                label_str = str(label_list)[1:-1]

                if (not t.is_valid()) or t.stop():
                    continue
                if (len(t.labels) <= self.num_k or t.layer == self.max_depth-1) and len(t.plans) == 0:
                    for k, v in t.labels.items():
                        thought = Thought({k: v}, 2, t, v, t.layer+1)
                        t.add_child(0, thought)
                    plan_dict[label_str] = t.plans
                    continue
                if len(t.plans) > 0:
                    plan_dict[label_str] = t.plans
                    for _ in t.plans.values():
                        for child in _:
                            thoughts.insert(0, child)
                    continue

                if label_str not in plan_dict:
                    contents = gpt.generate(t.labels, num_plans=self.num_plans, cat_name=t.name, n=1)
                    cnt += 1

                    plans = gpt.gen_plans(contents)
                    plans = self.check_plans(plans, t.labels)

                    for j in range(len(plans)):
                        i = len(t.plans)
                        for name, ls in plans[j].items():
                            if len(ls) == 1:
                                name = labels[ls[0]]

                            assert len(ls) <= len(t.labels), "gpt generate error."
                            l_dict = {l: labels[l] for l in ls}
                            thought = Thought(l_dict, 2, t, name, t.layer+1)
                            t.add_child(i, thought)
                            thoughts.insert(0, thought)

                    plan_dict[label_str] = t.plans
                else:
                    if t.feedback != 0 and len(t.labels) < len(t.parent.labels):
                        t.plans = plan_dict[label_str]
                if cnt % 10 == 0:
                    self.save(root, save_path)
            self.save(root, save_path)
            print(len(plan_dict))
            return root
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.save(root, save_path)
            exit(0)

    def save(self, root, save_path):
        out = root.to_dict()
        que = [(root, out)]

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

    def load(self, labels, load_path):
        def load_child(t_dict):
            assert t_dict["labels"].startswith('[') or t_dict["labels"].endswith(']'), "please check your json file."
            assert not t_dict["labels"] == "[]", "labels is empty, please check your json file."

            label_list = t_dict["labels"][1:-1].split(',')
            label_list = [int(l.strip()) for l in label_list]
            label_list.sort()
            label_dict = {l: labels[l] for l in label_list}

            label_str = str(label_list)[1:-1]
            # t = Thought(labels=label_dict, feedback=t_dict["feedback"], parent=None, name=t_dict["name"])
            t = Thought(labels=label_dict, feedback=t_dict["feedback"], parent=None, name=t_dict["name"], layer=t_dict["layer"])
            if len(label_dict) == 1:
                return t
            if "plans" not in t_dict:
                t_dict["plans"] = {}
            if label_str in plan_dict:
                t.plans = plan_dict[label_str]
                return t
            for i, ts in t_dict["plans"].items():
                for j in range(len(ts)):
                    child = load_child(ts[j])
                    child.parent = t
                    t.add_child(int(i), child)
            plan_dict[label_str] = t.plans
            return t

        plan_dict = {}
        tid = {"tid": -1}
        f = open(load_path, 'r')
        tot_data = json.load(f)
        root = load_child(tot_data)
        return root, plan_dict
