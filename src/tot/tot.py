import json
import copy
from src.gpt import GPT
from utils.util import Result
import random
import traceback


STATUS = {
    2: "sure",
    1: "likely",
    0: "impossible"
}


class Thought:
    def __init__(self, labels, feedback=2, parent=None, name=[]) -> None:
        self.labels = labels
        self.feedback = feedback
        self.parent = parent
        self.plans = {}
        self.name = name

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
    def __init__(self, plan_func, sim_func) -> None:
        self.plan_func = plan_func
        self.sim_func = sim_func
        self.root = None

    def estimate_plans(self, plans_w, func):
        result = []
        for plan_w in plans_w:
            out = func(plan_w)
            # out是2/1/0表示sure/likely/impossible，后续也可以比较启发式和最优的差别
            # 这里也可以考虑将index传给gpt让gpt评价（可尝试）
            result.append(out)
        # 选择最好的（最大的），如果有些指标是越小越好需要先处理
        # idx = np.array(result).argmax()
        return result

    def estimate_clusters(self, v, plan, plan_w, ts, func):
        similarity = func(v, plan_w)
        idx = similarity.argmin().item()
        t = ts[idx]
        for name, _ in plan_w.items():
            if idx == 0:
                break
            idx -= 1
        name_t = copy.deepcopy(self.name)
        name_t.append(name)
        return name_t, t

    def choose_plan(self, thought: Thought, node_dict, label_to_wnid):
        candidate = []
        for ts in thought.plans.values():
            # 启发式的选取，likely即可，随机选取TODO
            if ts[0].is_valid():
                candidate.append(ts)
        ts = random.choice(candidate)

        plans = []
        plan, plan_w = {}, {}
        for t in plans:
            plan[t.name] = []
            plan_w[t.name] = []
            for item in t.labels.values():
                plan[t.name].append(item)
                plan_w[t.name].append(node_dict[label_to_wnid[item]].weight)
        plans.append((plan, plan_w, ts))
        random.shuffle(plans)
        return plans

    def solve_once(self, v, plan, plan_w, ts):
        name, t = self.estimate_clusters(v, plan, plan_w, ts, self.sim_func)
        r = Result(name, STATUS[t.feedback])
        return t, r

    def dfs(self, v, node_dict, label_to_wnid, thought: Thought):
        def helper(thought: Thought, res: Result, ok: bool):
            if not thought.is_valid():
                return False, None
            if thought.stop():
                r = Result(thought.labels[0], STATUS[thought.feedback], parent=result)
                res.add(r)
                return True, thought.labels[0]

            plans = self.choose_plan(thought, node_dict, label_to_wnid)
            for plan, plan_w, ts in plans:
                t, r = self.solve_once(v, plan, plan_w, ts)
                res.add(r)
                ok, label = helper(t, r, ok)
                if ok:
                    break
            return ok, label

        result = Result(thought.name, STATUS[thought.feedback])
        _, label = helper(thought, result)
        return result, label

    def bfs(self, v, node_dict, label_to_wnid, thought: Thought):
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

    def solve(self, v, node_dict, label_to_wnid, method='dfs'):
        method = getattr(self, method)
        output = method(v, node_dict, label_to_wnid, self.root)
        return output

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

    def check_thought_labels(self, thought: Thought):
        label_set = []
        for ts in thought.plans.values():
            s = set()
            for t in ts:
                for l in t.labels.keys():
                    s.add(l)
            label_set.append(s)

        left_labels = []
        for s in label_set:
            r = []
            for i in thought.labels.keys():
                if not i in s:
                    r.append(i)
            left_labels.append(r)

        for r in left_labels:
            if len(r):
                return False, left_labels
        return True, left_labels

    def check(self):
        thoughts = [self.root]
        while len(thoughts):
            thought = thoughts.pop()
            ok, left = self.check_thought_labels(thought)
            assert ok, "labels dismatch between parent thought and children, please checkout the thought json file."
            for ts in thought.plans.values():
                for t in ts:
                    thoughts.append(t)

    def build_tot(self, labels, node_dict, label_to_wnid, node_children, tree, gpt: GPT, save_path, is_load=True):
        try:
            cnt = 0
            plan_dict = {}
            thoughts = []
            if not is_load:
                root = self.build_on_tree(labels, tree, node_children)
                # 合并
                self.root.plans[1] = root
            else:
                self.check()
                for item in self.root.plans[0]:
                    thoughts.insert(0, item)

            while len(thoughts):
                t = thoughts.pop()
                if (not t.is_valid()) or t.stop():
                    continue
                if len(t.labels) == 2:
                    for l in t.labels:
                        name_t = copy.deepcopy(t.name)
                        name_t.append(labels[l])
                        thought = Thought({l: labels[l]}, 2, t, name_t)
                        t.add_child(0, thought)
                        thoughts.insert(0, thought)
                    continue
                if len(t.plans) > 0 and len(t.name) > 1:
                    label_list = list(t.labels.keys())
                    label_list.sort()
                    plan_dict[str(label_list)[1:-1]] = t.plans

                    plans_w = []
                    for content in t.plans.values():
                        plan_w = {}
                        for items in content:
                            plan_w[items.name[-1]] = []
                            for label_id in list(items.labels):
                                if label_id not in labels:
                                    continue
                                plan_w[items.name[-1]].append(node_dict[label_to_wnid[labels[label_id]]].weight)
                        plans_w.append(plan_w)

                    estimate = self.estimate_plans(plans_w, self.plan_func)
                    for i in range(len(estimate)):
                        for child in t.plans[i]:
                            child.feedback = estimate[i]
                            thoughts.insert(0, child)
                    continue

                label_list = list(t.labels.keys())
                label_list.sort()
                label_str = str(label_list)[1:-1]
                if label_str not in plan_dict:
                    contents = gpt.generate(t.labels, num_plans=2, num_categories=2, n=1)
                    plans, plans_w = gpt.gen_plans(contents, node_dict, label_to_wnid, t.labels)
                    estimate = self.estimate_plans(plans_w, self.plan_func)
                    cnt += 1

                    if sum(estimate) == 0:
                        for k, v in t.labels.items():
                            name_t = copy.deepcopy(t.name)
                            name_t.append(v)
                            t.add_child(0, Thought({k: v}, 2, t, name_t))
                    else:
                        for i in range(len(estimate)):
                            for name, ls in plans[i].items():
                                name_t = copy.deepcopy(t.name)
                                name_t.append(name)

                                l_dict = {}
                                for l in ls:
                                    l_dict[l] = labels[l]
                                thought = Thought(l_dict, estimate[i], t, name_t)
                                t.add_child(i, thought)
                                thoughts.insert(0, thought)

                    plan_dict[label_str] = t.plans
                    ok, left = self.check_thought_labels(t)
                    if not ok:
                        for i in range(len(left)):
                            if len(left[i]):
                                l_dict = {}
                                for l in left[i]:
                                    l_dict[l] = labels[l]
                                name_t = copy.deepcopy(t.name)
                                name_t.append("Other")
                                other = Thought(l_dict, 1, t, name_t)
                                t.add_child(i, other)
                                thoughts.insert(0, other)
                else:
                    t.plans = plan_dict[label_str]
                if cnt % 10 == 0:
                    self.save(save_path)
            self.save(save_path)
        except Exception as e:
            self.save(save_path)
            print(e)
            print(traceback.format_exc())
            a = 0

    def save(self, save_path):
        def save_child(t):
            if len(t.plans) == 0:
                return t.to_dict()

            save_dict = t.to_dict()
            for i, plan in t.plans.items():
                save_dict["plans"][i] = []
                for t_child in plan:
                    save_dict["plans"][i].append(save_child(t_child))
            return save_dict

        out = save_child(self.root)
        out = json.dumps(out, indent=4, separators=(',', ': '))
        f = open(save_path, 'w')
        f.write(out)
        f.close()

    def load(self, load_path, labels):
        def load_child(t_dict):
            assert t_dict["labels"].startswith('[') or t_dict["labels"].endswith(']'), "please check your json file."
            label_list = t_dict["labels"][1:-1].split(',')
            label_dict = {}
            for l in label_list:
                l = int(l.strip())
                label_dict[l] = labels[l]

            t = Thought(labels=label_dict, feedback=t_dict["feedback"], parent=None, name=t_dict["name"])
            for i, ts in t_dict["plans"].items():
                for t_child in ts:
                    child = load_child(t_child)
                    child.parent = t
                    t.add_child(int(i), child)
            return t

        f = open(load_path, 'r')
        tot_data = json.load(f)
        self.root = load_child(tot_data)
