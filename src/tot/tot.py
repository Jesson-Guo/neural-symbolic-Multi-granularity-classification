import numpy as np
import copy
from src.gpt import GPT


class Thought(object):
    def __init__(self, labels, feedback=2, parent=None, name=[]) -> None:
        self.labels = labels
        self.feedback = feedback
        self.parent = parent
        self.children = {}
        self.name = name
        self.sub_labels = None

    def is_valid(self):
        return self.feedback > 0

    def stop(self):
        return len(self.labels) == 1

    def add_child(self, t):
        i = len(self.children)
        self.children[i] = t

    def update_parent(self, t):
        self.parent = t

    def gen_plans(self, contents, node_dict, label_to_wnid):
        plans, plans_w = [], []
        for content in contents:
            content = content.replace('\n', ' ').strip()
            content = content.split('Plan')[1:]
            for s in content:
                categories = s.split('Category')[1:]
                c, c_w = {}, {}
                for item in categories:
                    item = item.split('-')
                    name = item[0].split(':')[-1].strip()
                    c[name] = []
                    c_w[name] = []
                    for l in item[1:]:
                        l = l.strip()
                        c[name].append(l)
                        c_w[name].append(node_dict[label_to_wnid[l]].weight)
                plans.append(c)
                plans_w.append(c_w)
        return plans, plans_w

    def estimate_plans(self, plans_w, func):
        result = []
        for plan in plans_w:
            out = func(plan)
            # out是2/1/0表示sure/likely/impossible，后续也可以比较启发式和最优的差别
            # 这里也可以考虑将index传给gpt让gpt评价（可尝试）
            result.append(out)
        # 选择最好的（最大的），如果有些指标是越小越好需要先处理
        # idx = np.array(result).argmax()
        return result

    def estimate_clusters(self, v, plan, plan_w, func):
        similarity = func(v, plan_w)
        idx = np.array(similarity).argmin()
        for name, _ in plan_w:
            if idx == 0:
                break
            idx -= 1
        name_t = copy.deepcopy(self.name)
        name_t.append(name)
        return name_t, plan[name]


class ToT(object):
    prompt_template = '''Giving {plans} plans to divide the INPUT into {categories} categories with title in one word or one phrase.
    INPUT: {input}
    '''

    def __init__(self, plan_func, sim_func) -> None:
        self.plan_func = plan_func
        self.sim_func = sim_func

    def construct_prompt(self, labels, p, c):
        s = "" + f"'{labels[0]}'"
        for i in range(1, len(labels)):
            s += f", '{labels[i]}'"
        prompt = self.prompt_template.format(plans=p, categories=c, input=s)
        return prompt

    def solve_once(self, v, node_dict, label_to_wnid, thought: Thought, gpt: GPT):
        prompt = self.construct_prompt(thought.labels)
        contents = gpt.generate(prompt, n=1)

        plans, plans_w = thought.gen_plans(contents, node_dict, label_to_wnid)
        result = thought.estimate_plans(plans, plans_w, self.plan_func)
        for i in range(len(result)):
            name, labels = thought.estimate_clusters(v, plans[i], plans_w[i], self.sim_func)
            t = Thought(labels, result[i], thought, name)
            thought.add_child(t)

    def dfs(self, v, node_dict, label_to_wnid, thought: Thought, gpt: GPT):
        pass

    def bfs(self, v, node_dict, label_to_wnid, thought: Thought, gpt: GPT):
        thoughts = [thought]
        name = None
        while len(thoughts):
            t = thoughts.pop()
            if not t.is_valid():
                continue
            if t.stop():
                name = t.name
                name.append(t.labels[0])
                break
            self.solve_once(v, node_dict, label_to_wnid, t, gpt)
            for _, v in t.items():
                thoughts.append(v)
        return name

    def solve(self, v, labels, node_dict, label_to_wnid, gpt: GPT, method='bfs'):
        thought = Thought(labels, name=['Thing'])
        if method == 'bfs':
            output = self.bfs(v, node_dict, label_to_wnid, thought, gpt)
        elif method == 'dfs':
            output = self.bfs(v, node_dict, label_to_wnid, thought, gpt)
        return output[-1], output
