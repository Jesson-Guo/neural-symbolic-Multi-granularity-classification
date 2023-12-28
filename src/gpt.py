import tiktoken
import json
from openai import OpenAI


class GPT(object):
    completion_tokens = 0
    prompt_tokens = 0
    prompt_sample = '''Input: {0: "airplane", 1: "bird", 2: "cat", 3: "dog"}
    Answer: {"Plan1": {"FlyingObjects": [0, 1], "Pets": [2, 3]}, "Plan2": {"NonHuman": [0, 1, 2], "Pets": [3]}}'''
    prompt_template = '''Giving {num_plans} plans to classify all values from Input into different sets with the class name and only word id. Each value can only belong to one set.
    {sample}
    Input: {input}'''

    def __init__(
        self,
        client: OpenAI,
        model: str = 'gpt-3.5-turbo',
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def construct_prompt(self, labels, num_plans=2, num_categories=2):
        labels_copy = {}
        for k, v in labels.items():
            labels_copy[k] = v.split('.')[0]
        prompt = self.prompt_template.format(num_plans=num_plans, sample=self.prompt_sample, input=str(labels_copy))
        return prompt

    def generate(self, labels, num_plans=2, num_categories=2, n=1, stop=None):
        prompt = self.construct_prompt(labels, num_plans, num_categories)
        messages = [
            {"role": "system", "content": 'You are a helpful assistant and have knowledge of Python. Your response should be in JSON format.'},
            {"role": "user", "content": prompt}
        ]
        outputs = []
        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                n=cnt,
                stop=stop,
            )
            outputs.extend([choice.message.content for choice in completion.choices])

        self.completion_tokens += completion.usage.completion_tokens
        self.prompt_tokens += completion.usage.prompt_tokens

        return outputs

    def gen_plans(self, contents, node_dict, label_to_wnid, labels):
        contents = contents[0]
        contents = contents.replace('\n', '')
        contents = json.loads(contents)

        plans, plans_w = [], []
        for content in contents.values():
            plans.append(content)
            plan_w = {}
            for name, label_ids in content.items():
                plan_w[name] = []
                for label_id in label_ids:
                    plan_w[name].append(node_dict[label_to_wnid[labels[label_id]]].weight)
            plans_w.append(plan_w)

        return plans, plans_w


class FakeGPT(GPT):
    fake_cat = '''"{name}": {labels}'''
    fake_plan = '''"Plan{num}":{plan}'''

    def __init__(
        self,
        client: OpenAI,
        model: str = 'gpt-3.5-turbo',
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, labels, num_plans=2, num_categories=2, n=1, stop=None):
        prompt = self.construct_prompt(labels, num_plans, num_categories)
        mid = len(labels) // 2
        labels_a, labels_b = [], []
        i = 0
        for k in labels.keys():
            if i < mid:
                labels_a.append(k)
            else:
                labels_b.append(k)
            i += 1

        cat1 = self.fake_cat.format(name='A', labels=str(labels_a))
        cat2 = self.fake_cat.format(name='B', labels=str(labels_b))

        plan1 = self.fake_plan.format(num=1, plan="{"+cat1+', '+cat2+"\n}")
        plan2 = self.fake_plan.format(num=2, plan="{"+cat1+', '+cat2+"\n}")

        return ["{"+plan1+", "+plan2+"}"]
