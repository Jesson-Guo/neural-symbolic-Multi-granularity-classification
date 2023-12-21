from openai import OpenAI


class GPT(object):
    completion_tokens = 0
    prompt_tokens = 0

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

    def generate(self, prompt, n=1, stop=None):
        messages = [{"role": "user", "content": prompt}]
        outputs = []
        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=cnt,
                stop=stop,
            )
            outputs.extend([choice.message.content for choice in completion.choices])

        self.completion_tokens += completion.usage.completion_tokens
        self.prompt_tokens += completion.usage.prompt_tokens

        return outputs


class FakeGPT(object):
    fake_label = '''\n- {label}'''
    fake_cat = '''Category {num}: {name}{cat}'''
    fake_plan = '''Plan {num}:{plan}'''

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

    def generate(self, labels, n=1, stop=None):
        l1 = self.fake_label.format(label=labels[0])
        l2 = ""
        for i in range(1, len(labels)):
            l2 += self.fake_label.format(label=labels[i])

        cat1 = self.fake_cat.format(num=1, name='A', cat=l1)
        cat2 = self.fake_cat.format(num=2, name='B', cat=l2)

        plan1 = self.fake_plan.format(num=1, plan=cat1)
        plan2 = self.fake_plan.format(num=2, plan=cat2)

        return [plan1+"\n\n"+plan2]
