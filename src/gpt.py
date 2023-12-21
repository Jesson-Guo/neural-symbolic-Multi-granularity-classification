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
