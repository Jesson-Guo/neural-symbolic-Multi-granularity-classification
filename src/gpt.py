import openai


completion_tokens = prompt_tokens = 0


def chatgpt(client, prompt, model='gpt-3.5-turbo', temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    messages = [{"role": "user", "content": prompt}]
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
        )
        outputs.extend([choice["message"]["content"] for choice in completion["choices"]])

        completion_tokens += completion["usage"]["completion_tokens"]
        prompt_tokens += completion["usage"]["prompt_tokens"]
    return outputs
