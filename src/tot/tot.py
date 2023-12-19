


class Thought(object):
    def __init__(self, labels, feedback, parent=None) -> None:
        self.labels = labels
        self.feedback = feedback
        self.parent = parent
        self.children = {}

    def infer(self, prompt, gpt, n):
        samples = gpt(prompt=prompt, n=n)
        


class ToT(object):
    def __init__(
        self,
    ) -> None:
        super.__init__()

    def construct_prompt(self, labels):
        pass

    def solve_once(self, thought: Thought):
        labels = thought.labels
        prompt = self.construct_prompt(labels)
