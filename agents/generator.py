from pathlib import Path
import yaml
from typing import (
    List
)
from abc import ABC, abstractmethod
from mcts.node import (
    Context
)
from .general import LLMEngine
from .feedbacker import (
    SimpleFeedbacker
)

class Generator(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def generate(self, contexts: List[Context], n_choices: int = 1, *args, **kwargs) -> List[Context]:
        pass

class SciGenerator(Generator):
    def __init__(self,
                 engine: LLMEngine,
                 topic: str
                 ):
        super().__init__()
        config_path = Path("agents") / "prompts" / "sci-generator.yml"
        with open(config_path) as f:
            prompts = yaml.safe_load(f)
            sys_prompt = prompts["sys_prompt"]
            sys_prompt=f"{sys_prompt}\n\nYour research topic is: {topic}"
        self.engine = engine
        self.sys_prompt = sys_prompt
        self.feedbacker = SimpleFeedbacker(engine=engine)

    def generate(self, contexts, n_choices: int = 1, *args, **kwargs) -> List[Context]:
        ctx_choices = self.engine.gen_from_contexts(
            contexts=contexts,
            sys_prompt=self.sys_prompt,
            n_choices=n_choices,
            *args,
            **kwargs
        )
        for ctx in ctx_choices:
            if ctx.key == "search":
                observation = self.feedbacker.feedback(
                    contexts=contexts + [ctx]
                )
                ctx.observation = observation
        return ctx_choices

class TestGenerator(Generator):
    def __init__(self):
        super().__init__()
        self.seed = 0
    
    def generate(self, contexts, n_choices: int = 1, *args, **kwargs) -> List[Context]:
        ctx_choices = []
        for i in range(n_choices):
            context = Context(
                key='reasoning',
                content=f"{self.seed % 3 + 1}"
            )
            ctx_choices.append(context)
            self.seed += 1
        return ctx_choices