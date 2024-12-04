from pathlib import Path
import yaml
from typing import (
    List
)
from abc import ABC, abstractmethod
from mcts.node import (
    Context
)
from .general import LLMAgent
from .feedbacker import (
    SimpleFeedbacker
)

class Generator(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def generate(self, contexts: List[Context], *args, **kwargs) -> Context:
        pass

class SciGenerator(Generator):
    def __init__(self,
                 api_key: str,
                 base_url: str,
                 task: str,
                 model: str = "gpt-4o",
                 ):
        super().__init__()
        config_path = Path("agents") / "prompts" / "sci-generator.yml"
        with open(config_path) as f:
            prompts = yaml.safe_load(f)
            sys_prompt = prompts["sys_prompt"]
        self.agent = LLMAgent(
            api_key=api_key,
            base_url=base_url,
            model=model,
            sys_prompt=f"{sys_prompt}\n\nYour research topic is: {task}"
        )
        self.feedbacker = SimpleFeedbacker(
            base_url=base_url,
            api_key=api_key
        )

    def generate(self, contexts, *args, **kwargs) -> Context:
        context = self.agent.generate(
            contexts=contexts,
            *args,
            **kwargs
        )
        if context.key == "search":
            feedback = self.feedbacker.feedback(
                contexts=[context]
            )
            context.observation = feedback
        return context

class TestGenerator(Generator):
    def __init__(self):
        super().__init__()
        self.seed = 0
    
    def generate(self, contexts, *args, **kwargs) -> Context:
        context = Context(
            key='reasoning',
            content=f"{self.seed % 3 + 1}"
        )
        self.seed += 1
        return context