from abc import ABC, abstractmethod
from mcts.node import Context, Node
from typing import (
    List
)

class Generator(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def generate(self, contexts: List[Context], *args, **kwargs) -> Context:
        pass

# TODO
class SciGenerator(Generator):
    def __init__(self):
        super().__init__()

    def generate(self, contexts, *args, **kwargs):
        pass

import random
class TestGenerator(Generator):
    def __init__(self):
        super().__init__()
    
    def generate(self, contexts, *args, **kwargs):
        return Context(
            key='gen',
            content=f"{random.randint(1, 3)}"
        )