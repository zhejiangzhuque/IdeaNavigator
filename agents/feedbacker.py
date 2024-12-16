from typing import (
    List
)
from abc import ABC, abstractmethod
from .general import (
    LLMEngine
)
from mcts.node import (
    Context
)
from rag.general import(
    TestRAG
)

class Feedbacker(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def feedback(self,
                 contexts: List[Context],
                 *args, **kwargs
                 ) -> str:
        pass

class SimpleFeedbacker(Feedbacker):
    def __init__(self,
                 engine: LLMEngine
                 ):
        super().__init__()
        self.rag = TestRAG(engine)
    
    def feedback(self, contexts, *args, **kwargs) -> str:
        assert len(contexts) > 0 and contexts[-1].key == "search"
        response = self.rag.run(query=contexts[-1].content)
        
        return response