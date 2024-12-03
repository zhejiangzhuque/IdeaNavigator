from typing import (
    List
)
from abc import ABC, abstractmethod
from .general import (
    LLMAgent
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
                 ):
        pass

class SimpleFeedbacker(Feedbacker):
    def __init__(self,
                 base_url: str,
                 api_key: str
                 ):
        super().__init__()
        self.rag = TestRAG(
            base_url=base_url,
            api_key=api_key
        )
    
    def feedback(self, contexts, *args, **kwargs) -> str:
        assert len(contexts) > 0 and contexts[-1].key == "search"
        response = self.rag.run(query=contexts[-1].content)
        
        return response