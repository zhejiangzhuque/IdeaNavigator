from abc import ABC, abstractmethod
from agents.general import (
    LLMEngine
)

class RAG(ABC):
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def run(self, query: str) -> str:
        pass

from openai import Client

class TestRAG(RAG):
    def __init__(self,
                 engine: LLMEngine,
                 *args,
                 **kwargs
                 ):
        sys_prompt = (
            "You are a search engine. When I provide you with a query, you should respond with relevant research papers.\n"
            "The response format should be as follows:\n"
            "[1] Title\nContent\nReference: ... (in MLA format)\n"
            "[2] Title\nContent\nReference: ... (in MLA format)\n"
            "[3] Title\nContent\nReference: ... (in MLA format)\n"
            "...\n"
        )
        self.engine = engine
        self.sys_prompt = sys_prompt

    def run(self, query: str) -> str:
        response = self.engine.gen_from_prompt(sys_prompt=self.sys_prompt, prompt=query)[0]
        return response