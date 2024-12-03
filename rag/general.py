from abc import ABC, abstractmethod

class RAG(ABC):
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def run(self, query: str) -> str:
        pass

from openai import Client

class TestRAG(RAG):
    def __init__(self,
                 base_url: str,
                 api_key: str,
                 *args,
                 **kwargs
                 ):
        self.client = Client(
            base_url=base_url,
            api_key=api_key
        )
        self.sys_prompt = (
            "You are a search engine. When I provide you with a query, you should respond with relevant research papers.\n"
            "The response format should be as follows:\n"
            "[1] Title\nContent\nReference: ... (in MLA format)\n"
            "[2] Title\nContent\nReference: ... (in MLA format)\n"
            "[3] Title\nContent\nReference: ... (in MLA format)\n"
            "...\n"
        )

    def run(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {
                    "role": "system",
                    "content": self.sys_prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        ).choices[0].message.content
        return response