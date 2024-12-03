import re
from typing import List, Dict
from openai import Client
from mcts.node import (
    Context
)

class LLMAgent:
    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model: str,
                 sys_prompt: str,
                 ):
        self.client = Client(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.sys_prompt = sys_prompt
    
    def generate(self,
                 contexts: List[Context],
                 *args, **kwargs
                 ) -> Context | None:
        messages = [
            {
                "role": "system",
                "content": self.sys_prompt
            }
        ]
        for context in contexts:
            messages.append(
                {
                    "role": "assistant",
                    "content": context.value
                }
            )
            if context.observation is None:
                continue
            messages.append(
                {
                    "role": "user",
                    "content": context.observation
                }
            )
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            *args,
            **kwargs
        ).choices[0].message.content
        match_key = re.match(pattern=r"\[(.*?)\]", string=response)
        match_content = re.match(pattern=r"\[.*?\](.*)", string=response, flags=re.DOTALL)
        if not match_key or not match_content:
            return None
        key, content = match_key.group(1), match_content.group(1).strip()
        context = Context(
            key=key,
            content=content
        )
        return context