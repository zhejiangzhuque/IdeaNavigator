import re
import yaml
import json
from pathlib import Path
from abc import ABC, abstractmethod
from openai import Client
from typing import (
    List,
    Dict,
    Tuple
)
from mcts.node import (
    Context
)
from agents.general import (
    LLMAgent,
    PromptTemplate
)

class Rewarder(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_reward(self, contexts: List[Context], *args, **kwargs) -> Tuple[float, Dict | None]:
        pass

class SciRewarder(Rewarder):
    def __init__(self,
                 base_url: str,
                 api_key: str,
                 model: str,
                 topic: str
                 ):
        config_path = Path("agents") / "prompts" / "sci-rewarder.yml"
        with open(config_path) as f:
            prompts = yaml.safe_load(f)
            self.sys_prompt = PromptTemplate(
                template=prompts["sys_prompt"],
                parameters={
                    "topic": topic,
                    "idea": ""
                }
            )
        self.client = Client(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
    
    def get_reward(self,
                   contexts: List[Context],
                   *args, **kwargs) -> Tuple[float, Dict | None]:
        idea = contexts[-1].content
        self.sys_prompt.parameters["idea"] = idea
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.sys_prompt.value
                }
            ]
        ).choices[0].message.content
        jud_match = re.match(pattern=r"```json(.*?)```", string=response, flags=re.DOTALL)
        if not jud_match:
            # TODO
            # current: redo
            # print("format error")
            return self.get_reward(
                contexts=contexts
                *args, **kwargs
            )
        judgments = json.loads(jud_match.group(1))
        score_sum = 0
        for key in [
            "novelty",
            "feasibility",
            "clarity",
            "impact",
            "relevance"
        ]:
            # TODO handle the key error
            # current: redo
            # print("format error")
            if key not in judgments:
                return self.get_reward(
                    contexts=contexts,
                    *args, **kwargs
                )
            score_sum += int(judgments[key]["score"])
        score_avg = score_sum / 5
        return score_avg, judgments
            

class TestRewarder(Rewarder):
    def __init__(self):
        super().__init__()
        self.path = [0, -1, -2, 1, -1, -1, 1, 2, 3, 5, -5, -10, 3, 3, 3, 3, -2, -2, 0]
        # Maximum: 24
        # +3, +3, +1, +1, +1, +3, +1, +1, +1, +3
    
    def get_reward(self, contexts, *args, **kwargs) -> Tuple[float, Dict | None]:
        reward = 0.0
        idx = 0
        for context in contexts:
            idx += int(context.content)
            if idx >= len(self.path):
                break
            reward += self.path[idx]
        return reward, None
        