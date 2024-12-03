from abc import ABC, abstractmethod
from typing import List
from mcts.node import Context, Node

class Rewarder(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_reward(self, contexts: List[Context], *args, **kwargs) -> float:
        pass

class SciRewarder(Rewarder):
    def __init__(self):
        super().__init__()
    
    def get_reward(self, contexts, *args, **kwargs):
        pass

class TestRewarder(Rewarder):
    def __init__(self):
        super().__init__()
        self.path = [0, -1, -2, 1, -1, -1, 1, 2, 3, 5, -5, -10, 3, 3, 3, 3, -2, -2, 0]
        # Maximum: 24
        # +3, +3, +1, +1, +1, +3, +1, +1, +1, +3
    
    def get_reward(self, contexts, *args, **kwargs) -> float:
        reward = 0.0
        idx = 0
        for context in contexts:
            idx += int(context.content)
            if idx >= len(self.path):
                break
            reward += self.path[idx]
        return reward
        