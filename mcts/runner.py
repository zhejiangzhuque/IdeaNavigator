from typing import (
    Dict, List, Literal,
    Callable
)
from .node import Context, Node
from agents.generator import (
    Generator
)
from agents.rewarder import (
    Rewarder
)

class MCTSRunner:
    def __init__(self,
                 root: Node,
                 generator: Generator,
                 rewarder: Rewarder,
                 sampling_method: Literal["best", "epsilon"] = "best"
                 ):
        self.root = root
        self.generator = generator
        self.rewarder = rewarder
        self.sampling_method = sampling_method
        self.best_rollout = None
    
    def __expand(self,
                 current_node: Node,
                 contexts: List[Context],
                 n_exp: int
                 ) -> Context:
        for i in range(n_exp):
            child_context = self.generator.generate(
                contexts=contexts
            )
            child_node = Node(
                context=child_context,
                parent=current_node,
                depth=current_node.depth + 1
            )
            current_node.children.append(child_node)
    
    def __backprop(self,
                   leaf_node: Node,
                   reward: float
                   ):
        node = leaf_node
        while node:
            node.update(reward=reward)
            node = node.parent
    
    def __rollout(self,
                  contexts: List[Context],
                  terminal_func: Callable
                  ) -> List[Context]:
        rollout = contexts[:]
        while not terminal_func(rollout):
            gen_context = self.generator.generate(contexts=rollout)
            rollout.append(gen_context)
        return rollout
    
    def run(self,
            n_rollouts: int,
            n_exp: int,
            terminal_func: Callable = lambda contexts: contexts[-1].key == "terminate"
            ):
        cnt_rollouts = 0
        while cnt_rollouts < n_rollouts:
            current_node = self.root
            contexts = []
            while not current_node.is_leaf():
                if self.sampling_method == "best":
                    current_node = current_node.best_child() # select
                elif self.sampling_method == "epsilon":
                    current_node = current_node.epsilon_sample(epsilon=0.2 / self.root.visits)
                contexts.append(current_node.context)
            if current_node.visits == 0 or current_node == self.root:
                self.__expand(current_node=current_node, contexts=contexts, n_exp=n_exp) # expand
                current_node = current_node.children[0]
            rollout = self.__rollout(contexts=contexts, terminal_func=terminal_func) # rollout
            reward = self.rewarder.get_reward(rollout)
            if self.best_rollout is None or self.best_rollout["reward"] < reward:
                self.best_rollout = {
                    "rollout": rollout,
                    "reward": reward
                }
            self.__backprop(leaf_node=current_node, reward=reward) # back propagation
            cnt_rollouts += 1
            # print(f"Rollout {cnt_rollouts} was over")