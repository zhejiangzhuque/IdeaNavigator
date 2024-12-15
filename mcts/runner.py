import random
from typing import (
    Dict, List, Literal,
    Callable
)
from .node import Context, Node
from agents.generator import (
    Generator
)
from agents.feedbacker import (
    Feedbacker
)
from agents.rewarder import (
    Rewarder,
    IdeaArena
)
from utils.log import (
    logger
)
class MCTSRunner:
    def __init__(self,
                 root: Node,
                 generator: Generator,
                 rewarder: Rewarder,
                 sampling_method: Literal["best", "epsilon", "v-epsilon"] = "best",
                 exploration_wright: float = 1.0,
                 *args, **kwargs
                 ):
        self.root = root
        self.generator = generator
        self.rewarder = rewarder
        self.sampling_method = sampling_method
        self.exploration_wright = exploration_wright
        self.best_rollout = None
        self.pre_contexts = []
        self.rollout_history = []
        if self.sampling_method in ["epsilon", "v-epsilon"]:
            self.epsilon = kwargs.get("epsilon", 0.2)
    
    def __expand(self,
                 current_node: Node,
                 contexts: List[Context],
                 n_exp: int
                 ):
        exp_contexts = self.generator.generate(
            contexts=contexts,
            n_choices=n_exp
        )
        msg = "expanded nodes :\n"
        for exp_context in exp_contexts:
            msg += (str(exp_context) + "\n\n")
        logger.debug(msg=msg)
        for i in range(n_exp):
            child_context = exp_contexts[i]
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
        logger.debug(msg="backpropagation starting...")
        while node:
            node.update(reward=reward)
            node = node.parent
        logger.debug(msg="backpropagation was over.")
    
    def __rollout(self,
                  contexts: List[Context],
                  terminal_func: Callable
                  ) -> List[Context]:
        rollout = contexts[:]
        logger.debug(msg="rollout starting...")
        while not terminal_func(rollout):
            gen_context = self.generator.generate(contexts=rollout)[0]
            logger.debug(msg=f"next step : {gen_context}")
            rollout.append(gen_context)
        return rollout
    
    def __run_one_trial(self,
            trial_id: int,
            n_rollouts: int,
            n_exp: int,
            terminal_func: Callable,
            *args, **kwargs
            ):
        cnt_rollouts = 0
        if isinstance(self.rewarder, IdeaArena):
            logger.critical("Initializing the idea DB...")
            self.rewarder.clear_all()
            init_idea_cnt = kwargs.get("init_idea_cnt", 4)
            for i in range(init_idea_cnt):
                idea = self.__rollout(contexts=self.pre_contexts[:], terminal_func=terminal_func)[-1].content
                self.rewarder.add_idea(idea)
            logger.critical(f"{init_idea_cnt} initial ideas were generated.")
        while cnt_rollouts < n_rollouts:
            current_node = self.root
            contexts = self.pre_contexts[:]
            while not current_node.is_leaf():
                if self.sampling_method == "best":
                    current_node = current_node.best_child(exploration_weight=self.exploration_wright) # select
                elif self.sampling_method == "epsilon":
                    current_node = current_node.epsilon_sample(epsilon=self.epsilon, explaration_weight=self.exploration_wright)
                elif self.sampling_method == "v-epsilon":
                    current_node = current_node.epsilon_sample(epsilon=self.epsilon / self.root.visits, explaration_weight=self.exploration_wright)
                if current_node != self.root:
                    contexts.append(current_node.context)
            if terminal_func(contexts):
                if self.sampling_method == "best":
                    return
                cnt_rollouts += 1
                logger.critical(f"trial {trial_id}: rollout {cnt_rollouts} was over, current best value : {self.best_rollout['reward']}")
                continue
            if current_node.visits > 0 or current_node == self.root:
                self.__expand(current_node=current_node, contexts=contexts, n_exp=n_exp) # expand
                current_node = random.choice(current_node.children)
            rollout = self.__rollout(contexts=contexts, terminal_func=terminal_func) # rollout
            reward, judgment = self.rewarder.get_reward(rollout)
            logger.debug(msg=f"reward = {reward}")
            if judgment:
                logger.debug(msg=f"specific judgment:\n{judgment}")
            if isinstance(self.rewarder, IdeaArena):
                self.rewarder.add_idea(rollout[-1].content)
            if self.best_rollout is None or self.best_rollout["reward"] < reward:
                self.best_rollout = {
                    "rollout": rollout,
                    "reward": reward
                }
            self.__backprop(leaf_node=current_node, reward=reward) # back propagation
            cnt_rollouts += 1
            logger.critical(f"trial {trial_id}: rollout {cnt_rollouts} was over, current best value : {self.best_rollout['reward']}")
            
    
    def __next_step(self) -> bool:
        msg = "children nodes :\n"
        for idx, child in enumerate(self.root.children):
            msg += (
                f"child-{idx}: [{child.context.key}]\n"
                f"{child.context.content[:50]}...\n"
                f"UCT = {child.uct()}\n\n"
            )
        logger.debug(msg=msg)
        self.root = self.root.best_child()
        if self.root is None:
            return False
        self.pre_contexts.append(self.root.context)
        self.root.clear()
        return True
    
    def run(self,
            n_trials: int = -1,
            n_rollouts: int = 10,
            n_exp: int = 5,
            terminal_func: Callable = lambda contexts: contexts[-1].key == "terminate"
            ):
        cnt_trials = 0
        while n_trials < 0 or cnt_trials < n_trials:
            self.__run_one_trial(
                trial_id=cnt_trials,
                n_rollouts=n_rollouts,
                n_exp=n_exp,
                terminal_func=terminal_func
            )
            if not self.__next_step():
                # terminal node
                logger.critical(f"all trials were over, best value = {self.best_rollout['reward']}")
                break
            logger.critical(f"trial {cnt_trials} was over, next step:\n{self.root.context}")
            cnt_trials += 1
        