from argparse import ArgumentParser

from mcts.node import (
    Node,
    Context
)

from mcts.runner import (
    MCTSRunner
)
from agents.generator import (
    TestGenerator
)
from agents.rewarder import (
    TestRewarder
)

def mcts_demo():
    generator = TestGenerator()
    rewarder = TestRewarder()
    runner = MCTSRunner(
        root=Node(
            context=Context(
                key='gen',
                content="0"
            )
        ),
        generator=generator,
        rewarder=rewarder,
        sampling_method="epsilon"
    )
    runner.run(
        n_rollouts=100,
        n_exp=5,
        terminal_func=lambda contexts: sum(int(context.content) for context in contexts) >= len(rewarder.path)
    )
    best_rollout = runner.best_rollout
    reward = best_rollout["reward"]
    contexts = best_rollout["rollout"]
    for context in contexts:
        print(context)
    print(f"Final reward = {reward}")


def main():
    parser = ArgumentParser()
    parser.add_argument("-f", "--func", required=True, help="The testing function")

    args = parser.parse_args()
    func = args.func
    eval(func)()

if __name__ == "__main__":
    main()