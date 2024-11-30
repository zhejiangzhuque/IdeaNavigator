from argparse import ArgumentParser

from mcts.node import (
    Node,
    Context,
    root_node
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
            context=root_node()
        ),
        generator=generator,
        rewarder=rewarder,
        sampling_method="v-epsilon",
        epsilon = 0.4
    )
    runner.run(
        n_rollouts=20,
        n_exp=3,
        terminal_func=lambda contexts: sum(int(context.content) for context in contexts) >= len(rewarder.path)
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("-f", "--func", required=True, help="The testing function")

    args = parser.parse_args()
    func = args.func
    eval(func)()

if __name__ == "__main__":
    main()