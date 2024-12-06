import os
from dotenv import find_dotenv, load_dotenv
from mcts.node import (
    Node,
    root_node
)
from mcts.runner import (
    MCTSRunner
)
from agents.generator import (
    SciGenerator
)
from agents.rewarder import (
    IdeaArena
)
from utils.log import (
    logger
)

load_dotenv(find_dotenv())

api_key = os.environ["API_KEY"]
base_url = os.environ["BASE_URL"]

def main(opt):
    topic = opt.topic
    model = opt.model
    sampling_method = opt.sampling_method
    n_rollouts = opt.n_rollouts
    n_exp = opt.n_exp
    
    generator = SciGenerator(
        api_key=api_key,
        base_url=base_url,
        task=topic,
        model=model
    )
    rewarder = IdeaArena(
        base_url=base_url,
        api_key=api_key,
        topic=topic
    )
    runner = MCTSRunner(
        root=Node(context=root_node()),
        generator=generator,
        rewarder=rewarder,
        sampling_method=sampling_method,
        exploration_wright=1.0
    )
    runner.run(
        n_rollouts=n_rollouts,
        n_exp=n_exp,
        terminal_func=lambda contexts: len(contexts) > 0 and contexts[-1].key == "gen_idea"
    )

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--topic", type=str, help="The topic of research idea")
    parser.add_argument("--n_rollouts", type=int, help="number of rollouts per trial")
    parser.add_argument("--n_exp", type=int, help="number of expanded nodes")
    parser.add_argument("--model", type=str, help="The name of LLM to power the system")
    parser.add_argument("--sampling_method", type=str, help="The sampling method of MCTS")
    
    opt = parser.parse_args()
    main(opt)