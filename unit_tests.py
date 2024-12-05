import os
from dotenv import find_dotenv, load_dotenv
from argparse import ArgumentParser
from mcts.node import (
    Node,
    Context,
    root_node
)
from mcts.runner import (
    MCTSRunner
)
from agents.general import (
    LLMEngine
)
from agents.generator import (
    TestGenerator,
    SciGenerator
)
from agents.feedbacker import (
    SimpleFeedbacker
)
from agents.rewarder import (
    SciRewarder,
    TestRewarder
)
from rag.general import (
    TestRAG
)
from utils.log import (
    logger
)

load_dotenv(find_dotenv())

api_key = os.environ["API_KEY"]
base_url = os.environ["BASE_URL"]

def mcts_demo():
    generator = TestGenerator()
    rewarder = TestRewarder()
    runner = MCTSRunner(
        root=Node(
            context=root_node()
        ),
        generator=generator,
        rewarder=rewarder,
        sampling_method="epsilon",
        epsilon=0.1,
        exploration_wright=2.0
    )
    runner.run(
        n_rollouts=50,
        n_exp=3,
        terminal_func=lambda contexts: sum(int(context.content) for context in contexts) >= len(rewarder.path)
    )

def sci_generator():
    task = "Generated some scientific research idea related to language model compression"
    generator = SciGenerator(
        api_key=api_key,
        base_url=base_url,
        task=task,
        model='gpt-4o'
    )
    contexts = []
    while True:
        gen_context = generator.generate(
            contexts=contexts
        )[0]
        print(gen_context)
        contexts.append(gen_context)
        if gen_context.key == "gen_idea":
            break
    rewarder = SciRewarder(
        base_url=base_url,
        api_key=api_key,
        model="gpt-4o",
        topic=task
    )
    reward, judgments = rewarder.get_reward(contexts=contexts)
    print(judgments, reward, sep='\n')

def mcts_idea_gen():
    topic = "Generated some scientific research idea related to language model compression"
    generator = SciGenerator(
        api_key=api_key,
        base_url=base_url,
        task=topic,
        model='gpt-4o'
    )
    rewarder = SciRewarder(
        base_url=base_url,
        api_key=api_key,
        model="gpt-4o",
        topic=topic
    )
    runner = MCTSRunner(
        root=Node(context=root_node()),
        generator=generator,
        rewarder=rewarder,
        sampling_method='best',
        exploration_wright=1.0
    )
    runner.run(
        n_rollouts=10,
        n_exp=3,
        terminal_func=lambda contexts: len(contexts) > 0 and contexts[-1].key == "gen_idea"
    )

def multi_gen():
    engine = LLMEngine(
        api_key=api_key,
        base_url=base_url,
        model='gpt-4o',
        sys_prompt="You are an AI assistant."
    )
    results = engine.gen_from_prompt(prompt="Tell me a short joke.", n_choices=3)
    for idx, result in enumerate(results):
        print(f"choice {idx}: {result}")

def test_logger():
    logger.debug(msg="This is a debugging message.")
    logger.info(msg="This is an information.")
    logger.critical(msg="This is a critical message.")

def main():
    parser = ArgumentParser()
    parser.add_argument("-f", "--func", required=True, help="The testing function")

    args = parser.parse_args()
    func = args.func
    eval(func)()

if __name__ == "__main__":
    main()