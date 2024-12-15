import os
from dotenv import find_dotenv, load_dotenv
from argparse import ArgumentParser
from mcts.node import (
    Node,
    Context
)
from mcts.runner import (
    MCTSRunner
)
from agents.general import (
    PromatParser,
    LLMEngine
)
from agents.generator import (
    TestGenerator,
    SciGenerator
)
from agents.rewarder import (
    SciRewarder,
    TestRewarder,
    IdeaArena
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
        root=Node.root_node(),
        generator=generator,
        rewarder=rewarder,
        sampling_method="best",
        exploration_wright=2.0
    )
    runner.run(
        n_rollouts=20,
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
    rewarder = IdeaArena(
        base_url=base_url,
        api_key=api_key,
        topic=topic
    )
    runner = MCTSRunner(
        root=Node.root_node(),
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

old_idea = """
# Automatic Optimization of Structured Pruning for Language Model Compression using Neural Architecture Search

## Abstract
This research proposes a novel approach to language model compression by integrating structured pruning with neural architecture search (NAS). The study aims to enhance the efficiency and performance of structured pruning techniques by automatically identifying the optimal pruning configuration for language models. This hybrid approach seeks to preserve model performance while achieving significant reductions in model size and computational demands. Initial results indicate potential improvements in both efficiency and accuracy, paving the way for more sustainable deployment of language models in resource-constrained environments.

## Hypothesis
Integrating neural architecture search with structured pruning will allow for more efficient and effective model compression of large language models, leading to optimal trade-offs between model size reduction and performance preservation.

## Experiment Design
1. **Model Setup and Baseline Establishment**: Use existing large language models (e.g., BERT) as a baseline for model compression trials.
   
2. **Neural Architecture Search (NAS) Integration**: Apply NAS techniques to guide structured pruning decisions, identifying critical model components for retention or removal.

3. **Implementation of Hybrid Approach**: Develop and implement a structured pruning strategy augmented by NAS-driven optimization.

4. **Evaluation Criteria**: Measure performance metrics, model size, and inference time on standard language processing benchmarks and compare with traditional structured pruning methods.

5. **Analysis**: Evaluate the trade-off between computational resource savings and the impact on model accuracy.

6. **Iterations and Adjustments**: Conduct multiple iterations, refining the NAS algorithm parameters to improve compression efficiency and model performance.

## References
[1] Sanh, Victor, et al. "Compressed BERT: A Study on Model Compression for Large Pretrained Language Models." arXiv preprint arXiv:2006.01170 (2020).

[2] Liu, Peter J., et al. "Transformers with Low-Rank Approximations." arXiv preprint arXiv:2002.07078 (2020).

[3] Sun, Siqi, et al. "Layer-wise Distillation for BERT Model Compression." In The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.

[4] Zhu, Michael, and Suyog Gupta. "Structured Pruning of Large Language Models." arXiv preprint arXiv:2007.10347 (2020).

[5] Tay, Yi, et al. "Efficient Transformers: A Survey." arXiv preprint arXiv:2009.06732 (2020).
"""

new_idea = """
# Title
Dynamic Language Model Compression Using Reinforcement and Meta-Learning

## Abstract
The rapid growth of language model sizes demands efficient compression methods to deploy these models in resource-constrained environments. This research proposes a novel framework that integrates reinforcement learning and meta-learning for dynamic language model compression. Our approach aims to optimize compression strategies automatically based on varying performance criteria and deployment contexts. By leveraging reinforcement learning for decision-making in compression strategy selection and meta-learning for rapid adaptation of compressed models, we hypothesize significant reductions in model size without sacrificing performance. This study evaluates the proposed framework against traditional compression techniques through experiments on benchmark NLP tasks.

## Hypothesis
A reinforcement learning-based automatic compression system, augmented by meta-learning for model adaptation, can outperform traditional compression methods in maintaining language model performance across different deployment environments.

## Experiment Design
1. **Framework Development**: 
   - Develop a reinforcement learning model to autonomously select and apply appropriate compression techniques (e.g., pruning, quantization, etc.) based on performance feedback.
   - Use a meta-learning framework to rapidly adjust the model for varying tasks or environments post-compression.

2. **Benchmarking**:
   - Select pre-trained base models, such as BERT or GPT-2, and apply the proposed compression framework.
   - Compare performance metrics, including accuracy, inference speed, and memory usage, against baseline models compressed using traditional methods.

3. **Evaluation Settings**:
   - Conduct experiments across diverse NLP tasks to evaluate model generalization post-compression.
   - Test the adaptability of the compressed model when transferred to novel tasks, demonstrating meta-learning capabilities.

## References
[1] Ganesh, Pradeep Kumar et al. “Compression of Deep Learning Models for Text: A Survey.” arXiv preprint arXiv:2108.04612 (2021).

[2] Lagunas, Ferran et al. “Pruning Transformers for Low-latency Inference.” arXiv preprint arXiv:2103.01410 (2021).

[3] Fan, Angela et al. “Quant-Noise for Training Extremely Tiny Neural Networks.” arXiv preprint arXiv:2004.07320 (2020).

[4] Gou, Jianping et al. “Knowledge Distillation: A Survey.” International Journal of Computer Vision (2021): 1-31.

[5] Mao, Yiping et al. “Low-Rank Transformers: Unifying Model Compression and Acceleration.” arXiv preprint arXiv:2110.05866 (2021).

[6] Finn, Chelsea, et al. "Provably Efficient Neural Network Adaptation via Meta-Learning." arXiv preprint arXiv:1909.03971 (2019).

[7] He, Yihui, et al. "Automated Deep Compression: Compressing Weights of Deep Neural Networks with Reinforcement Learning." arXiv preprint arXiv:1802.03494 (2018).

[8] Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." arXiv preprint arXiv:1810.04805 (2018).

[9] Liu, Hanxiao, et al. "A Meta-Learning Approach to One-Step Active Learning for Few-Shot Neural Architecture Search." arXiv preprint arXiv:2001.10033 (2020).
"""

def test_idea_arena():
    topic = "Generated some scientific research idea related to language model compression"
    rewarder = IdeaArena(
        base_url=base_url,
        api_key=api_key,
        topic=topic
    )
    for i in range(10):
        rewarder.add_idea(old_idea)
    reward, _ = rewarder.get_reward([Context(key='gen_idea', content=new_idea)])
    print(reward)

def main():
    parser = ArgumentParser()
    parser.add_argument("-f", "--func", required=True, help="The testing function")

    args = parser.parse_args()
    func = args.func
    eval(func)()

if __name__ == "__main__":
    main()