"""Direct LLM Agent - No retrieval, just generation.

Uses generator provider for explicit lifecycle control.
Generator gets full GPU for maximum throughput.
"""

from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from ragicamp.agents.base import Agent, AgentResult, Query, Step, StepTimer
from ragicamp.core.logging import get_logger
from ragicamp.core.step_types import BATCH_GENERATE, GENERATE
from ragicamp.models.providers import GeneratorProvider
from ragicamp.utils.prompts import PromptBuilder, PromptConfig

logger = get_logger(__name__)


class DirectLLMAgent(Agent):
    """Direct LLM agent - just generation, no retrieval.

    Execution:
    1. Load generator (full GPU)
    2. Batch generate all answers
    3. Unload generator
    """

    def __init__(
        self,
        name: str,
        generator_provider: GeneratorProvider,
        prompt_builder: PromptBuilder | None = None,
    ):
        """Initialize agent with generator provider.

        Args:
            name: Agent identifier
            generator_provider: Provides generator with lazy loading
            prompt_builder: For building prompts
        """
        super().__init__(name)

        self.generator_provider = generator_provider
        self.prompt_builder = prompt_builder or PromptBuilder(PromptConfig())

    def run(
        self,
        queries: list[Query],
        *,
        on_result: Callable[[AgentResult], None] | None = None,
        checkpoint_path: Path | None = None,
        show_progress: bool = True,
    ) -> list[AgentResult]:
        """Process all queries with batch generation."""
        # Load checkpoint if resuming
        results, completed_idx = [], set()
        if checkpoint_path:
            results, completed_idx = self._load_checkpoint(checkpoint_path)

        pending = [q for q in queries if q.idx not in completed_idx]

        if not pending:
            logger.info("All queries already completed")
            return results

        logger.info("Processing %d queries (direct generation)", len(pending))

        # Build all prompts
        prompts = [self.prompt_builder.build_direct(q.text) for q in pending]

        # Load generator, generate, unload
        with self.generator_provider.load() as generator:
            with StepTimer(BATCH_GENERATE, model=self.generator_provider.model_name) as step:
                step.input = {"n_prompts": len(prompts)}
                answers = generator.batch_generate(prompts)
                step.output = {"n_answers": len(answers)}

        # Generator is now unloaded

        # Build results
        iterator = zip(pending, prompts, answers, strict=True)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Building results")

        new_results = []
        for query, prompt, answer in iterator:
            result = AgentResult(
                query=query,
                answer=answer,
                steps=[
                    Step(
                        type=GENERATE,
                        input=query.text,
                        output=answer,
                        model=self.generator_provider.model_name,
                    )
                ],
                prompt=prompt,
            )
            new_results.append(result)
            results.append(result)

            if on_result:
                on_result(result)

        if checkpoint_path:
            self._save_checkpoint(results, checkpoint_path)

        logger.info("Completed %d queries", len(new_results))
        return results


# =============================================================================
# Factory function for easy creation
# =============================================================================


def create_direct_llm_agent(
    name: str,
    generator_model: str,
    generator_backend: str = "vllm",
) -> DirectLLMAgent:
    """Create a DirectLLMAgent with provider.

    Args:
        name: Agent name
        generator_model: Generator model name
        generator_backend: "vllm" or "hf"

    Returns:
        Configured DirectLLMAgent
    """
    from ragicamp.models.providers import GeneratorConfig, GeneratorProvider

    generator_provider = GeneratorProvider(
        GeneratorConfig(
            model_name=generator_model,
            backend=generator_backend,
        )
    )

    return DirectLLMAgent(
        name=name,
        generator_provider=generator_provider,
    )
