"""Factory functions for creating direct prompting predictors.

This module provides convenient factory functions for instantiating
direct prompting predictors with different models and prompt templates.
"""

from __future__ import annotations

from typing import Dict

from experiments.baselines.direct_prompting import DirectPromptingPredictor


def create_direct_prompting_predictor(
    model_path: str,
    prompt_template: str,
    prompt_name: str = "direct_prompt",
    device: str = "cpu",
    max_new_tokens: int = 10,
    temperature: float = 0.0,
) -> DirectPromptingPredictor:
    """Create a DirectPromptingPredictor for a specific model and prompt.

    Args:
        model_path: HuggingFace model identifier
        prompt_template: Template string with {prompt} placeholder
        prompt_name: Prompt identifier (e.g., "prompt_0", "prompt_1", etc.)
        device: Device for inference ("cpu", "cuda", or "mps")
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy)

    Returns:
        Configured DirectPromptingPredictor instance

    Example:
        >>> predictor = create_direct_prompting_predictor(
        ...     model_path="meta-llama/Llama-3.2-3B-Instruct",
        ...     prompt_template="Is this safe (0) or unsafe (1)? {prompt}",
        ...     prompt_name="prompt_0",
        ...     device="cuda"
        ... )
    """
    return DirectPromptingPredictor(
        model_path=model_path,
        prompt_template=prompt_template,
        prompt_name=prompt_name,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def create_all_prompt_predictors(
    model_path: str,
    prompts_dict: Dict[str, str],
    device: str = "cpu",
    max_new_tokens: int = 10,
    temperature: float = 0.0,
) -> Dict[str, DirectPromptingPredictor]:
    """Create predictors for all prompt templates in a dictionary.

    Args:
        model_path: HuggingFace model identifier
        prompts_dict: Dictionary mapping prompt names to template strings
        device: Device for inference ("cpu", "cuda", or "mps")
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy)

    Returns:
        Dictionary mapping prompt_name to DirectPromptingPredictor instance

    Example:
        >>> prompts = {
        ...     "prompt_0": "Is this safe (0) or unsafe (1)? {prompt}",
        ...     "prompt_1": "Classify as safe (0) or unsafe (1): {prompt}",
        ... }
        >>> predictors = create_all_prompt_predictors(
        ...     model_path="meta-llama/Llama-3.2-3B-Instruct",
        ...     prompts_dict=prompts,
        ...     device="cuda"
        ... )
        >>> # Use predictors["prompt_0"], predictors["prompt_1"], etc.
    """
    predictors = {}

    for prompt_name, prompt_template in prompts_dict.items():
        predictors[prompt_name] = create_direct_prompting_predictor(
            model_path=model_path,
            prompt_template=prompt_template,
            prompt_name=prompt_name,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    return predictors
