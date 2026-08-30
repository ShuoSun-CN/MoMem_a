"""Reward-function templates for MoMem GRPO training."""

import re
from collections import Counter


def _completion_text(completion):
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first_item = completion[0]
        if isinstance(first_item, dict):
            return str(first_item.get("content") or first_item.get("text") or "").strip()
        return str(first_item).strip()
    if isinstance(completion, dict):
        return str(completion.get("content") or completion.get("text") or "").strip()
    return str(completion).strip()


def _prompt_text(prompt):
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt:
        first_item = prompt[0]
        if isinstance(first_item, dict):
            return str(first_item.get("content") or first_item.get("text") or "").strip()
        return str(first_item).strip()
    return str(prompt).strip()


def normalize_text(text):
    return " ".join(str(text).strip().lower().split())


def extract_boxed_answer(text):
    if text is None:
        return ""

    text = str(text).strip("\n\r")
    boxed_matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    return ""


def extract_numeric_answer(text):
    candidate = extract_boxed_answer(text) or str(text or "")
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", candidate)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return candidate.strip()


def normalize_answer(text):
    text = extract_numeric_answer(text)
    return re.sub(r"\s+", "", str(text).strip().lower()).replace(",", "")


def exact_match_reward(completions, gold_answer, **kwargs):
    """Return 1.0 when a completion matches the gold answer, else 0.0."""
    rewards = []
    if gold_answer is None:
        raise ValueError("gold_answer is required for exact_match_reward.")

    if not isinstance(gold_answer, list):
        gold_answer = [gold_answer] * len(completions)

    for completion, gold in zip(completions, gold_answer):
        pred = normalize_answer(_completion_text(completion))
        gold_norm = normalize_answer(gold)
        rewards.append(1.0 if pred and pred == gold_norm else 0.0)
    return rewards


def numeric_exact_match_reward(completions, gold_answer, **kwargs):
    """Exact-match reward after extracting the last numeric answer."""
    return exact_match_reward(completions, gold_answer=gold_answer, **kwargs)


def majority_vote_reward(prompts, completions, **kwargs):
    """Reward answers that agree with the consensus among completions for the same prompt."""
    prompt_clusters = {}
    for prompt, completion in zip(prompts, completions):
        prompt_text = _prompt_text(prompt)
        prompt_clusters.setdefault(prompt_text, []).append(_completion_text(completion))

    consensus_map = {}
    for prompt_text, grouped_completions in prompt_clusters.items():
        extracted_answers = [normalize_answer(text) for text in grouped_completions]
        extracted_answers = [answer for answer in extracted_answers if answer]
        if extracted_answers:
            consensus_map[prompt_text] = Counter(extracted_answers).most_common(1)[0][0]
        else:
            consensus_map[prompt_text] = None

    rewards = []
    for prompt, completion in zip(prompts, completions):
        prompt_text = _prompt_text(prompt)
        consensus = consensus_map.get(prompt_text)
        extracted = normalize_answer(_completion_text(completion))
        rewards.append(1.0 if consensus and extracted == consensus else 0.0)
    return rewards


def verifier_reward(completions, verifier_scores=None, **kwargs):
    """Pass through benchmark/verifier scores when they are already provided."""
    if verifier_scores is None:
        raise ValueError("verifier_scores is required for verifier_reward.")
    if not isinstance(verifier_scores, list):
        verifier_scores = [verifier_scores] * len(completions)
    return [float(score) for score in verifier_scores]


def execution_reward_template(completions, runner=None, **kwargs):
    """Template for code or tool environments that provide an external execution judge."""
    if runner is None:
        raise ValueError("execution_reward_template requires a runner callable.")
    return [1.0 if runner(_completion_text(completion), **kwargs) else 0.0 for completion in completions]


def dispatch_reward(
    completions,
    gold_answer=None,
    verifier_scores=None,
    reward_mode=None,
    runner=None,
    **kwargs,
):
    """Pick a reward path based on the signals available in the environment."""
    task_type = str(kwargs.get("task_type") or kwargs.get("dataset_name") or "").lower()

    if verifier_scores is not None:
        return verifier_reward(completions, verifier_scores=verifier_scores, **kwargs)

    if runner is not None or task_type in {"code", "program", "tool", "execution"}:
        return execution_reward_template(completions, runner=runner, **kwargs)

    if reward_mode == "numeric" or task_type in {"math", "arithmetic", "numeric"}:
        return numeric_exact_match_reward(completions, gold_answer=gold_answer, **kwargs)

    if gold_answer is not None:
        return exact_match_reward(completions, gold_answer=gold_answer, **kwargs)

    raise ValueError(
        "dispatch_reward could not infer a reward path. Provide gold_answer, "
        "verifier_scores, or a runner callable."
    )
