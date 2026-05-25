"""Centralized English prompt templates for MoMem."""

DEFAULT_SYSTEM_PROMPT = (
    "Answer the request directly. Use step-by-step reasoning only when needed. "
    "Put the final answer in boxed{} when appropriate."
)

INTERACTIVE_REASONING_SYSTEM_PROMPT = (
    "If you need reason, please reason step by step, and put your final answer within boxed{}."
)


def normalize_prompt_language(prompt_language: str | None) -> str:
    return "en"


def render_prompt(name: str, prompt_language: str | None = "en", **kwargs) -> str:
    try:
        template = PROMPT_TEMPLATES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt template: {name!r}") from exc
    return template.format(**kwargs)


PROMPT_TEMPLATES = {
    "router_pre_reason": """Task: you will see a user request. Decide whether an experience query should be constructed; if so, write a very brief experience-query sketch. Do not answer the request itself.

Here, "experience-query sketch" only means writing an abstract description of the handling logic the current request may need, so it can later be compared with method summaries from past tasks. The sketch must describe how to handle this kind of problem, without including the current answer, concrete numbers, problem-specific details, or intermediate computation.

Focus on whether the request can yield reusable handling logic, implementation order, check points, or negative lessons. Do not judge by response length, number of steps, or surface complexity; a short answer, direct implementation, or basic calculation may still contain reusable experience. Even if the handling is lightweight, write a sketch when it could remind a future task how to proceed, what to check first, or what mistake to avoid.

Do not write a sketch when the request is only factual lookup, translation, rewriting, summarization, format conversion, or casual chat, or when you can only name the topic but cannot describe any handling logic, check point, implementation order, or negative lesson.

If True, the sketch must include:
- Methods used: likely methods, strategies, implementation choices, check points, or issues to avoid, without named entities, concrete numbers, file names, or final answers.
- Rough reasoning-step sketch: 2 to 5 coarse ordered steps, without solving the task.

Output format must be exactly:
Build experience query: $$$True$$$
or
Build experience query: $$$False$$$
Experience-query sketch: <<<
[if True, write:
Methods used: ...
Rough reasoning-step sketch: ...
if False, write NONE]
>>>

Current request:
{query}
""",
    "router_filter_explicit_memory": """Task: you will see a current request and several saved text records from the past. Output only the record content that can be directly used in the current answer.

Nature of the candidate records: they are directly reusable text records, such as user facts, preferences, constraints, environment state, prior commitments, definitions, formulas, theorems, short rules, checklists, or compact workflows.

Output relevant content when:
- The records contain concrete information that is required or clearly helpful for answering the current request.
- The request refers to prior information, such as "what I said before", "use my preference", "reuse the previous setting", or "continue the earlier topic".
- Injecting the records would improve factual accuracy, consistency, or constraint following.

Output NONE when:
- The records are empty, irrelevant, or only topically similar without materially helping the answer.
- The records are mainly long proofs, full derivations, solution transcripts, or only similar in topic rather than directly reusable facts/rules/procedures.
- The request needs new reasoning, coding, planning, or proof, but the records provide no directly reusable information.

Requirements:
- Do not output the decision rationale.
- Do not output irrelevant candidate records.
- You may merge, deduplicate, or lightly rewrite relevant records, but do not invent information absent from the candidates.
- Do not decide whether to use learned problem-solving/debugging/planning experience; this task only filters text records that can be placed directly into context.

Current request:
{query}

Candidate records:
{explicit_memories}

Output format must be exactly:
Directly usable text records: <<<
[relevant content; write NONE if there is no relevant content]
>>>
""",
    "router_confirm_implicit_activate": """Task: you will see the current request, an experience-query sketch for the current request, and several method summaries from past tasks. Decide whether to activate past experience.

The past-task method summaries are not context to place in the answer, and they are not hints for solving the current problem. Use them only to decide whether the current request may benefit from learned handling experience, check points, or negative lessons.

Do not require the past task and the current request to be fully isomorphic. A past summary does not need to cover the full solution of the current request; it is enough if it can help at one local stage, such as suggesting a modeling angle, decomposition order, verification habit, boundary check, or assumption that is easy to get wrong. The decision is whether the model should be allowed to refer to these handling experiences, not whether the past experience can directly solve the current request.

Do not use past experience when the summaries are empty, or when they only share domain words or keywords but cannot suggest any handling action, checking action, or error-avoidance cue; also do not use it when the current request is only factual lookup, translation, rewriting, summarization, format conversion, or casual chat.

Current request:
{query}

Current request experience-query sketch:
{query_logic}

Past-task method summaries:
{logic_trajectory_evidence}

Output exactly one line:
Activate experience: $$$True$$$
or
Activate experience: $$$False$$$
""",
    "decoupler_confirm_explicit_truth": """Task: you will see a completed user-assistant interaction. Decide whether it should be saved as a directly reusable text record.

A text record stores stable information that can be used later as background context. Such information should come from the user, the input material, or outside context itself, such as user preferences, environment state, project settings, dataset descriptions, long-term constraints, or a task state that may be continued later. Its value is that a future answer can directly know this information.

Do not treat content produced by the assistant's reasoning, calculation, proof, coding, debugging, or verification in this interaction as a text record. Even if an answer, conclusion, implementation result, or problem condition is specific, queryable, and writable as one sentence, it should not be saved when it is mainly part of the current task instance rather than independent background state for future use.

If the value of the information is mainly "how to obtain it, how to handle similar problems, or how to avoid mistakes", it does not belong in this text-record store.

Output requirements:
- First give a brief rationale in 1 to 3 sentences
- Then output one field:
Reusable text record: <<<
[directly reusable text record; write NONE if False]
>>>
- Final line must be exactly: Save text record: $$$True$$$ or Save text record: $$$False$$$

Conversation Messages:
{formatted_messages}
""",
    "decoupler_confirm_implicit_truth": """Task: you will see a completed user-assistant interaction. Decide whether it demonstrates a handling method worth reusing later, and extract its methods and step summary.

Reusable handling experience is a way of solving, planning, debugging, verifying, or using tools that could help on future tasks with a similar structure but different surface details. It stores the experience of how to handle the task, not the final answer to this task.

Focus on whether the interaction shows an input-to-output handling structure: how the problem was decomposed, how a method was chosen, how reasoning was organized, how results were checked, how failures were located, or how a solution was repaired. Correct solutions can be saved; wrong or incomplete solutions should not be dismissed automatically, because they may reveal transferable pitfalls, missing checks, wrong assumptions, or paths to avoid. Negative experience does not require the interaction to complete the repair; if a reusable error type, failure cause, or missing check is visible, save it as negative experience rather than presenting the invalid method as correct.

Even if no complete reproducible method can be extracted, save negative experience when you can name an error pattern, unverified assumption, skipped inference, or missing check that future tasks should avoid.

Do not judge reusability by response length, number of steps, or surface complexity. A short answer, direct implementation, or basic calculation may still contain reusable handling experience. The key is whether you can extract an underlying logic, check point, implementation order, or negative lesson that could guide future tasks.

Do not save user facts, environment state, or stable text records that can be placed directly into future context. If the interaction only recalls facts, translates or rewrites text, changes format, do not save it.

If True, the three fields below must summarize the transferable structure. If False, write NONE for all three fields.

Output requirements:
- First provide Evidence snippets (up to 3 short phrases)
- Then explain in 1 to 3 sentences why the signal is or is not worth storing as transferable processing experience
- Then output three fields:
The question mainly cause reason: <<<
[question that cause reason]
>>>
Methods used: <<<
[methods used; write NONE if False]
>>>
Rough reasoning-step sketch: <<<
[rough reasoning-step sketch; write NONE if False]
>>>
- Final line must be exactly: Save method experience: $$$True$$$ or Save method experience: $$$False$$$

Conversation Messages:
{formatted_messages}
""",
}
