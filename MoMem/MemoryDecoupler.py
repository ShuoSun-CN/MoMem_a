import re
import asyncio
from typing import Any

try:
    from .AgentClient import AgentClient
    from .ExplicitMemoryPlus import ExplicitMemoryPlus
    from .ImplicitMemory import ImplicitMemory
    from .prompts import (
        INTERACTIVE_REASONING_SYSTEM_PROMPT,
        normalize_prompt_language,
        render_prompt,
    )
except ImportError:
    from AgentClient import AgentClient
    from ImplicitMemory import ImplicitMemory
    from prompts import (
        INTERACTIVE_REASONING_SYSTEM_PROMPT,
        normalize_prompt_language,
        render_prompt,
    )
    from ExplicitMemoryPlus import ExplicitMemoryPlus

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class MemoryDecoupler:
    def __init__(self, decouple_agent=None, prompt_language: str = "en", verbose: bool = False):
        self.decouple_agent = decouple_agent if decouple_agent else AgentClient()
        self.prompt_language = normalize_prompt_language(prompt_language)
        self.verbose = verbose

    def _extract_bool(self, response: str, field_name: str) -> bool:
        if not response:
            return False

        escaped_field = re.escape(field_name)

        strict_pattern = rf"{escaped_field}\s*:\s*\$\$\$(True|False)\$\$\$"
        strict_matches = re.findall(strict_pattern, response, flags=re.IGNORECASE)
        if strict_matches:
            return strict_matches[-1].lower() == "true"

        loose_pattern = rf"{escaped_field}\s*:\s*(True|False)"
        loose_matches = re.findall(loose_pattern, response, flags=re.IGNORECASE)
        if loose_matches:
            return loose_matches[-1].lower() == "true"

        return False

    def _stringify_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        chunks.append(str(text).strip())
                elif item:
                    chunks.append(str(item).strip())
            return "\n".join(chunk for chunk in chunks if chunk)
        return str(content).strip()

    def _normalize_messages(self, messages) -> list[dict[str, str]]:
        if not isinstance(messages, list):
            return []

        normalized_messages = []
        for message in messages:
            if not isinstance(message, dict):
                continue

            role = str(message.get("role", "")).strip()
            content = self._stringify_content(message.get("content"))
            if not role or not content:
                continue

            normalized_messages.append({"role": role, "content": content})

        return normalized_messages

    def _format_messages_for_prompt(self, messages) -> str:
        normalized_messages = self._normalize_messages(messages)
        if not normalized_messages:
            return str(messages)

        formatted_lines = []
        for message in normalized_messages:
            formatted_lines.append(f"{message['role']}: {message['content']}")
        return "\n".join(formatted_lines)

    def _format_pair_with_context_for_prompt(
        self,
        context_messages,
        user_message,
        assistant_message,
    ) -> str:
        context_text = self._format_messages_for_prompt(context_messages)
        current_text = self._format_messages_for_prompt([user_message, assistant_message])

        if not context_text:
            return current_text

        return (
            "Earlier context (use only to resolve references and background in the current interaction; "
            "do not output True only because this section contains facts):\n"
            f"{context_text}\n\n"
            "Current interaction to judge (judge only whether this turn adds, confirms, supplements, "
            "or updates reusable information):\n"
            f"{current_text}"
        )

    def confirm_explicit_truth(self, formatted_messages: str) -> tuple[bool, str]:
        prompt = render_prompt(
            "decoupler_confirm_explicit_truth",
            self.prompt_language,
            formatted_messages=formatted_messages,
        )
        response = self.decouple_agent.complete(prompt)
        field_name = "Save text record"
        
        return self._extract_bool(response, field_name), response

    def confirm_implicit_truth(self, formatted_messages: str) -> tuple[bool, str]:
        prompt = render_prompt(
            "decoupler_confirm_implicit_truth",
            self.prompt_language,
            formatted_messages=formatted_messages,
        )
        response = self.decouple_agent.complete(prompt)
        field_name = "Save method experience"
        
        return self._extract_bool(response, field_name), response

    def decouple(
        self,
        messages,
        ex_mem: ExplicitMemoryPlus,
        im_mem: ImplicitMemory,
        ex_update: bool = True,
        im_update: bool = True,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        explicit_truth_list=[]
        implicit_truth_list=[]
        explicit_response_list=[]
        implicit_response_list=[]
        logic_trajectory_entry_list = []
        
        if messages[0].get("role") == "system":
            messages = messages[1:]
            
        for user_message, assistant_message in zip(messages[::2], messages[1::2]):
            if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
                raise ValueError("Messages should alternate between user and assistant roles.")
            formatted_messages = self._format_messages_for_prompt([user_message, assistant_message])

            explicit_truth, response_explicit = self.confirm_explicit_truth(formatted_messages)

            if self.verbose:
                print(formatted_messages)
            implicit_truth, response_implicit = self.confirm_implicit_truth(formatted_messages)

            logic_trajectory_entry = ""

            if self.verbose:
                print({
                    "explicit_truth": explicit_truth,
                    "implicit_truth": implicit_truth,
                    "raw_response_explicit": response_explicit,
                    "raw_response_implicit": response_implicit,
                })
            
            explicit_truth_list.append(explicit_truth)
            implicit_truth_list.append(implicit_truth) 
            explicit_response_list.append(response_explicit)
            implicit_response_list.append(response_implicit)

            if ex_update and explicit_truth and ex_mem is not None:
                ex_mem.add(
                    [user_message, assistant_message],
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                )

            if im_update and implicit_truth and im_mem is not None:
                update_result = im_mem.update_implicit_experience(
                    messages=[user_message, assistant_message],
                    experiential_signal=response_implicit,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                )
                if isinstance(update_result, dict):
                    logic_trajectory_entry = update_result.get("logic_trajectory_entry", "")
            logic_trajectory_entry_list.append(logic_trajectory_entry)
        
        return {
            "explicit_truth_list": explicit_truth_list,
            "implicit_truth_list": implicit_truth_list,
            "logic_trajectory_entry_list": logic_trajectory_entry_list,
            "logic_trajectory_entry": logic_trajectory_entry_list[-1] if logic_trajectory_entry_list else "",
            "raw_response_explicit": explicit_response_list,
            "raw_response_implicit": implicit_response_list,
        }

    def decouple_batch_only(
        self,
        messages_batch: list[list[dict]],
        ex_mem: ExplicitMemoryPlus | None = None,
        im_mem: ImplicitMemory | None = None,
        ex_update: bool = False,
        im_update: bool = False,
        concurrency: int = 8,
        show_progress: bool = False,
        progress_every: int = 100,
        context_window_pairs: int = 0,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        """Run batched decoupling while preserving the single-pair prompt format."""
        if not isinstance(messages_batch, list):
            raise ValueError("messages_batch must be a List[List[Dict]].")
        if concurrency <= 0:
            raise ValueError("concurrency must be greater than 0.")
        if progress_every <= 0:
            raise ValueError("progress_every must be greater than 0.")
        if context_window_pairs < 0:
            raise ValueError("context_window_pairs cannot be negative.")

        def _progress_log(message: str, progress_bar=None) -> None:
            if not show_progress:
                return

            if progress_bar is not None and tqdm is not None:
                progress_bar.write(message)
            else:
                print(message)

        def _create_progress_bar(total: int, desc: str):
            if not show_progress or total <= 0 or tqdm is None:
                return None

            return tqdm(
                total=total,
                desc=desc,
                unit="pair",
                dynamic_ncols=True,
                smoothing=0.1,
            )

        def _update_progress_bar(progress_bar, explicit_count: int, implicit_count: int) -> None:
            if progress_bar is None:
                return

            progress_bar.update(1)
            progress_bar.set_postfix(
                explicit=explicit_count,
                implicit=implicit_count,
                refresh=False,
            )

        if show_progress:
            _progress_log(
                f"[decouple_batch_only] start: conversations={len(messages_batch)}, "
                f"concurrency={concurrency}"
            )
            if tqdm is None:
                print("[decouple_batch_only] tqdm is unavailable; using text progress.")

        pair_records = []
        for conv_idx, messages in enumerate(messages_batch):
            normalized = self._normalize_messages(messages)
            if normalized and normalized[0].get("role") == "system":
                normalized = normalized[1:]

            pairs = []
            for user_message, assistant_message in zip(normalized[::2], normalized[1::2]):
                if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
                    continue
                pairs.append((user_message, assistant_message))

            for pair_idx, (user_message, assistant_message) in enumerate(pairs):
                context_messages = []
                if context_window_pairs:
                    start_idx = max(0, pair_idx - context_window_pairs)
                    for context_user, context_assistant in pairs[start_idx:pair_idx]:
                        context_messages.extend([context_user, context_assistant])
                pair_records.append(
                    {
                        "conv_idx": conv_idx,
                        "pair_idx": pair_idx,
                        "context_messages": context_messages,
                        "user_message": user_message,
                        "assistant_message": assistant_message,
                    }
                )

        if not pair_records:
            return {
                "explicit_truth_list": [],
                "implicit_truth_list": [],
                "raw_response_explicit": [],
                "raw_response_implicit": [],
                "logic_trajectory_entry_list": [],
            }
        if show_progress:
            _progress_log(f"[decouple_batch_only] preprocessing done: valid_pairs={len(pair_records)}")

        async def _run_parallel_decouple(records: list[dict], limit: int):
            sem = asyncio.Semaphore(limit)
            total = len(records)
            results = [None] * total

            async def _worker(idx: int, record: dict):
                try:
                    formatted = self._format_pair_with_context_for_prompt(
                        record.get("context_messages") or [],
                        record["user_message"],
                        record["assistant_message"],
                    )
                    async with sem:
                        explicit_truth, response_explicit = await asyncio.to_thread(
                            self.confirm_explicit_truth, formatted
                        )
                    async with sem:
                        implicit_truth, response_implicit = await asyncio.to_thread(
                            self.confirm_implicit_truth, formatted
                        )
                    return idx, {
                        "explicit_truth": explicit_truth,
                        "implicit_truth": implicit_truth,
                        "raw_response_explicit": response_explicit,
                        "raw_response_implicit": response_implicit,
                    }
                except Exception as e:
                    return idx, {
                        "explicit_truth": False,
                        "implicit_truth": False,
                        "raw_response_explicit": "",
                        "raw_response_implicit": f"Exception: {e}",
                    }

            tasks = [asyncio.create_task(_worker(i, r)) for i, r in enumerate(records)]
            done = 0
            explicit_done = 0
            implicit_done = 0
            progress_bar = _create_progress_bar(total, "model judgment")
            try:
                for task in asyncio.as_completed(tasks):
                    idx, item = await task
                    results[idx] = item
                    done += 1
                    explicit_done += int(item["explicit_truth"])
                    implicit_done += int(item["implicit_truth"])

                    if progress_bar is not None:
                        _update_progress_bar(progress_bar, explicit_done, implicit_done)
                    elif show_progress and (done % progress_every == 0 or done == total):
                        print(f"[decouple_batch_only] model judgment progress: {done}/{total}")
            finally:
                if progress_bar is not None:
                    progress_bar.close()

            return results

        try:
            asyncio.get_running_loop()
            in_running_loop = True
        except RuntimeError:
            in_running_loop = False

        parsed_results = []
        if in_running_loop:
            progress_bar = _create_progress_bar(len(pair_records), "model judgment")
            explicit_done = 0
            implicit_done = 0
            try:
                for i, record in enumerate(pair_records, start=1):
                    formatted = self._format_pair_with_context_for_prompt(
                        record.get("context_messages") or [],
                        record["user_message"],
                        record["assistant_message"],
                    )
                    explicit_truth, response_explicit = self.confirm_explicit_truth(formatted)
                    implicit_truth, response_implicit = self.confirm_implicit_truth(formatted)
                    parsed_results.append(
                        {
                            "explicit_truth": explicit_truth,
                            "implicit_truth": implicit_truth,
                            "raw_response_explicit": response_explicit,
                            "raw_response_implicit": response_implicit,
                        }
                    )
                    explicit_done += int(explicit_truth)
                    implicit_done += int(implicit_truth)

                    if progress_bar is not None:
                        _update_progress_bar(progress_bar, explicit_done, implicit_done)
                    elif show_progress and (i % progress_every == 0 or i == len(pair_records)):
                        print(f"[decouple_batch_only] model judgment progress: {i}/{len(pair_records)}")
            finally:
                if progress_bar is not None:
                    progress_bar.close()
        else:
            parsed_results = asyncio.run(_run_parallel_decouple(pair_records, concurrency))

        explicit_truth_list = []
        implicit_truth_list = []
        explicit_response_list = []
        implicit_response_list = []
        logic_trajectory_entry_list = []

        for idx, parsed in enumerate(parsed_results):
            explicit_truth = parsed["explicit_truth"]
            implicit_truth = parsed["implicit_truth"]

            explicit_truth_list.append(explicit_truth)
            implicit_truth_list.append(implicit_truth)
            explicit_response_list.append(parsed["raw_response_explicit"])
            implicit_response_list.append(parsed["raw_response_implicit"])

            record = pair_records[idx]
            user_message = record["user_message"]
            assistant_message = record["assistant_message"]

            if ex_update and explicit_truth and ex_mem is not None:
                ex_mem.add(
                    [user_message, assistant_message],
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                )

            logic_trajectory_entry = ""
            if im_update and implicit_truth and im_mem is not None:
                update_result = im_mem.update_implicit_experience(
                    messages=[user_message, assistant_message],
                    experiential_signal=parsed["raw_response_implicit"],
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                )
                if isinstance(update_result, dict):
                    logic_trajectory_entry = update_result.get("logic_trajectory_entry", "")

            logic_trajectory_entry_list.append(logic_trajectory_entry)

        if show_progress:
            print("[decouple_batch_only] done")

        return {
            "pair_records": pair_records,
            "explicit_truth_list": explicit_truth_list,
            "implicit_truth_list": implicit_truth_list,
            "raw_response_explicit": explicit_response_list,
            "raw_response_implicit": implicit_response_list,
            "logic_trajectory_entry_list": logic_trajectory_entry_list,
        }


if __name__=="__main__":
    from AgentClient import AgentClient
    agent=AgentClient("local")
    Mem=MemoryDecoupler(decouple_agent=agent)
    while True:
        q=input()
        if q=="exit":
            break
        system_prompt=INTERACTIVE_REASONING_SYSTEM_PROMPT
        response = agent.complete(
                    q,
                    system_prompt=system_prompt
                )
        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content":response},
            ]
        print(Mem.decouple(messages,None,None))
