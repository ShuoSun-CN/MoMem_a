from pathlib import Path
from copy import deepcopy
from typing import Any
import os
import re

try:
    from .ExplicitMemory import ExplicitMemoryMem0 as LogicTrajectoryBank
    from .config import DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG
except ImportError:
    from ExplicitMemory import ExplicitMemoryMem0 as LogicTrajectoryBank
    from config import DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG

NEW_EXPLICIT_MEMORY_MEM0_CONFIG = deepcopy(DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG)
logic_trajectory_bank_path = os.getenv(
    "MOMEM_LOGIC_TRAJECTORY_PATH",
    str(Path.home() / ".cache" / "momem" / "logic_trajectory_bank"),
)
NEW_EXPLICIT_MEMORY_MEM0_CONFIG["model_config"]["vector_store"]["config"]["path"] = logic_trajectory_bank_path



class ImplicitMemory:
    def __init__(
        self,
        base_model,
        lora_path,
        need_grpo,
        reward_func=None,
        reward_name=None,
        system_prompt="",
        reward_mode=None,
    ):
        self.lora_path = lora_path
        self.grpo_lora = None
        if need_grpo:
            try:
                from .OnlineGRPO import OnlineGRPO
            except ImportError:
                from OnlineGRPO import OnlineGRPO

            self.grpo_lora = OnlineGRPO(
                base_model,
                lora_path,
                reward_func=reward_func,
                reward_name=reward_name,
                system_prompt=system_prompt,
                reward_mode=reward_mode,
            )

        self.logic_trajectory_bank = LogicTrajectoryBank(config=NEW_EXPLICIT_MEMORY_MEM0_CONFIG)

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
            if role and content:
                normalized_messages.append({"role": role, "content": content})
        return normalized_messages

    def _extract_last_user_query(self, messages) -> str:
        for message in reversed(self._normalize_messages(messages)):
            if message["role"] == "user":
                return message["content"]
        return ""

    def _build_generation_messages(self, prompt, system_prompt: str = ""):
        if isinstance(prompt, list):
            return prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self._stringify_content(prompt)})
        return messages

    def _normalize_logic_descriptor(self, logic_descriptor: str | None) -> str:
        clean_text = self._stringify_content(logic_descriptor)
        if not clean_text:
            return ""

        lowered = clean_text.lower()
        if lowered in {"none", "n/a", "null", "no signal", "no experiential signal"}:
            return ""
        return clean_text

    def _extract_block(self, response: str, field_name: str) -> str:
        if not response:
            return ""

        escaped_field = re.escape(field_name)
        block_pattern = rf"{escaped_field}\s*:\s*<<<\s*(.*?)\s*>>>"
        block_matches = re.findall(block_pattern, response, flags=re.IGNORECASE | re.DOTALL)
        if block_matches:
            return block_matches[-1].strip()

        return ""

    def _extract_experiential_signal(self, response: str | None) -> str:
        response_text = self._stringify_content(response)
        if not response_text:
            return ""

        question_reason = self._normalize_logic_descriptor(
            self._extract_block(response_text, "The question mainly cause reason")
        )
        methods_used = self._normalize_logic_descriptor(
            self._extract_block(response_text, "Methods used")
        )
        step_sketch = self._normalize_logic_descriptor(
            self._extract_block(response_text, "Rough reasoning-step sketch")
        )

        if question_reason or methods_used or step_sketch:
            parts = []
            if question_reason:
                parts.append(f"The question mainly cause reason: {question_reason}")
            if methods_used:
                parts.append(f"Methods used: {methods_used}")
            if step_sketch:
                parts.append(f"Rough reasoning-step sketch: {step_sketch}")
            return "\n".join(parts).strip()

        summary = self._normalize_logic_descriptor(
            self._extract_block(response_text, "Experiential summary")
        )
        if summary:
            return summary

        return self._normalize_logic_descriptor(response_text)

    def build_logic_descriptor(self, messages, experiential_signal: str | None = None) -> str:
        return self._extract_experiential_signal(experiential_signal)

    def build_logic_trajectory_entry(self, messages, logic_descriptor: str | None = None) -> str:
        """Build p_i=(x_i, l_i) text for the Logic-Indexed Trajectory Bank."""
        descriptor = self._normalize_logic_descriptor(logic_descriptor)
        if not descriptor:
            return ""

        user_query = self._extract_last_user_query(messages)
        entry_parts = []
        if user_query:
            entry_parts.append(f"User Query: {user_query}")
        entry_parts.append(descriptor)
        return "\n".join(entry_parts).strip()

    def retrieve_logic_trajectory_evidence(self, query, user_id=None, agent_id=None, run_id=None):
        if not query:
            return []

        relevant_memories = self.logic_trajectory_bank.get_mem(
            query=query,
            limit=3,
            type="str",
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )

        return relevant_memories

    def update_lora_experience(self, messages):
        if self.grpo_lora is None:
            return
        
        self.grpo_lora.update(messages)

    def generate_with_implicit(self, prompt, system_prompt: str = "", **generation_kwargs):
        if self.grpo_lora is None:
            raise RuntimeError("ImplicitMemory was not initialized with OnlineGRPO.")
        messages = self._build_generation_messages(prompt, system_prompt=system_prompt)
        return self.grpo_lora.generate(messages, **generation_kwargs)

    def generate_base(self, prompt, system_prompt: str = "", **generation_kwargs):
        if self.grpo_lora is None:
            raise RuntimeError("ImplicitMemory was not initialized with OnlineGRPO.")
        messages = self._build_generation_messages(prompt, system_prompt=system_prompt)
        return self.grpo_lora.generate_base(messages, **generation_kwargs)

    def update_implicit_experience(
        self,
        messages,
        experiential_signal: str | None = None,
        logic_descriptor: str | None = None,
        update_lora: bool = True,
        update_trajectory_bank: bool = True,
        user_id=None,
        agent_id=None,
        run_id=None,
    ) -> dict:
        descriptor = self._normalize_logic_descriptor(logic_descriptor)
        if not descriptor:
            descriptor = self.build_logic_descriptor(
                messages=messages,
                experiential_signal=experiential_signal,
            )

        logic_trajectory_entry = ""
        if update_trajectory_bank:
            logic_trajectory_entry = self.build_logic_trajectory_entry(
                messages=messages,
                logic_descriptor=descriptor,
            )
            if logic_trajectory_entry:
                self.logic_trajectory_bank.add(
                    logic_trajectory_entry,
                    infer=False,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                )

        if update_lora:
            self.update_lora_experience(messages)

        return {
            "logic_descriptor": descriptor,
            "logic_trajectory_entry": logic_trajectory_entry,
        }

    def update(
        self,
        messages,
        experiential_signal: str | None = None,
        logic_descriptor: str | None = None,
        **kwargs,
    ):
        return self.update_implicit_experience(
            messages=messages,
            experiential_signal=experiential_signal,
            logic_descriptor=logic_descriptor,
            **kwargs,
        )

    def update_all(
        self,
        messages,
        experiential_signal: str | None = None,
        logic_descriptor: str | None = None,
        **kwargs,
    ):
        return self.update(
            messages=messages,
            experiential_signal=experiential_signal,
            logic_descriptor=logic_descriptor,
            **kwargs,
        )
        
    
    
