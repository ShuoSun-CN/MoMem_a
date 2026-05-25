import re

try:
    from .AgentClient import AgentClient
    from .ImplicitMemory import ImplicitMemory
    from .ExplicitMemoryPlus import ExplicitMemoryPlus
    from .prompts import normalize_prompt_language, render_prompt
except ImportError:
    from AgentClient import AgentClient
    from ImplicitMemory import ImplicitMemory
    from ExplicitMemoryPlus import ExplicitMemoryPlus
    from prompts import normalize_prompt_language, render_prompt


class MemoryRouter:
    def __init__(self, route_agent=None, prompt_language: str = "en"):
        self.route_agent = route_agent
        self.api_client = AgentClient() if not route_agent else route_agent
        self.prompt_language = normalize_prompt_language(prompt_language)

    def _extract_bool(self, response: str, field_name: str) -> bool:
        if not response:
            return False

        escaped_field = re.escape(field_name)
        pattern = rf"{escaped_field}\s*:\s*\$\$\$(True|False)\$\$\$"
        match = re.search(pattern, response, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"

        fallback_match = re.search(
            rf"{escaped_field}.*?(True|False)",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not fallback_match:
            return False

        return fallback_match.group(1).lower() == "true"

    def _extract_block(self, response: str, field_name: str) -> str:
        if not response:
            return ""

        escaped_field = re.escape(field_name)
        block_pattern = rf"{escaped_field}\s*:\s*<<<\s*(.*?)\s*>>>"
        matches = re.findall(block_pattern, response, flags=re.IGNORECASE | re.DOTALL)
        if not matches:
            return ""

        return matches[-1].strip()

    def _normalize_signal(self, text: str) -> str:
        clean_text = (text or "").strip()
        if not clean_text:
            return ""
        lowered = clean_text.lower()
        if lowered in {"none", "null", "n/a"}:
            return ""
        return clean_text

    def build_logic_retrieval_sketch(self, query: str) -> str:
        if not query or not str(query).strip():
            return ""

        prompt = render_prompt(
            "router_pre_reason",
            self.prompt_language,
            query=query,
        )
        response = self.api_client.complete(prompt)
        bool_fields = ["Build experience query", "Need method sketch"]
        block_fields = ["Experience-query sketch", "Method sketch"]

        if not any(self._extract_bool(response, field_name) for field_name in bool_fields):
            return ""

        for block_field in block_fields:
            block = self._normalize_signal(self._extract_block(response, block_field))
            if block:
                return block
        return ""

    def pre_reason(self, query: str):
        return self.build_logic_retrieval_sketch(query)

    def extract_relevant_explicit_memory(self, query, explicit_memories) -> str:
        prompt = render_prompt(
            "router_filter_explicit_memory",
            self.prompt_language,
            query=query,
            explicit_memories=explicit_memories or "NONE",
        )
        response = self.api_client.complete(prompt)
        field_name = "Directly usable text records"
        return self._normalize_signal(self._extract_block(response, field_name))

    def confirm_implicit_activate(
        self,
        query,
        query_logic,
        logic_trajectory_evidence=None,
    ) -> bool:
        prompt = render_prompt(
            "router_confirm_implicit_activate",
            self.prompt_language,
            query=query,
            query_logic=query_logic,
            logic_trajectory_evidence=logic_trajectory_evidence or "NONE",
        )
        response = self.api_client.complete(prompt)
        field_names = ["Activate experience", "Use past experience"]
        return any(self._extract_bool(response, field_name) for field_name in field_names)

    def build_prompt(self, query: str, relevant_explicit_memory: str) -> str:
        if relevant_explicit_memory:
            return f"User's query: {query}\nRelevant factual memories:\n{relevant_explicit_memory}"
        return query

    def route(
        self,
        query,
        ex_mem: ExplicitMemoryPlus,
        im_mem: ImplicitMemory,
        user_id=None,
        agent_id=None,
        run_id=None,
    ) -> dict:
        explicit_memory_candidates = ""
        if ex_mem is not None:
            explicit_memory_candidates = ex_mem.get_mem(
                query=query,
                limit=5,
                type="str",
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
            )

        relevant_explicit_memory = self.extract_relevant_explicit_memory(
            query=query,
            explicit_memories=explicit_memory_candidates,
        )

        logic_retrieval_sketch = self.build_logic_retrieval_sketch(query=query)
        if logic_retrieval_sketch and im_mem is not None:
            logic_trajectory_evidence = im_mem.retrieve_logic_trajectory_evidence(
                query=logic_retrieval_sketch,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
            )
        else:
            logic_trajectory_evidence = []

        if logic_retrieval_sketch:
            implicit_activation = self.confirm_implicit_activate(
                query=query,
                query_logic=logic_retrieval_sketch,
                logic_trajectory_evidence=logic_trajectory_evidence,
            )
        else:
            implicit_activation = False

        explicit_activation = bool(relevant_explicit_memory)
        prompt = self.build_prompt(
            query=query,
            relevant_explicit_memory=relevant_explicit_memory,
        )

        return {
            "explicit_activation": explicit_activation,
            "implicit_activation": implicit_activation,
            "prompt": prompt,
            "explicit_memory_candidates": explicit_memory_candidates,
            "relevant_explicit_memory": relevant_explicit_memory,
            "logic_trajectory_evidence": logic_trajectory_evidence,
            "logic_retrieval_sketch": logic_retrieval_sketch,
        }
