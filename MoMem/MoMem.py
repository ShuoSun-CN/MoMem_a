import os
from pathlib import Path
from copy import deepcopy

try:
    from .AgentClient import AgentClient
    from .ExplicitMemoryPlus import ExplicitMemoryPlus
    from .ImplicitMemory import ImplicitMemory
    from .MemoryDecoupler import MemoryDecoupler
    from .MemoryRouter import MemoryRouter
    from .config import DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG
    from .prompts import DEFAULT_SYSTEM_PROMPT
except ImportError:
    from AgentClient import AgentClient
    from ExplicitMemoryPlus import ExplicitMemoryPlus
    from ImplicitMemory import ImplicitMemory
    from MemoryDecoupler import MemoryDecoupler
    from MemoryRouter import MemoryRouter
    from config import DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG
    from prompts import DEFAULT_SYSTEM_PROMPT


def _build_explicit_memory_config():
    config = deepcopy(DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG)
    base_path = Path(os.getenv("MOMEM_MEMORY_DIR", Path.home() / ".cache" / "momem"))

    content_model_config = deepcopy(config["model_config"])
    content_model_config["vector_store"]["config"]["path"] = str(base_path / "content")
    content_model_config["vector_store"]["config"]["collection_name"] = "qwen3_embedding_memory_content"

    dialogue_model_config = deepcopy(config["model_config"])
    dialogue_model_config["vector_store"]["config"]["path"] = str(base_path / "dialogue")
    dialogue_model_config["vector_store"]["config"]["collection_name"] = "qwen3_embedding_memory_dialogue"

    config["content_model_config"] = content_model_config
    config["dialogue_model_config"] = dialogue_model_config
    return config


explicitmem_config = _build_explicit_memory_config()


class MoMem:
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
        self.explicit_memory = ExplicitMemoryPlus(config=explicitmem_config)
        self.implicit_memory = ImplicitMemory(
            base_model=base_model,
            lora_path=lora_path,
            need_grpo=need_grpo,
            reward_func=reward_func,
            reward_name=reward_name,
            system_prompt=system_prompt,
            reward_mode=reward_mode,
        )
        
        self.base_model = base_model
        self.base_agent = None
        self.remote_agent = AgentClient("remote")
        self.memory_router = MemoryRouter(route_agent=self.remote_agent)
        self.memory_decoupler = MemoryDecoupler(decouple_agent=self.remote_agent)
        self._closed = False

    def _lora_path_exists(self) -> bool:
        lora_path = getattr(self.implicit_memory, "lora_path", None)
        return bool(lora_path) and os.path.exists(lora_path)

    @staticmethod
    def _first_generation_text(response):
        if isinstance(response, list):
            return response[0] if response else ""
        return response or ""

    def _get_base_agent(self):
        if self.base_agent is None:
            self.base_agent = AgentClient("local", model_name=self.base_model)
        return self.base_agent

    def _generate(self, prompt, system_prompt, use_implicit: bool):
        grpo_lora = getattr(self.implicit_memory, "grpo_lora", None)
        if grpo_lora is not None:
            if use_implicit:
                return self._first_generation_text(
                    self.implicit_memory.generate_with_implicit(
                        prompt,
                        system_prompt=system_prompt,
                    )
                )
            return self._first_generation_text(
                self.implicit_memory.generate_base(
                    prompt,
                    system_prompt=system_prompt,
                )
            )

        if use_implicit:
            if self._lora_path_exists():
                return self._get_base_agent().complete(
                    prompt,
                    use_lora=True,
                    lora_path=self.implicit_memory.lora_path,
                    system_prompt=system_prompt,
                )

        return self._get_base_agent().complete(prompt, system_prompt=system_prompt)

    def close(self):
        if self._closed:
            return

        try:
            if hasattr(self, "explicit_memory") and self.explicit_memory is not None:
                self.explicit_memory.close()
        except Exception as e:
            print(f"Failed to close explicit_memory: {e}")

        try:
            if (
                hasattr(self, "implicit_memory")
                and self.implicit_memory is not None
                and hasattr(self.implicit_memory, "logic_trajectory_bank")
                and self.implicit_memory.logic_trajectory_bank is not None
            ):
                self.implicit_memory.logic_trajectory_bank.close()
        except Exception as e:
            print(f"Failed to close implicit_memory.logic_trajectory_bank: {e}")

        self._closed = True

    def chat_with_mem(
        self,
        query,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        decouple=True,
        user_id=None,
        agent_id=None,
        run_id=None,
    ):
        routing_result = self.memory_router.route(
            query=query,
            ex_mem=self.explicit_memory,
            im_mem=self.implicit_memory,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        prompt = routing_result['prompt']
        response = self._generate(
            prompt=prompt,
            system_prompt=system_prompt,
            use_implicit=routing_result["implicit_activation"],
        )
            
            
        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
        if decouple:
            self.memory_decoupler.decouple(
                messages=messages,
                ex_mem=self.explicit_memory,
                im_mem=self.implicit_memory,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
            )

        return response

    def _chat_with_implicit_memory(
        self,
        query,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        decouple=True,
        user_id=None,
        agent_id=None,
        run_id=None,
    ):
        routing_result = self.memory_router.route(
            query=query,
            ex_mem=self.explicit_memory,
            im_mem=self.implicit_memory,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        prompt = query
        response = self._generate(
            prompt=prompt,
            system_prompt=system_prompt,
            use_implicit=routing_result["implicit_activation"],
        )
            
            
        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
        if decouple:
            self.memory_decoupler.decouple(
                    messages=messages,
                    ex_mem=self.explicit_memory,
                    im_mem=self.implicit_memory,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                )
        return response

    def _chat_with_explicit_memory(
        self,
        query,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_id=None,
        agent_id=None,
        run_id=None,
    ):
        routing_result = self.memory_router.route(
            query=query,
            ex_mem=self.explicit_memory,
            im_mem=self.implicit_memory,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        response = self._generate(
            prompt=routing_result["prompt"],
            system_prompt=system_prompt,
            use_implicit=False,
        )
        

        return response
    def memorize_only(
        self,
        batch_messages,
        concurrency: int = 8,
        show_progress: bool = True,
        progress_every: int = 200,
        ex_update=False,
        im_update=False,
        ex_mem=None,
        im_mem=None,
        user_id=None,
        agent_id=None,
        run_id=None,
        
    ):
        return self.memory_decoupler.decouple_batch_only(
            messages_batch=batch_messages,
            ex_mem=ex_mem if ex_mem is not None else self.explicit_memory,
            im_mem=im_mem if im_mem is not None else self.implicit_memory,
            ex_update=ex_update,
            im_update=im_update,
            concurrency=concurrency,
            show_progress=show_progress,
            progress_every=progress_every,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )

        
