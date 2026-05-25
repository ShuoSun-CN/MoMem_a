import asyncio
from typing import Any, Dict, List, Optional, Union
import os
os.environ.setdefault("MEM0_TELEMETRY", "False")
from mem0 import Memory

try:
    from .config import DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG
except ImportError:
    from config import DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG


class ExplicitMemoryPlus:
    """Two-bank explicit memory wrapper built on the mem0 OSS SDK."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        _config = config if config is not None else DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG

        base_model_config = _config.get("model_config")
        content_model_config = _config.get("content_model_config") or base_model_config
        dialogue_model_config = _config.get("dialogue_model_config") or base_model_config
        if content_model_config is None or dialogue_model_config is None:
            raise ValueError(
                "ExplicitMemoryPlus requires model_config, or both "
                "content_model_config and dialogue_model_config."
            )

        self.mem_content = Memory.from_config(content_model_config)
        self.mem_dialogue = Memory.from_config(dialogue_model_config)
        
        self.default_scope = dict(_config.get("scope_config", {}))

        self._closed = False

    @staticmethod
    def _build_scope_kwargs(
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        scope_kwargs: Dict[str, Any] = {}
        if user_id is not None:
            scope_kwargs["user_id"] = user_id
        if agent_id is not None:
            scope_kwargs["agent_id"] = agent_id
        if run_id is not None:
            scope_kwargs["run_id"] = run_id
        return scope_kwargs

    @staticmethod
    def _extract_results(response: Union[Dict[str, Any], List[Dict[str, Any]], None]) -> List[Dict[str, Any]]:
        """Extract a results list from common mem0 response shapes."""
        if response is None:
            return []
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    def search_content(
        self,
        query: str,
        limit: int = 3,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        rerank: bool = True,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """Search content-memory entries relevant to `query`."""
        
        scope_kwargs = self._build_scope_kwargs(user_id=user_id, agent_id=agent_id, run_id=run_id)

        return self.mem_content.search(
            query=query,
            limit=limit,
            filters=filters,
            threshold=threshold,
            rerank=rerank,
            **scope_kwargs,
        )
        
    def search_dialogue(
            self,
            query: str,
            limit: int = 3,
            filters: Optional[Dict[str, Any]] = None,
            threshold: Optional[float] = None,
            rerank: bool = True,
            user_id: Optional[str] = None,
            agent_id: Optional[str] = None,
            run_id: Optional[str] = None,
        ):
            """Search dialogue-memory entries relevant to `query`."""
            
            scope_kwargs = self._build_scope_kwargs(user_id=user_id, agent_id=agent_id, run_id=run_id)

            return self.mem_dialogue.search(
                query=query,
                limit=limit,
                filters=filters,
                threshold=threshold,
                rerank=rerank,
                **scope_kwargs,
            )
        
    def get_mem(
        self,
        query: str,
        limit: int = 3,
        threshold: Optional[float] = None,
        rerank: bool = True,
        type: str = "str",
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """Return combined content/dialogue results as prompt text or raw entries."""
        
        relevant_dialogues = self.search_dialogue(
            query=query,
            limit=limit,
            threshold=threshold,
            rerank=rerank,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        relevant_contents = self.search_content(
            query=query,
            limit=limit,
            threshold=threshold,
            rerank=rerank,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        contents_list = self._extract_results(relevant_contents)
        dialogue_list=self._extract_results(relevant_dialogues)
        
        memories_list=contents_list+dialogue_list
        
        if type == "str":
            return "\n".join(f"- {entry.get('memory', '')}" for entry in memories_list)

        return memories_list

    def get_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """List all memories in the optional scope."""
        scope_kwargs = self._build_scope_kwargs(user_id=user_id, agent_id=agent_id, run_id=run_id)
        print(f"Fetching all memories for scope={scope_kwargs or 'default'}...")

        content_memories = self.mem_content.get_all(limit=limit, filters=filters, **scope_kwargs)
        dialogue_memories = self.mem_dialogue.get_all(limit=limit, filters=filters, **scope_kwargs)
        content_res = content_memories.get("results", []) if isinstance(content_memories, dict) else content_memories
        dialogue_res = dialogue_memories.get("results", []) if isinstance(dialogue_memories, dict) else dialogue_memories
        res = list(content_res or []) + list(dialogue_res or [])
        print(f"Fetched {len(res)} memories.")
        return res

    def add(
        self,
        messages: Union[str, List[Dict[str, str]]],
        metadata: Optional[Dict[str, Any]] = None,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """Add a memory to both content and dialogue banks."""
        
        scope_kwargs = self._build_scope_kwargs(user_id=user_id, agent_id=agent_id, run_id=run_id)

        content_result = self.mem_content.add(
            messages,
            metadata=metadata,
            infer=True,
            memory_type=memory_type,
            prompt=prompt,
            **scope_kwargs,
        )
        
        dialogue_result = self.mem_dialogue.add(
            messages,
            metadata=metadata,
            infer=False,
            memory_type=memory_type,
            prompt=prompt,
            **scope_kwargs,
        )
        return {"content": content_result, "dialogue": dialogue_result}

    def delete(self, memory_id: str):
        """Delete one content-bank memory by id."""
        return self.mem_content.delete(memory_id=memory_id)

    def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """Delete all content and dialogue memories in a specified scope."""
        scope_kwargs = self._build_scope_kwargs(user_id=user_id, agent_id=agent_id, run_id=run_id)
        if not scope_kwargs:
            raise ValueError("delete_all requires at least one scope: user_id, agent_id, or run_id.")

        content_result = self.mem_content.delete_all(**scope_kwargs)
        dialogue_result = self.mem_dialogue.delete_all(**scope_kwargs)
        return {"content": content_result, "dialogue": dialogue_result}

    def close(self, verbose: bool = False):
        """Close the underlying vector-store clients when available."""
        if self._closed:
            return

        if hasattr(self.mem_content, "vector_store") and hasattr(self.mem_content.vector_store, "client"):
            try:
                client = self.mem_content.vector_store.client
                if client is not None:
                    client.close()
                self.mem_content.vector_store.client = None
                if verbose:
                    print("Database client closed.")
            except Exception as e:
                print(f"Unexpected error while closing database client: {e}")
        if hasattr(self.mem_dialogue, "vector_store") and hasattr(self.mem_dialogue.vector_store, "client"):
            try:
                client = self.mem_dialogue.vector_store.client
                if client is not None:
                    client.close()
                self.mem_dialogue.vector_store.client = None
                if verbose:
                    print("Database client closed.")
            except Exception as e:
                print(f"Unexpected error while closing database client: {e}")
        self._closed = True

    def update(self, memory_id: str, data: str) -> Dict[str, Any]:
        """Update one content-bank memory by id."""
        return self.mem_content.update(memory_id=memory_id, data=data)
