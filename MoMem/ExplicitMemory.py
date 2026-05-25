import asyncio
from typing import Any, Dict, List, Optional, Union
import os
os.environ.setdefault("MEM0_TELEMETRY", "False")
from mem0 import Memory

try:
    from .config import DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG
except ImportError:
    from config import DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG


class ExplicitMemoryMem0:
    """Thin wrapper around the mem0 OSS SDK explicit-memory store."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        _config = config if config is not None else DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG

        self.mem = Memory.from_config(_config['model_config'])
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

    def search(
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
        """Search memories relevant to `query` within the optional scope."""
        
        scope_kwargs = self._build_scope_kwargs(user_id=user_id, agent_id=agent_id, run_id=run_id)

        return self.mem.search(
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
        """Return search results as prompt text or raw memory entries."""
        
        relevant_memories = self.search(
            query=query,
            limit=limit,
            threshold=threshold,
            rerank=rerank,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        memories_list = self._extract_results(relevant_memories)

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

        all_memories = self.mem.get_all(limit=limit, filters=filters, **scope_kwargs)
        res = all_memories.get("results", []) if isinstance(all_memories, dict) else all_memories
        print(f"Fetched {len(res)} memories.")
        return res

    def add(
        self,
        messages: Union[str, List[Dict[str, str]]],
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """Add a memory from raw text or a chat-message list."""
        
        scope_kwargs = self._build_scope_kwargs(user_id=user_id, agent_id=agent_id, run_id=run_id)

        return self.mem.add(
            messages,
            metadata=metadata,
            infer=infer,
            memory_type=memory_type,
            prompt=prompt,
            **scope_kwargs,
        )

    async def add_many_async(
        self,
        items: List[Union[str, Dict[str, str], List[Dict[str, str]]]],
        metadata_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        concurrency: int = 8,
        return_exceptions: bool = True,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> List[Any]:
        """Add multiple memories concurrently through the synchronous mem0 API."""
        if concurrency <= 0:
            raise ValueError("concurrency must be greater than 0.")
        if metadata_list is not None and len(metadata_list) != len(items):
            raise ValueError("metadata_list must have the same length as items.")

        sem = asyncio.Semaphore(concurrency)

        async def _worker(
            item: Union[str, Dict[str, str], List[Dict[str, str]]],
            metadata: Optional[Dict[str, Any]],
        ):
            async with sem:
                return await asyncio.to_thread(
                    self.add,
                    messages=item,
                    metadata=metadata,
                    infer=infer,
                    memory_type=memory_type,
                    prompt=prompt,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                )

        tasks = []
        for i, item in enumerate(items):
            item_metadata = metadata_list[i] if metadata_list is not None else None
            tasks.append(_worker(item, item_metadata))

        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    def delete(self, memory_id: str):
        """Delete one memory by id."""
        return self.mem.delete(memory_id=memory_id)

    def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """Delete all memories in a specified scope."""
        scope_kwargs = self._build_scope_kwargs(user_id=user_id, agent_id=agent_id, run_id=run_id)
        if not scope_kwargs:
            raise ValueError("delete_all requires at least one scope: user_id, agent_id, or run_id.")

        return self.mem.delete_all(**scope_kwargs)

    def close(self, verbose: bool = False):
        """Close the underlying vector-store client when available."""
        if self._closed:
            return

        if hasattr(self.mem, "vector_store") and hasattr(self.mem.vector_store, "client"):
            try:
                client = self.mem.vector_store.client
                if client is not None:
                    client.close()
                self.mem.vector_store.client = None
                if verbose:
                    print("Database client closed.")
            except Exception as e:
                print(f"Unexpected error while closing database client: {e}")
        self._closed = True

    def update(self, memory_id: str, data: str) -> Dict[str, Any]:
        """Update one memory by id."""
        return self.mem.update(memory_id=memory_id, data=data)
