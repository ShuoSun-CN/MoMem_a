from pathlib import Path
import os

home_path = os.getenv("MOMEM_EXPLICIT_MEMORY_PATH", str(Path.home() / ".cache" / "momem" / "explicit_memory"))
DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG = {
    "model_config": {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-5.4",
                "temperature": 0.2,
                "max_tokens": 2000,
                "top_p": 1.0
            }
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "qwen3-embedding:0.6b",

            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "qwen3_embedding_memory",
                "path": home_path,
                "embedding_model_dims": 1024,
                "on_disk": True
            }
        }
    },
}
