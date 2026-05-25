# MoMem

This directory contains the anonymized code package for the EMNLP submission.
It keeps only the MoMem implementation, prompt templates, and a lightweight
LongMemEvo evaluation script. Large generated outputs, local cache files,
notebooks, and private experiment artifacts are intentionally excluded.

## Contents

- `MoMem/`: core package.
- `examples/evaluate_longmemevo.py`: parameterized LongMemEvo evaluation script.
- `requirements.txt`: runtime dependencies.
- `.env.example`: environment variables used by remote and local backends.

## Setup

```bash
pip install -r requirements.txt
```

The remote backend uses OpenAI only. No API keys or private endpoints are stored
in the code.

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5.4"
export VLLM_BASE_URL="http://127.0.0.1:8090"
export VLLM_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
```

## Example

```python
from MoMem import MoMem

agent = MoMem(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    lora_path="./outputs/implicit_memory_lora_adapter",
    need_grpo=False,
)

response = agent.chat_with_mem(
    "Answer the user request here.",
    user_id="example_user",
)
print(response)
agent.close()
```

## LongMemEvo Evaluation

```bash
python examples/evaluate_longmemevo.py \
  --dataset-path /path/to/longmemeval_s_cleaned_fixed.json \
  --agent-backend remote \
  --agent-model gpt-5.4 \
  --max-samples 100 \
  --output results/longmemevo_eval.json
```

For local vLLM inference, set `VLLM_BASE_URL` and `VLLM_MODEL_NAME`, then use
`--agent-backend local`.
