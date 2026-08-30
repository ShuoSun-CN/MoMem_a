import argparse
import copy
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def str2bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def parse_args() -> argparse.Namespace:
    default_output_dir = Path(__file__).resolve().parent / "results"
    default_output = default_output_dir / f"longmemevo_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    parser = argparse.ArgumentParser(
        description="LongMemEvo evaluation with explicit-memory ingestion, retrieval, and answering."
    )
    parser.add_argument("--dataset-path", type=str, required=True, help="LongMemEvo data path")
    parser.add_argument("--question-type", type=str, default=None, help="Filter by question_type.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index after filtering.")
    parser.add_argument("--max-samples", type=int, default=5, help="Maximum number of samples to evaluate.")
    parser.add_argument("--prepare-only", action="store_true", help="Preview samples without memory or model calls.")

    parser.add_argument("--collection-name", type=str, default="longmemevo_explicit_memory", help="Qdrant collection name.")
    parser.add_argument("--user-id", type=str, default="longmemevo_user", help="mem0 user_id")
    parser.add_argument("--agent-id", type=str, default=None, help="mem0 agent_id")
    parser.add_argument("--run-prefix", type=str, default="longmemevo_run", help="run_id prefix for each sample.")
    parser.add_argument("--search-limit", type=int, default=5, help="Number of memories retrieved per question.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional retrieval threshold.")
    parser.add_argument("--rerank", type=str2bool, default=True, help="Whether to enable reranking.")
    parser.add_argument("--infer", type=str2bool, default=True, help="Whether to use mem0 inference when adding memories.")
    parser.add_argument("--memory-type", type=str, default=None, help="mem0 memory_type")
    parser.add_argument(
        "--cleanup-after-sample",
        type=str2bool,
        default=True,
        help="Whether to clean the run_id memory scope after each sample.",
    )

    parser.add_argument(
        "--agent-backend",
        type=str,
        default=os.getenv("API_CLIENT_BACKEND", "local"),
        choices=["local", "vllm", "local_vllm", "remote"],
        help="AgentClient backend.",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default=None,
        help="Local vLLM model name; ignored by the remote OpenAI path.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant. Use the provided memory context to answer briefly and accurately.",
        help="System prompt used for answering.",
    )
    parser.add_argument("--output", type=str, default=str(default_output), help="Output JSON file.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N samples.")
    return parser.parse_args()


def normalize_turns(session_turns: Any) -> List[Dict[str, str]]:
    if not isinstance(session_turns, list):
        return []
    normalized: List[Dict[str, str]] = []
    for turn in session_turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "user")).strip().lower()
        if role not in {"user", "assistant", "system"}:
            role = "user"
        content = turn.get("content", "")
        if content is None:
            continue
        content = str(content).strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def normalize_text(text: Any) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def text_hit(pred: Any, gold: Any) -> bool:
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    if not pred_norm or not gold_norm:
        return False
    return gold_norm in pred_norm


def extract_results(search_response: Any) -> List[Dict[str, Any]]:
    if search_response is None:
        return []
    if isinstance(search_response, dict):
        res = search_response.get("results", [])
        return res if isinstance(res, list) else []
    if isinstance(search_response, list):
        return search_response
    return []


def build_agent_prompt(question: str, question_date: str, memories: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, mem in enumerate(memories, start=1):
        metadata = mem.get("metadata", {}) if isinstance(mem, dict) else {}
        session_id = metadata.get("session_id", "unknown_session")
        session_date = metadata.get("session_date", "unknown_date")
        memory_text = str(mem.get("memory", "")).strip()
        if not memory_text:
            continue
        lines.append(f"[{idx}] ({session_date}, {session_id}) {memory_text}")

    memory_block = "\n".join(lines) if lines else "(No relevant memory found)"
    return (
        f"Current question date: {question_date}\n"
        f"Relevant memories:\n{memory_block}\n\n"
        f"Question: {question}\n\n"
        "Please answer the question only using the memory above. "
        "If memory is insufficient, answer: I don't know."
    )


def build_memory_system(args: argparse.Namespace):
    from MoMem.config import DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG
    try:
        from MoMem.ExplicitMemory import ExplicitMemoryMem0
    except ModuleNotFoundError as e:
        if str(e) == "No module named 'mem0'" or "mem0" in str(e):
            raise RuntimeError(
                "mem0 is not installed. Install mem0 and the required embedding/vector-store "
                "dependencies before running without --prepare-only."
            ) from e
        raise

    config = copy.deepcopy(DEFAULT_EXPLICIT_MEMORY_MEM0_CONFIG)
    config["model_config"]["vector_store"]["config"]["collection_name"] = args.collection_name
    return ExplicitMemoryMem0(config=config)


def build_sample_scope(
    args: argparse.Namespace,
    sample: Dict[str, Any],
    original_idx: int,
    question_id: str,
) -> Dict[str, Optional[str]]:
    sample_user_id = str(sample.get("user_id") or f"{args.user_id}_{original_idx}_{question_id}")
    run_id = f"{args.run_prefix}_{original_idx}_{question_id}" if args.run_prefix else None
    return {
        "user_id": sample_user_id,
        "agent_id": args.agent_id,
        "run_id": run_id,
    }


def build_agent(args: argparse.Namespace):
    from MoMem.AgentClient import AgentClient

    backend = "local_vllm" if args.agent_backend in {"local", "vllm", "local_vllm"} else "remote"
    model_name = args.agent_model if backend == "local_vllm" else None
    return AgentClient(backend=backend, model_name=model_name)


def safe_float_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def select_samples(dataset: List[Dict[str, Any]], args: argparse.Namespace) -> List[Tuple[int, Dict[str, Any]]]:
    indexed = list(enumerate(dataset))
    if args.question_type:
        indexed = [(i, s) for i, s in indexed if s.get("question_type") == args.question_type]
    if args.start_index > 0:
        indexed = indexed[args.start_index :]
    if args.max_samples is not None and args.max_samples >= 0:
        indexed = indexed[: args.max_samples]
    return indexed


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("Dataset top-level must be a list.")

    selected = select_samples(dataset, args)
    preview = [
        {
            "original_index": i,
            "question_id": s.get("question_id"),
            "question_type": s.get("question_type"),
            "question": s.get("question"),
            "answer": s.get("answer"),
            "haystack_session_count": len(s.get("haystack_sessions", [])),
        }
        for i, s in selected[:5]
    ]

    if args.prepare_only:
        prepare_payload = {
            "prepare_only": True,
            "dataset_name": "LongMemEvo",
            "dataset_path": str(dataset_path),
            "selected_count": len(selected),
            "args": vars(args),
            "preview": preview,
        }
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(prepare_payload, f, ensure_ascii=False, indent=2)
        print(f"[PrepareOnly] selected={len(selected)}, output={output_path}")
        return

    memory = build_memory_system(args)
    agent = build_agent(args)

    results: List[Dict[str, Any]] = []
    type_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"count": 0, "answer_text_hit_count": 0, "answer_session_hit_count": 0}
    )
    summary = {
        "requested_samples": len(selected),
        "evaluated_samples": 0,
        "skipped_invalid": 0,
        "add_error_count": 0,
        "add_fallback_session_count": 0,
        "add_fallback_turn_success": 0,
        "add_fallback_turn_failed": 0,
        "search_error_count": 0,
        "answer_text_hit_count": 0,
        "answer_session_hit_count": 0,
    }

    try:
        for loop_idx, (original_idx, sample) in enumerate(selected, start=1):
            question = sample.get("question", "")
            haystack_sessions = sample.get("haystack_sessions", [])
            haystack_dates = sample.get("haystack_dates", [])
            haystack_session_ids = sample.get("haystack_session_ids", [])
            answer_session_ids = sample.get("answer_session_ids", [])
            question_id = sample.get("question_id", f"sample_{original_idx}")
            question_type = sample.get("question_type", "unknown")
            question_date = sample.get("question_date", "")
            gold_answer = sample.get("answer", "")
            scope_kwargs = build_sample_scope(args, sample, original_idx, question_id)
            sample_user_id = scope_kwargs["user_id"]
            run_id = scope_kwargs["run_id"]

            if not isinstance(haystack_sessions, list) or not isinstance(question, str):
                summary["skipped_invalid"] += 1
                continue

            if args.cleanup_after_sample:
                try:
                    memory.delete_all(**scope_kwargs)
                except Exception:
                    pass

            ingested_session_count = 0
            for session_idx, session_turns in enumerate(haystack_sessions):
                normalized_turns = normalize_turns(session_turns)
                if not normalized_turns:
                    continue

                session_date = haystack_dates[session_idx] if session_idx < len(haystack_dates) else None
                session_id = haystack_session_ids[session_idx] if session_idx < len(haystack_session_ids) else None
                metadata = {
                    "dataset": "LongMemEvo",
                    "question_id": question_id,
                    "question_type": question_type,
                    "session_index": session_idx,
                    "session_date": session_date,
                    "session_id": session_id,
                }

                try:
                    memory.add(
                        messages=normalized_turns,
                        metadata=metadata,
                        infer=args.infer,
                        memory_type=args.memory_type,
                        **scope_kwargs,
                    )
                    ingested_session_count += 1
                except Exception:
                    summary["add_error_count"] += 1
                    summary["add_fallback_session_count"] += 1
                    for turn in normalized_turns:
                        try:
                            memory.add(
                                messages=[turn],
                                metadata=metadata,
                                infer=args.infer,
                                memory_type=args.memory_type,
                                **scope_kwargs,
                            )
                            summary["add_fallback_turn_success"] += 1
                        except Exception:
                            summary["add_fallback_turn_failed"] += 1

            raw_search_response: Any = None
            retrieved_memories: List[Dict[str, Any]] = []
            try:
                raw_search_response = memory.search(
                    query=question,
                    limit=args.search_limit,
                    threshold=args.threshold,
                    rerank=args.rerank,
                    **scope_kwargs,
                )
                retrieved_memories = extract_results(raw_search_response)
            except Exception as e:
                summary["search_error_count"] += 1
                raw_search_response = {"error": str(e)}

            prompt = build_agent_prompt(question=question, question_date=question_date, memories=retrieved_memories)
            predicted_answer = agent.complete(prompt=prompt, system_prompt=args.system_prompt)
            predicted_answer = predicted_answer if isinstance(predicted_answer, str) else ""

            retrieved_session_ids = []
            for mem in retrieved_memories:
                if not isinstance(mem, dict):
                    continue
                session_id = (mem.get("metadata") or {}).get("session_id")
                if session_id is not None:
                    retrieved_session_ids.append(str(session_id))

            answer_session_set = {str(sid) for sid in answer_session_ids if sid is not None}
            retrieved_session_set = set(retrieved_session_ids)
            answer_session_hit = len(answer_session_set.intersection(retrieved_session_set)) > 0
            answer_text_hit = text_hit(pred=predicted_answer, gold=gold_answer)

            summary["evaluated_samples"] += 1
            summary["answer_text_hit_count"] += int(answer_text_hit)
            summary["answer_session_hit_count"] += int(answer_session_hit)

            type_stats[question_type]["count"] += 1
            type_stats[question_type]["answer_text_hit_count"] += int(answer_text_hit)
            type_stats[question_type]["answer_session_hit_count"] += int(answer_session_hit)

            result_item = {
                "original_index": original_idx,
                "question_id": question_id,
                "question_type": question_type,
                "question_date": question_date,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "answer_session_ids": answer_session_ids,
                "user_id": sample_user_id,
                "agent_id": args.agent_id,
                "run_id": run_id,
                "ingested_session_count": ingested_session_count,
                "retrieved_count": len(retrieved_memories),
                "retrieved_session_ids": retrieved_session_ids,
                "answer_text_hit": answer_text_hit,
                "answer_session_hit": answer_session_hit,
                "retrieved_memories_text": "\n".join(
                    f"- {str(mem.get('memory', ''))}" for mem in retrieved_memories if isinstance(mem, dict)
                ),
                "retrieved_memories": retrieved_memories,
                "raw_search_response": raw_search_response,
            }
            results.append(sanitize_for_json(result_item))

            if args.cleanup_after_sample:
                try:
                    memory.delete_all(**scope_kwargs)
                except Exception:
                    pass

            if args.progress_every > 0 and (loop_idx % args.progress_every == 0 or loop_idx == len(selected)):
                print(f"[Progress] {loop_idx}/{len(selected)} done.")
    finally:
        try:
            memory.close()
        except Exception:
            pass

    evaluated = summary["evaluated_samples"]
    summary["answer_text_hit_rate"] = safe_float_ratio(summary["answer_text_hit_count"], evaluated)
    summary["answer_session_hit_rate"] = safe_float_ratio(summary["answer_session_hit_count"], evaluated)

    by_question_type: Dict[str, Dict[str, Any]] = {}
    for q_type, stats in type_stats.items():
        count = stats["count"]
        by_question_type[q_type] = {
            "count": count,
            "answer_text_hit_count": stats["answer_text_hit_count"],
            "answer_text_hit_rate": safe_float_ratio(stats["answer_text_hit_count"], count),
            "answer_session_hit_count": stats["answer_session_hit_count"],
            "answer_session_hit_rate": safe_float_ratio(stats["answer_session_hit_count"], count),
        }

    output_payload = {
        "dataset_name": "LongMemEvo",
        "dataset_path": str(dataset_path),
        "script": str(Path(__file__).resolve()),
        "run_time": datetime.now().isoformat(),
        "args": vars(args),
        "summary": summary,
        "by_question_type": by_question_type,
        "results": results,
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"[Done] evaluated={evaluated}, output={output_path}")


if __name__ == "__main__":
    main()
