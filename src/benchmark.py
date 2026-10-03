from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversation benchmark datasets from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return recall score based on how many expected facts appear in answer.

    - 0 if none of the expected facts appear
    - 0.5 if some appear (generalized as matching / total)
    - 1.0 if all appear
    Uses casefold() for robust case-insensitive matching.
    """
    if not expected:
        return 1.0

    ans_norm = answer.casefold()
    matched = sum(1 for item in expected if item.casefold() in ans_norm)

    if matched == 0:
        return 0.0
    if matched == len(expected):
        return 1.0
    return round(matched / len(expected), 2)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score evaluating recall accuracy and response structure."""
    recall = recall_points(answer, expected)
    cleaned = answer.strip()
    if not cleaned:
        return 0.0

    length = len(cleaned)
    length_score = 1.0 if 30 <= length <= 600 else 0.7 if length > 0 else 0.0
    quality = 0.75 * recall + 0.25 * length_score
    return round(quality, 2)


def run_agent_benchmark(
    agent_name: str,
    agent: BaselineAgent | AdvancedAgent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over conversations and cross-session recall questions."""
    all_recall_scores: list[float] = []
    all_quality_scores: list[float] = []
    all_user_ids: set[str] = set()

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        all_user_ids.add(user_id)
        turns: list[str] = conv.get("turns", [])
        recall_questions: list[dict[str, Any]] = conv.get("recall_questions", [])

        # Feed all conversation turns in the conversation's own thread
        for turn_text in turns:
            agent.reply(user_id=user_id, thread_id=conv_id, message=turn_text)

        # Cross-session evaluation: ask each recall question in a distinct fresh thread
        for idx, rq in enumerate(recall_questions):
            question = rq["question"]
            expected = rq.get("expected_contains", [])
            fresh_thread_id = f"recall-{conv_id}-{idx}"

            res = agent.reply(user_id=user_id, thread_id=fresh_thread_id, message=question)
            ans = res.get("reply", "")

            r_score = recall_points(ans, expected)
            q_score = heuristic_quality(ans, expected)
            all_recall_scores.append(r_score)
            all_quality_scores.append(q_score)

    avg_recall = (
        round(sum(all_recall_scores) / len(all_recall_scores), 2)
        if all_recall_scores
        else 0.0
    )
    avg_quality = (
        round(sum(all_quality_scores) / len(all_quality_scores), 2)
        if all_quality_scores
        else 0.0
    )

    agent_tokens = agent.token_usage()
    prompt_tokens = agent.prompt_token_usage()

    if isinstance(agent, AdvancedAgent):
        growth = sum(agent.memory_file_size(u) for u in all_user_ids)
        compactions = agent.compaction_count()
    else:
        growth = 0
        compactions = 0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a clean table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = [
        [
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            f"{r.memory_growth_bytes:,}",
            f"{r.compactions}",
        ]
        for r in rows
    ]

    try:
        from tabulate import tabulate

        return tabulate(table_data, headers=headers, tablefmt="github")
    except ImportError:
        # Fallback markdown table generator
        col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *table_data)]
        header_str = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
        sep_str = "| " + " | ".join("-" * w for w in col_widths) + " |"
        row_strs = [
            "| " + " | ".join(val.ljust(w) for val, w in zip(row, col_widths)) + " |"
            for row in table_data
        ]
        return "\n".join([header_str, sep_str] + row_strs)


def main() -> None:
    """Run standard benchmark and long-context stress benchmark."""
    config = load_config(Path(__file__).resolve().parent.parent)

    std_data_path = config.data_dir / "conversations.json"
    stress_data_path = config.data_dir / "advanced_long_context.json"

    std_conversations = load_conversations(std_data_path)
    stress_conversations = load_conversations(stress_data_path)

    # 1. Standard Benchmark
    bench_dir_std = config.state_dir / "benchmarks" / "standard"
    if bench_dir_std.exists():
        shutil.rmtree(bench_dir_std)
    bench_dir_std.mkdir(parents=True, exist_ok=True)

    cfg_std_base = LabConfig(
        base_dir=config.base_dir,
        data_dir=config.data_dir,
        state_dir=bench_dir_std / "baseline",
        compact_threshold_tokens=config.compact_threshold_tokens,
        compact_keep_messages=config.compact_keep_messages,
        model=config.model,
        judge_model=config.judge_model,
    )
    cfg_std_adv = LabConfig(
        base_dir=config.base_dir,
        data_dir=config.data_dir,
        state_dir=bench_dir_std / "advanced",
        compact_threshold_tokens=config.compact_threshold_tokens,
        compact_keep_messages=config.compact_keep_messages,
        model=config.model,
        judge_model=config.judge_model,
    )

    baseline_std = BaselineAgent(config=cfg_std_base, force_offline=True)
    advanced_std = AdvancedAgent(config=cfg_std_adv, force_offline=True)

    row_std_base = run_agent_benchmark("Baseline", baseline_std, std_conversations, cfg_std_base)
    row_std_adv = run_agent_benchmark("Advanced", advanced_std, std_conversations, cfg_std_adv)

    print("=== Standard Benchmark ===")
    print(format_rows([row_std_base, row_std_adv]))
    print()

    # 2. Long-Context Stress Benchmark
    bench_dir_stress = config.state_dir / "benchmarks" / "stress"
    if bench_dir_stress.exists():
        shutil.rmtree(bench_dir_stress)
    bench_dir_stress.mkdir(parents=True, exist_ok=True)

    cfg_stress_base = LabConfig(
        base_dir=config.base_dir,
        data_dir=config.data_dir,
        state_dir=bench_dir_stress / "baseline",
        compact_threshold_tokens=config.compact_threshold_tokens,
        compact_keep_messages=config.compact_keep_messages,
        model=config.model,
        judge_model=config.judge_model,
    )
    cfg_stress_adv = LabConfig(
        base_dir=config.base_dir,
        data_dir=config.data_dir,
        state_dir=bench_dir_stress / "advanced",
        compact_threshold_tokens=config.compact_threshold_tokens,
        compact_keep_messages=config.compact_keep_messages,
        model=config.model,
        judge_model=config.judge_model,
    )

    baseline_stress = BaselineAgent(config=cfg_stress_base, force_offline=True)
    advanced_stress = AdvancedAgent(config=cfg_stress_adv, force_offline=True)

    row_stress_base = run_agent_benchmark(
        "Baseline", baseline_stress, stress_conversations, cfg_stress_base
    )
    row_stress_adv = run_agent_benchmark(
        "Advanced", advanced_stress, stress_conversations, cfg_stress_adv
    )

    print("=== Long-Context Stress Benchmark ===")
    print(format_rows([row_stress_base, row_stress_adv]))
    print()


if __name__ == "__main__":
    main()
