#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

OPTION_LETTERS = ("A", "B", "C", "D")


def render_mcq_prompt(question: str, options: dict[str, str]) -> str:
    lines = [f"Question: {question}", "", "Options:"]
    for key in OPTION_LETTERS:
        lines.append(f"{key}. {options[key]}")
    lines.extend(("", "Answer:"))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build original medeval parquet validation sets.")
    parser.add_argument("--test-data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def medqa_row(item: dict) -> dict:
    return {
        "prompt": [{"role": "user", "content": render_mcq_prompt(item["question"], item["options"])}],
        "reward_model": {"ground_truth": str(item["answer_idx"]).strip().upper()},
        "extra_info": {"dataset": "medqa"},
        "data_source": "medqa",
    }


def mmlu_row(item: dict) -> dict:
    options = {chr(65 + i): choice for i, choice in enumerate(item["choices"])}
    return {
        "prompt": [{"role": "user", "content": render_mcq_prompt(item["question"], options)}],
        "reward_model": {"ground_truth": chr(65 + int(item["answer"]))},
        "extra_info": {"dataset": "mmlu_medical"},
        "data_source": "mmlu_medical",
    }


def medmcqa_row(item: dict) -> dict:
    options = {"A": item["opa"], "B": item["opb"], "C": item["opc"], "D": item["opd"]}
    answer = chr(65 + int(item["cop"])) if int(item["cop"]) != -1 else "A"
    return {
        "prompt": [{"role": "user", "content": render_mcq_prompt(item["question"], options)}],
        "reward_model": {"ground_truth": answer},
        "extra_info": {"dataset": "medmcqa"},
        "data_source": "medmcqa",
    }


def write_dataset(src: Path, out: Path, mapper) -> None:
    rows = [mapper(item) for item in load_jsonl(src)]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {len(rows)} rows to {out}")


def main() -> None:
    args = parse_args()
    test_data_dir = Path(args.test_data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    write_dataset(test_data_dir / "medqa_test.jsonl", out_dir / "medqa_val.parquet", medqa_row)
    write_dataset(test_data_dir / "mmlu_medical_test.jsonl", out_dir / "mmlu_medical_val.parquet", mmlu_row)
    write_dataset(test_data_dir / "medmcqa_test.jsonl", out_dir / "medmcqa_val.parquet", medmcqa_row)


if __name__ == "__main__":
    main()
