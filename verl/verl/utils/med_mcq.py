from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping


OPTION_LETTERS = ("A", "B", "C", "D")


def render_mcq_prompt(question: str, options: Iterable[str] | Dict[str, str]) -> str:
    lines = [f"Question: {question}", "", "Options:"]
    if isinstance(options, dict):
        for key in OPTION_LETTERS:
            lines.append(f"{key}. {options[key]}")
    else:
        for key, value in zip(OPTION_LETTERS, options):
            lines.append(f"{key}. {value}")
    lines.extend(("", "Answer:"))
    return "\n".join(lines)


def extract_answer_letter_fallback(response: str) -> str:
    response = response.strip().upper()

    direct_match = re.search(r"\b([ABCD])\b", response)
    if direct_match:
        return direct_match.group(1)

    answer_match = re.search(r"ANSWER\s*[:：]?\s*([ABCD])\b", response)
    if answer_match:
        return answer_match.group(1)

    paren_match = re.search(r"\(([ABCD])\)", response)
    if paren_match:
        return paren_match.group(1)

    return "A"


def extract_answer_letter_from_top_logprobs(top_logprobs: Mapping[str, float] | None) -> str | None:
    if not top_logprobs:
        return None

    option_scores: dict[str, float] = {}
    for token, logprob in top_logprobs.items():
        normalized = str(token).strip().upper()
        if normalized in OPTION_LETTERS:
            option_scores[normalized] = float(logprob)

    if not option_scores:
        return None
    return max(option_scores, key=option_scores.get)
