from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_XML_TAG_REGEX = re.compile(r"<(/?)([A-Za-z_][A-Za-z0-9_]*)([^>]*)>", flags=re.DOTALL)
_ANSWER_BLOCK_REGEX = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
_TAG_REGEX = re.compile(r"<(/?)([A-Za-z_][A-Za-z0-9_]*)([^>]*)>", flags=re.DOTALL)
_ATTR_REGEX = re.compile(r'\s+([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
_ALLOWED_TAGS = {"think", "call_tool", "tool_output", "snippet", "webpage", "answer"}
_ALLOWED_TOOL_NAMES = {"google_search", "browse_webpage"}
_ALLOWED_ATTRS_BY_TAG = {
    "think": set(),
    "call_tool": {"name"},
    "tool_output": set(),
    "snippet": set(),
    "webpage": set(),
    "answer": set(),
}


class TranscriptBlock:
    def __init__(self, tag: str, attrs: dict[str, str], content: str, raw: str):
        self.tag = tag
        self.attrs = attrs
        self.content = content
        self.raw = raw


class TranscriptParseResult:
    def __init__(
        self,
        blocks: list[TranscriptBlock],
        unknown_tags: list[str],
        unclosed_tags: list[str],
        mismatched_tags: list[str],
        stray_texts: list[str],
        invalid_attrs: list[str],
    ):
        self.blocks = blocks
        self.unknown_tags = unknown_tags
        self.unclosed_tags = unclosed_tags
        self.mismatched_tags = mismatched_tags
        self.stray_texts = stray_texts
        self.invalid_attrs = invalid_attrs


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except Exception:
        return {}


def _get_tournament_config(tournament_reward: Any = None, **kwargs) -> dict[str, Any]:
    cfg = {
        "model": os.getenv("TOURNAMENT_JUDGE_MODEL_NAME", os.getenv("JUDGE_MODEL_NAME", "Qwen2.5-72B-Instruct")),
        "temperature": 0.0,
        "max_tokens": 1024,
        "timeout": 240,
        "max_retries": 3,
        "retry_sleep": 1.5,
        "api_key": None,
        "api_url": os.getenv(
            "TOURNAMENT_JUDGE_API_URL",
            os.getenv(
                "JUDGE_API_URL",
                "http://localhost:8000/v1/chat/completions",
            ),
        ),
        "top_p": 0.95,
        "seed": 0,
        "max_rollout_chars": 120000,
        "group_size": 2,
        "num_winners_per_group": 1,
        "target_finalists": 1,
        "score_increment": 1.0,
        "equal_score_reward": 0.5,
        "num_tournament_repeats": 1,
    }
    reward_kwargs = _as_plain_dict(kwargs.pop("reward_kwargs", None))
    cfg.update(_as_plain_dict(tournament_reward))
    cfg.update(reward_kwargs)
    for key, value in kwargs.items():
        if value is not None and key in cfg:
            cfg[key] = value
    cfg["api_key"] = (
        cfg.get("api_key")
        or os.getenv("TOURNAMENT_JUDGE_API_KEY")
        or os.getenv("JUDGE_API_KEY")
    )
    return cfg


def _validate_tournament_config(cfg: dict[str, Any]) -> tuple[str, str]:
    api_key = str(cfg.get("api_key") or "").strip()
    api_url = str(cfg.get("api_url") or "").strip()
    if not api_url:
        raise ValueError("Missing tournament judge API URL.")
    return api_key, api_url


def _request_ignoring_env_proxy(method: str, url: str, **kwargs) -> requests.Response:
    with requests.Session() as session:
        session.trust_env = False
        return session.request(method, url, **kwargs)


def _parse_attrs(tag: str, attr_text: str) -> tuple[dict[str, str], bool]:
    raw_text = attr_text or ""
    attrs: dict[str, str] = {}
    cursor = 0

    for match in _ATTR_REGEX.finditer(raw_text):
        if raw_text[cursor : match.start()].strip():
            return attrs, False
        key = match.group(1)
        value = match.group(2)
        if key in attrs:
            return attrs, False
        attrs[key] = value
        cursor = match.end()

    if raw_text[cursor:].strip():
        return attrs, False

    allowed_attrs = _ALLOWED_ATTRS_BY_TAG.get(tag, set())
    if any(key not in allowed_attrs for key in attrs):
        return attrs, False
    if tag == "call_tool" and "name" not in attrs:
        return attrs, False
    return attrs, True


def _contains_xml_like_tag(text: str) -> bool:
    return bool(_XML_TAG_REGEX.search(text or ""))


def _parse_transcript(text: str) -> TranscriptParseResult:
    text = text or ""
    blocks: list[TranscriptBlock] = []
    unknown_tags: list[str] = []
    unclosed_tags: list[str] = []
    mismatched_tags: list[str] = []
    stray_texts: list[str] = []
    invalid_attrs: list[str] = []
    stack: list[dict[str, Any]] = []
    cursor = 0

    for match in _TAG_REGEX.finditer(text):
        if match.start() > cursor and not stack:
            raw_between = text[cursor : match.start()]
            if raw_between.strip():
                stray_texts.append(raw_between.strip())

        is_closing = match.group(1) == "/"
        tag = match.group(2)
        attr_text = match.group(3) or ""

        if tag not in _ALLOWED_TAGS:
            unknown_tags.append(tag)

        if not is_closing:
            attrs, attrs_valid = _parse_attrs(tag, attr_text)
            if not attrs_valid:
                invalid_attrs.append(tag)
            stack.append(
                {
                    "tag": tag,
                    "attrs": attrs,
                    "open_end": match.end(),
                    "start": match.start(),
                }
            )
            cursor = match.end()
            continue

        if attr_text.strip():
            invalid_attrs.append(tag)

        if not stack:
            mismatched_tags.append(tag)
            cursor = match.end()
            continue

        if stack[-1]["tag"] == tag:
            item = stack.pop()
        else:
            match_index = next((idx for idx in range(len(stack) - 1, -1, -1) if stack[idx]["tag"] == tag), None)
            mismatched_tags.append(tag)
            if match_index is None:
                cursor = match.end()
                continue
            for dangling in stack[match_index + 1 :]:
                unclosed_tags.append(dangling["tag"])
            item = stack[match_index]
            stack = stack[:match_index]

        raw = text[item["start"] : match.end()]
        content = text[item["open_end"] : match.start()]
        if not stack:
            blocks.append(TranscriptBlock(tag=item["tag"], attrs=item["attrs"], content=content, raw=raw))
        cursor = match.end()

    if cursor < len(text) and not stack:
        raw_tail = text[cursor:]
        if raw_tail.strip():
            stray_texts.append(raw_tail.strip())

    for item in stack:
        unclosed_tags.append(item["tag"])

    return TranscriptParseResult(
        blocks=blocks,
        unknown_tags=unknown_tags,
        unclosed_tags=unclosed_tags,
        mismatched_tags=mismatched_tags,
        stray_texts=stray_texts,
        invalid_attrs=invalid_attrs,
    )


def _analyze_format(solution_str: str) -> dict[str, Any]:
    parsed = _parse_transcript(solution_str)
    blocks = parsed.blocks
    failure_reasons: list[str] = []

    if parsed.unknown_tags:
        failure_reasons.append("tag:unknown_tags")
    if parsed.invalid_attrs:
        failure_reasons.append("tag:invalid_attrs")
    if parsed.unclosed_tags or parsed.mismatched_tags:
        failure_reasons.append("tag:broken_xml")
    if parsed.stray_texts:
        failure_reasons.append("structure:stray_text_outside_tags")

    invalid_tool_name_blocks = [
        block
        for block in blocks
        if block.tag == "call_tool" and block.attrs.get("name", "").strip() not in _ALLOWED_TOOL_NAMES
    ]
    if invalid_tool_name_blocks:
        failure_reasons.append("tool:unknown_tool_name")

    answer_blocks = [(idx, block) for idx, block in enumerate(blocks) if block.tag == "answer"]
    answer_tag_count = len(answer_blocks)
    tool_call_count = sum(
        1 for block in blocks if block.tag == "call_tool" and block.attrs.get("name", "").strip() in _ALLOWED_TOOL_NAMES
    )
    tool_output_count = sum(1 for block in blocks if block.tag == "tool_output")

    rollout_answer = _extract_rollout_answer(solution_str)
    if not rollout_answer:
        failure_reasons.append("answer:invalid_or_missing_final_answer")

    structure_valid = not failure_reasons
    if structure_valid:
        if not blocks or blocks[0].tag != "think" or not blocks[0].content.strip():
            failure_reasons.append("structure:must_start_with_non_empty_think")
            structure_valid = False

    if structure_valid:
        if blocks[-1].tag != "answer":
            failure_reasons.append("structure:must_end_with_answer")
            structure_valid = False

    if structure_valid:
        i = 1
        last_index = len(blocks) - 1
        if i >= last_index:
            failure_reasons.append("structure:must_have_at_least_one_tool_cycle")
            structure_valid = False

    total_google_cycles = 0
    total_browse_cycles = 0
    segment_count = 0

    def _matches_triplet(start_idx: int, tool_name: str) -> bool:
        if start_idx + 2 >= len(blocks):
            return False
        call_block = blocks[start_idx]
        tool_output_block = blocks[start_idx + 1]
        think_block = blocks[start_idx + 2]
        return (
            call_block.tag == "call_tool"
            and call_block.attrs.get("name", "").strip() == tool_name
            and bool(call_block.content.strip())
            and tool_output_block.tag == "tool_output"
            and bool(tool_output_block.content.strip())
            and think_block.tag == "think"
            and bool(think_block.content.strip())
        )

    if structure_valid:
        i = 1
        last_index = len(blocks) - 1
        while i < last_index:
            google_cycles_in_segment = 0
            while i < last_index and _matches_triplet(i, "google_search"):
                google_cycles_in_segment += 1
                total_google_cycles += 1
                i += 3

            if google_cycles_in_segment == 0:
                failure_reasons.append("structure:each_segment_must_start_with_google_search_cycle")
                structure_valid = False
                break

            while i < last_index and _matches_triplet(i, "browse_webpage"):
                total_browse_cycles += 1
                i += 3

            segment_count += 1

        if structure_valid and i != last_index:
            failure_reasons.append("structure:invalid_tool_cycle_order")
            structure_valid = False

    format_reward = 1.0 if structure_valid else 0.0

    return {
        "format_reward": float(format_reward),
        "format_valid": int(structure_valid),
        "failure_reasons": failure_reasons,
        "answer_tag_count": int(answer_tag_count),
        "tool_call_count": int(tool_call_count),
        "tool_output_count": int(tool_output_count),
        "google_search_cycle_count": int(total_google_cycles),
        "browse_webpage_cycle_count": int(total_browse_cycles),
        "segment_count": int(segment_count),
        "stray_text_count": int(len(parsed.stray_texts)),
        "rollout_answer": rollout_answer,
    }


def _extract_rollout_answer(solution_str: str) -> str:
    text = solution_str or ""
    matches = list(_ANSWER_BLOCK_REGEX.finditer(text))
    if len(matches) != 1:
        return ""

    match = matches[0]
    answer_content = (match.group(1) or "").strip()
    if not answer_content:
        return ""
    if _contains_xml_like_tag(answer_content):
        return ""
    if text[match.end() :].strip():
        return ""
    return answer_content


def _extract_query_and_rubrics(ground_truth: Any, extra_info: dict[str, Any] | None = None) -> tuple[str, list[dict[str, Any]]]:
    extra_info = extra_info or {}

    if isinstance(ground_truth, dict):
        query = str(ground_truth.get("query", "") or extra_info.get("query", "")).strip()
        rubrics = ground_truth.get("rubrics", extra_info.get("rubrics", []))
    else:
        query = str(extra_info.get("query", "") or ground_truth or "").strip()
        rubrics = extra_info.get("rubrics", [])

    if hasattr(rubrics, "tolist"):
        rubrics = rubrics.tolist()
    if not isinstance(rubrics, list):
        rubrics = []

    normalized_rubrics: list[dict[str, Any]] = []
    for item in rubrics:
        if isinstance(item, dict):
            normalized_rubrics.append(item)
        else:
            try:
                normalized_rubrics.append(dict(item))
            except Exception:
                continue
    return query, normalized_rubrics


def _build_tournament_prompt(
    query: str,
    rubrics: list[dict[str, Any]],
    candidate_texts: list[str],
    num_winners: int,
) -> str:
    rubric_lines = []
    for idx, item in enumerate(rubrics, start=1):
        rubric_lines.append(
            f"{idx}. description: {str(item.get('description', item.get('rubric', ''))).strip()}\n"
            f"   dimension: {str(item.get('dimension', '')).strip()}\n"
            f"   importance: {str(item.get('importance', '')).strip()}\n"
            f"   title: {str(item.get('title', '')).strip()}"
        )

    rubric_block = "\n".join(rubric_lines) if rubric_lines else "No rubrics provided."
    candidate_blocks = []
    for idx, text in enumerate(candidate_texts, start=1):
        candidate_blocks.append(f"Candidate {idx}:\n{text}")
    candidate_block = "\n\n".join(candidate_blocks)
    return (
        "You are an expert judge for deep-research quality.\n"
        "Compare the candidate responses to the same query using the rubric below.\n\n"
        "Query:\n"
        f"{query}\n\n"
        "Rubric (description, dimension, importance, title):\n"
        f"{rubric_block}\n\n"
        "Candidates:\n"
        f"{candidate_block}\n\n"
        "Selection rule:\n"
        f"- Select exactly {num_winners} winner(s).\n"
        "- Prefer responses that are accurate, complete, well-supported, and directly answer the query.\n"
        "- Return JSON only, without markdown or explanations.\n"
        '- Use 1-based candidate numbers with this schema: {"winners": [1]}\n'
    )


def _parse_tournament_response(content: str, num_candidates: int, num_winners: int) -> list[int]:
    cleaned = (content or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    parsed: Any
    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = None

    winners: list[Any] = []
    if isinstance(parsed, dict):
        winners = parsed.get("winners", parsed.get("winner", []))
    elif isinstance(parsed, list):
        winners = parsed
    elif isinstance(parsed, (int, float)):
        winners = [parsed]
    else:
        winners = re.findall(r"\d+", cleaned)

    if not isinstance(winners, list):
        winners = [winners]

    parsed_winners: list[int] = []
    for value in winners:
        try:
            local_idx = int(value) - 1
        except Exception:
            continue
        if 0 <= local_idx < num_candidates and local_idx not in parsed_winners:
            parsed_winners.append(local_idx)
        if len(parsed_winners) >= num_winners:
            break

    if len(parsed_winners) != num_winners:
        raise ValueError(f"Tournament judge returned invalid winners: {cleaned}")
    return parsed_winners


def _summarize_tournament_response(resp: Any, limit: int = 800) -> str:
    if resp is None:
        return "resp=None"

    try:
        if isinstance(resp, dict):
            text = json.dumps(resp, ensure_ascii=False, default=str)
        elif hasattr(resp, "model_dump_json"):
            text = resp.model_dump_json(exclude_none=False)
        elif hasattr(resp, "model_dump"):
            text = json.dumps(resp.model_dump(), ensure_ascii=False, default=str)
        else:
            text = repr(resp)
    except Exception as exc:
        text = f"<failed to serialize response: {exc}; repr={repr(resp)}>"

    text = text.replace("\n", "\\n")
    if len(text) > limit:
        text = text[:limit] + "...<truncated>"
    return text


def _extract_openai_chat_text(resp: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    choices = resp.get("choices") or []
    if not choices:
        raise ValueError(f"Tournament judge chat response has no choices. summary={_summarize_tournament_response(resp)}")

    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = str(message.get("content") or "").strip()
    if not content:
        raise ValueError(
            f"Tournament judge chat response has empty content. summary={_summarize_tournament_response(resp)}"
        )

    metadata = {
        "finish_reason": choice.get("finish_reason"),
        "usage": resp.get("usage", {}),
        "model_version": resp.get("model"),
        "response_id": resp.get("id"),
    }
    return content, metadata


def _build_judge_request(prompt: str, cfg: dict[str, Any], api_url: str) -> tuple[dict[str, str], dict[str, Any]]:
    api_key = str(cfg.get("api_key") or "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    system_prompt = "You are a strict deep-research tournament judge. Return only valid JSON without markdown."
    payload = {
        "model": str(cfg["model"]),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(cfg["temperature"]),
        "max_tokens": int(cfg["max_tokens"]),
        "top_p": float(cfg["top_p"]),
        "seed": int(cfg["seed"]),
    }
    return headers, payload


def _judge_tournament_group(
    prompt: str,
    cfg: dict[str, Any],
    num_candidates: int,
    num_winners: int,
) -> tuple[list[int], str, str]:
    _api_key, api_url = _validate_tournament_config(cfg)
    last_error: Exception | None = None

    for attempt in range(1, int(cfg["max_retries"]) + 1):
        try:
            headers, payload = _build_judge_request(prompt, cfg, api_url)
            response = _request_ignoring_env_proxy(
                "POST",
                api_url,
                headers=headers,
                json=payload,
                timeout=int(cfg["timeout"]),
            )
            response.raise_for_status()
            resp = response.json()
            content, _metadata = _extract_openai_chat_text(resp)
            winners = _parse_tournament_response(content, num_candidates, num_winners)
            return winners, content, ""
        except Exception as exc:
            last_error = exc
            if attempt < int(cfg["max_retries"]):
                time.sleep(float(cfg["retry_sleep"]) * attempt)

    indices = list(range(num_candidates))
    random.shuffle(indices)
    logger.warning("Tournament judge failed after retries; falling back to random winners. error=%s", last_error)
    return indices[:num_winners], "", str(last_error) if last_error is not None else "Unknown tournament judge error"


def _normalize_tournament_scores(scores: list[float], equal_score_reward: float = 0.5) -> list[float]:
    if not scores:
        return scores
    min_score = min(scores)
    max_score = max(scores)
    denom = max_score - min_score
    if denom < 1e-12:
        return [float(equal_score_reward)] * len(scores)
    return [(score - min_score) / denom for score in scores]


def _run_tournament(
    query: str,
    rubrics: list[dict[str, Any]],
    rollout_answers: list[str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    n = len(rollout_answers)
    raw_scores = [0.0] * n
    judge_outputs: list[str] = []
    judge_errors: list[str] = []
    rounds = 0
    group_count = 0

    if n <= 1:
        return {
            "raw_scores": raw_scores,
            "rewards": _normalize_tournament_scores(raw_scores, float(cfg["equal_score_reward"])),
            "rounds": rounds,
            "group_count": group_count,
            "judge_outputs": judge_outputs,
            "judge_errors": judge_errors,
        }

    rng = random.Random(int(cfg["seed"]))
    repeats = max(1, int(cfg["num_tournament_repeats"]))
    group_size = max(2, int(cfg["group_size"]))
    target_finalists = max(1, int(cfg["target_finalists"]))
    num_winners_per_group = max(1, int(cfg["num_winners_per_group"]))
    score_increment = float(cfg["score_increment"])

    for repeat_idx in range(repeats):
        candidate_indices = list(range(n))
        while len(candidate_indices) > target_finalists:
            rounds += 1
            rng.shuffle(candidate_indices)
            effective_num_groups = max(1, len(candidate_indices) // group_size)
            groups = [[] for _ in range(effective_num_groups)]
            for offset, idx in enumerate(candidate_indices):
                groups[offset % effective_num_groups].append(idx)
            groups = [group for group in groups if group]
            group_count += len(groups)

            next_candidates: list[int] = []
            for group in groups:
                effective_winners = min(num_winners_per_group, len(group) - 1)
                if effective_winners < 1:
                    winners = group
                    judge_output = ""
                    judge_error = ""
                else:
                    group_texts = [rollout_answers[idx] for idx in group]
                    prompt = _build_tournament_prompt(
                        query=query,
                        rubrics=rubrics,
                        candidate_texts=group_texts,
                        num_winners=effective_winners,
                    )
                    winner_local_indices, judge_output, judge_error = _judge_tournament_group(
                        prompt=prompt,
                        cfg=cfg,
                        num_candidates=len(group),
                        num_winners=effective_winners,
                    )
                    winners = [group[local_idx] for local_idx in winner_local_indices if local_idx < len(group)]

                judge_outputs.append(judge_output)
                judge_errors.append(judge_error)
                for idx in winners:
                    raw_scores[idx] += score_increment
                next_candidates.extend(winners)

            candidate_indices = list(dict.fromkeys(next_candidates))
            if not candidate_indices or len(candidate_indices) >= n:
                break

        logger.debug("Tournament repeat %s finished with candidates=%s", repeat_idx + 1, candidate_indices)

    rewards = _normalize_tournament_scores(raw_scores, float(cfg["equal_score_reward"]))
    return {
        "raw_scores": raw_scores,
        "rewards": rewards,
        "rounds": rounds,
        "group_count": group_count,
        "judge_outputs": judge_outputs,
        "judge_errors": judge_errors,
    }


def _count_tool_calls(text: str, tool_name: str) -> int:
    if not text:
        return 0
    pattern = rf'<call_tool\s+name="{re.escape(tool_name)}"[^>]*>'
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def compute_score(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    tournament_reward: dict[str, Any] | None = None,
    items: list[dict[str, Any]] | None = None,
    **kwargs,
) -> dict[str, Any]:
    if items is not None:
        return compute_score_batch(items=items, tournament_reward=tournament_reward, **kwargs)

    cfg = _get_tournament_config(tournament_reward=tournament_reward, **kwargs)
    extra_info = _as_plain_dict(extra_info)
    format_info = _analyze_format(solution_str)

    rollout_answer = format_info["rollout_answer"]
    if len(rollout_answer) > int(cfg["max_rollout_chars"]):
        rollout_answer = rollout_answer[: int(cfg["max_rollout_chars"])]

    tournament_reward_value = float(cfg["equal_score_reward"]) if rollout_answer else 0.0

    format_reward = float(format_info["format_reward"])
    reward = float(tournament_reward_value + format_reward)

    google_search_call_count = _safe_int(extra_info.get("google_search_call_count"), None)
    if google_search_call_count is None:
        google_search_call_count = _count_tool_calls(solution_str, "google_search")

    browse_webpage_call_count = _safe_int(extra_info.get("browse_webpage_call_count"), None)
    if browse_webpage_call_count is None:
        browse_webpage_call_count = _count_tool_calls(solution_str, "browse_webpage")

    result = {
        "score": reward,
        "custom_reward": reward,
        "tournament_reward": float(tournament_reward_value),
        "tournament_raw_score": 0.0,
        "tournament_round_count": 0,
        "tournament_group_count": 0,
        "tournament_judge_error": "",
        "format_reward": float(format_reward),
        "format_valid": int(format_info["format_valid"]),
        "format_failure_reasons": json.dumps(format_info["failure_reasons"], ensure_ascii=False),
        "answer_tag_count": int(format_info["answer_tag_count"]),
        "tool_call_count": int(format_info["tool_call_count"]),
        "tool_output_count": int(format_info["tool_output_count"]),
        "google_search_cycle_count": int(format_info["google_search_cycle_count"]),
        "browse_webpage_cycle_count": int(format_info["browse_webpage_cycle_count"]),
        "segment_count": int(format_info["segment_count"]),
        "stray_text_count": int(format_info["stray_text_count"]),
        "google_search_call_count": int(google_search_call_count),
        "browse_webpage_call_count": int(browse_webpage_call_count),
        "rollout_answer": rollout_answer,
        "tournament_judge_model": str(cfg["model"]),
    }

    return result


def compute_score_batch(
    items: list[dict[str, Any]],
    tournament_reward: dict[str, Any] | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    cfg = _get_tournament_config(tournament_reward=tournament_reward, **kwargs)
    if not items:
        return []

    first = items[0]
    query, rubrics = _extract_query_and_rubrics(
        first.get("ground_truth"),
        extra_info=_as_plain_dict(first.get("extra_info")),
    )

    format_infos = []
    rollout_answers = []
    for item in items:
        format_info = _analyze_format(str(item.get("solution_str", "") or ""))
        rollout_answer = format_info["rollout_answer"]
        if len(rollout_answer) > int(cfg["max_rollout_chars"]):
            rollout_answer = rollout_answer[: int(cfg["max_rollout_chars"])]
        format_infos.append(format_info)
        rollout_answers.append(rollout_answer)

    tournament_input_answers = [answer if answer else "" for answer in rollout_answers]
    tournament_info = _run_tournament(query=query, rubrics=rubrics, rollout_answers=tournament_input_answers, cfg=cfg)
    tournament_rewards = tournament_info["rewards"]
    tournament_raw_scores = tournament_info["raw_scores"]
    judge_errors = [error for error in tournament_info["judge_errors"] if error]
    judge_error_text = " | ".join(list(dict.fromkeys(judge_errors))[:3])

    results = []
    for item, format_info, rollout_answer, tournament_value, raw_score in zip(
        items,
        format_infos,
        rollout_answers,
        tournament_rewards,
        tournament_raw_scores,
        strict=True,
    ):
        solution_str = str(item.get("solution_str", "") or "")
        extra_info = _as_plain_dict(item.get("extra_info"))
        if not rollout_answer:
            tournament_value = 0.0

        format_reward = float(format_info["format_reward"])
        reward = float(tournament_value + format_reward)

        google_search_call_count = _safe_int(extra_info.get("google_search_call_count"), None)
        if google_search_call_count is None:
            google_search_call_count = _count_tool_calls(solution_str, "google_search")

        browse_webpage_call_count = _safe_int(extra_info.get("browse_webpage_call_count"), None)
        if browse_webpage_call_count is None:
            browse_webpage_call_count = _count_tool_calls(solution_str, "browse_webpage")

        results.append(
            {
                "score": reward,
                "custom_reward": reward,
                "tournament_reward": float(tournament_value),
                "tournament_raw_score": float(raw_score),
                "tournament_round_count": int(tournament_info["rounds"]),
                "tournament_group_count": int(tournament_info["group_count"]),
                "tournament_judge_error": judge_error_text,
                "format_reward": float(format_reward),
                "format_valid": int(format_info["format_valid"]),
                "format_failure_reasons": json.dumps(format_info["failure_reasons"], ensure_ascii=False),
                "answer_tag_count": int(format_info["answer_tag_count"]),
                "tool_call_count": int(format_info["tool_call_count"]),
                "tool_output_count": int(format_info["tool_output_count"]),
                "google_search_cycle_count": int(format_info["google_search_cycle_count"]),
                "browse_webpage_cycle_count": int(format_info["browse_webpage_cycle_count"]),
                "segment_count": int(format_info["segment_count"]),
                "stray_text_count": int(format_info["stray_text_count"]),
                "google_search_call_count": int(google_search_call_count),
                "browse_webpage_call_count": int(browse_webpage_call_count),
                "rollout_answer": rollout_answer,
                "tournament_judge_model": str(cfg["model"]),
            }
        )

    return results


compute_score.NEEDS_REWARD_CONFIG = False
