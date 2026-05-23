from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from prompt.score_prompt_en import generate_merged_score_prompt as en_merged_score_prompt  # noqa: E402
from prompt.score_prompt_zh import generate_merged_score_prompt as zh_merged_score_prompt  # noqa: E402


def _load_function(module_path: Path, function_name: str):
    spec = importlib.util.spec_from_file_location(f"_drb_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


calculate_weighted_scores = _load_function(BENCH_DIR / "utils/score_calculator.py", "calculate_weighted_scores")

RACE_DIMS = ("comprehensiveness", "insight", "instruction_following", "readability")


class JudgeHTTPStatusError(RuntimeError):
    def __init__(self, status_code: int, response_text: str, api_url: str):
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(f"Judge HTTP {status_code} for url: {api_url}; response_text={response_text[:2000]}")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_json_from_markdown(text: str) -> str | None:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1].strip()
    return None


def format_criteria_list(criteria_data: dict[str, Any]) -> str:
    criteria_for_prompt = {}
    criterions_dict = criteria_data.get("criterions", {})
    for dim, criterions_list in criterions_dict.items():
        if not isinstance(criterions_list, list):
            continue
        criteria_for_prompt[dim] = []
        for crit_item in criterions_list:
            if isinstance(crit_item, dict) and "criterion" in crit_item and "explanation" in crit_item:
                criteria_for_prompt[dim].append(
                    {
                        "criterion": crit_item["criterion"],
                        "explanation": crit_item["explanation"],
                    }
                )
    return json.dumps(criteria_for_prompt, ensure_ascii=False, indent=2)


def _request_ignoring_env_proxy(method: str, url: str, **kwargs) -> requests.Response:
    with requests.Session() as session:
        session.trust_env = False
        return session.request(method, url, **kwargs)


class GenerateContentJudgeClient:
    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.api_url = str(
            config.get("api_url")
            or os.getenv("JUDGE_API_URL", "http://localhost:8000/v1/chat/completions")
        ).strip()
        if not self.api_url:
            raise ValueError("Missing judge API URL. Set JUDGE_API_URL or trainer.deep_research_bench.judge.api_url.")

        self.api_key = str(config.get("api_key") or os.getenv("JUDGE_API_KEY", "")).strip()
        self.model = str(config.get("model") or os.getenv("JUDGE_MODEL_NAME", "Qwen2.5-72B-Instruct"))
        self.temperature = float(config.get("temperature", 0.0))
        self.top_p = float(config.get("top_p", 0.95))
        self.max_tokens = int(config.get("max_tokens", 8192))
        self.timeout = int(config.get("timeout", 600))

    def _build_request(self, user_prompt: str, system_prompt: str) -> tuple[dict[str, str], dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": 0,
        }
        return headers, payload

    def generate(self, user_prompt: str, system_prompt: str = "") -> tuple[str, dict[str, Any]]:
        headers, payload = self._build_request(user_prompt, system_prompt)
        response = _request_ignoring_env_proxy(
            "POST",
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        status_code = response.status_code
        response_text = response.text
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise JudgeHTTPStatusError(status_code, response_text, self.api_url) from exc
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"Judge chat response has no choices: {data}")
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = str(message.get("content") or "").strip()
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage", {})
        model_version = data.get("model")
        response_id = data.get("id")

        if not content:
            raise ValueError(f"Judge response has empty content: {data}")

        metadata = {
            "judge_finish_reason": finish_reason,
            "judge_usage": usage,
            "judge_max_tokens": self.max_tokens,
            "judge_truncated": finish_reason in {"MAX_TOKENS", "length"},
            "judge_model": self.model,
            "judge_model_version": model_version,
            "judge_response_id": response_id,
        }
        return str(content), metadata


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_map(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    return {item["prompt"]: item for item in load_jsonl(str(path)) if item.get("prompt")}


def _score_one_article(
    article: dict[str, Any],
    reference_map: dict[str, dict[str, Any]],
    criteria_map: dict[str, dict[str, Any]],
    judge_client: GenerateContentJudgeClient,
    max_retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    task_id = article.get("id")
    prompt = article.get("prompt", "")
    language = article.get("language", "en")

    if prompt not in reference_map:
        return {"id": task_id, "prompt": prompt, "error": "Reference article not found"}
    if prompt not in criteria_map:
        return {"id": task_id, "prompt": prompt, "error": "Evaluation criteria not found"}

    criteria_data = criteria_map[prompt]
    try:
        criteria_list_str = format_criteria_list(criteria_data)
    except Exception as exc:
        return {"id": task_id, "prompt": prompt, "error": f"Failed to format criteria: {exc}"}

    prompt_template = zh_merged_score_prompt if language == "zh" else en_merged_score_prompt
    user_prompt = prompt_template.format(
        task_prompt=prompt,
        article_1=article.get("article", ""),
        article_2=reference_map[prompt].get("article", ""),
        criteria_list=criteria_list_str,
    )

    last_error: Exception | None = None
    llm_output_json = None
    llm_response = ""
    judge_metadata: dict[str, Any] = {}
    for attempt in range(1, max_retries + 1):
        try:
            llm_response, judge_metadata = judge_client.generate(user_prompt=user_prompt, system_prompt="")
            json_str = extract_json_from_markdown(llm_response)
            if not json_str:
                raise ValueError("Failed to extract JSON from judge response")
            llm_output_json = json.loads(json_str)
            missing_dims = [dim for dim in RACE_DIMS if dim not in llm_output_json]
            if missing_dims:
                raise ValueError(f"Missing expected dimensions: {missing_dims}")
            break
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)
    else:
        result = {
            "id": task_id,
            "prompt": prompt,
            "error": f"Failed after {max_retries} retries: {last_error}",
            "model_output": llm_response[:500] if llm_response else "",
            **judge_metadata,
        }
        if isinstance(last_error, JudgeHTTPStatusError):
            result["judge_status_code"] = last_error.status_code
            result["judge_response_text"] = last_error.response_text[:4000]
        return result

    try:
        scores = calculate_weighted_scores(llm_output_json, criteria_data, language)
        target_total = scores["target"]["total"]
        reference_total = scores["reference"]["total"]
        overall_score = target_total / (target_total + reference_total) if target_total + reference_total > 0 else 0.0

        normalized_dims = {}
        for dim in RACE_DIMS:
            dim_key = f"{dim}_weighted_avg"
            target_score = scores["target"]["dims"].get(dim_key, 0.0)
            reference_score = scores["reference"]["dims"].get(dim_key, 0.0)
            normalized_dims[dim] = (
                target_score / (target_score + reference_score) if target_score + reference_score > 0 else 0.0
            )
    except Exception as exc:
        return {"id": task_id, "prompt": prompt, "error": f"Error calculating scores: {exc}"}

    return {
        "id": task_id,
        "prompt": prompt,
        "comprehensiveness": float(normalized_dims["comprehensiveness"]),
        "insight": float(normalized_dims["insight"]),
        "instruction_following": float(normalized_dims["instruction_following"]),
        "readability": float(normalized_dims["readability"]),
        "overall_score": float(overall_score),
        **judge_metadata,
    }


def evaluate_race_articles(
    articles: list[dict[str, Any]],
    bench_dir: str | os.PathLike[str] | None = None,
    criteria_file: str | os.PathLike[str] | None = None,
    reference_file: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    max_workers: int = 4,
    max_retries: int = 3,
    retry_sleep: float = 1.5,
    judge_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    bench_path = Path(bench_dir or BENCH_DIR)
    criteria_path = Path(criteria_file or bench_path / "data/criteria_data/criteria.jsonl")
    reference_path = Path(reference_file or bench_path / "data/test_data/cleaned_data/reference.jsonl")

    reference_map = _load_map(reference_path)
    criteria_map = _load_map(criteria_path)
    judge_client = GenerateContentJudgeClient(judge_config)

    results = []
    worker_count = max(1, int(max_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _score_one_article,
                article,
                reference_map,
                criteria_map,
                judge_client,
                int(max_retries),
                float(retry_sleep),
            )
            for article in articles
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        raw_path = output_path / "raw_results.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for result in sorted(results, key=lambda item: str(item.get("id", ""))):
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    successful = [item for item in results if "error" not in item]
    if not successful:
        error_count = len(results)
        raise RuntimeError(f"deep_research_bench RACE evaluation produced no successful results ({error_count} errors).")

    return {
        "deep_research_bench/race_Comprehensiveness": _safe_mean(
            [float(item.get("comprehensiveness", 0.0)) for item in successful]
        ),
        "deep_research_bench/race_Insight": _safe_mean([float(item.get("insight", 0.0)) for item in successful]),
        "deep_research_bench/race_Instruction_Following": _safe_mean(
            [float(item.get("instruction_following", 0.0)) for item in successful]
        ),
        "deep_research_bench/race_Readability": _safe_mean(
            [float(item.get("readability", 0.0)) for item in successful]
        ),
        "deep_research_bench/race_ overall_score": _safe_mean(
            [float(item.get("overall_score", 0.0)) for item in successful]
        ),
    }
