import asyncio
import fcntl
import json
import logging
import os
import re
import tempfile
import time
from typing import Any, Optional
from urllib.parse import urlparse
from uuid import uuid4

import requests

from .base_tool import BaseTool
from .schemas import OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, OpenAIFunctionSchema
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
JINA_RPM_LIMIT = int(os.getenv("JINA_RATE_LIMIT_RPM", "20"))
JINA_MAX_RETRIES = int(os.getenv("JINA_MAX_RETRIES", "3"))
JINA_RETRY_BASE_DELAY = float(os.getenv("JINA_RETRY_BASE_DELAY", "2.0"))
JINA_RATE_LIMIT_STATE_FILE = os.getenv(
    "JINA_RATE_LIMIT_STATE_FILE",
    os.path.join(tempfile.gettempdir(), "tournament_grpo_jina_global_rate_limit.state"),
)

_jina_rate_limiter = None


class _CrossProcessRateLimiter:
    def __init__(self, requests_per_minute: float):
        self._interval = 60.0 / requests_per_minute
        self._state_file = JINA_RATE_LIMIT_STATE_FILE
        state_dir = os.path.dirname(self._state_file)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)

    def _reserve_slot(self) -> float:
        now = time.time()
        fd = os.open(self._state_file, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            with os.fdopen(fd, "r+", encoding="utf-8", closefd=False) as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    raw = handle.read().strip()
                    last_reserved = float(raw) if raw else 0.0
                    reserved_at = max(now, last_reserved + self._interval)
                    handle.seek(0)
                    handle.truncate()
                    handle.write(f"{reserved_at:.9f}")
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            os.close(fd)

        return max(0.0, reserved_at - now)

    async def acquire(self) -> None:
        wait = await asyncio.to_thread(self._reserve_slot)
        if wait > 0:
            await asyncio.sleep(wait)


def _get_jina_rate_limiter() -> _CrossProcessRateLimiter:
    global _jina_rate_limiter
    if _jina_rate_limiter is None:
        _jina_rate_limiter = _CrossProcessRateLimiter(JINA_RPM_LIMIT)
    return _jina_rate_limiter


def _require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable is not set")
    return value


def _with_proxy_scheme(proxy_url: str) -> str:
    proxy_url = (proxy_url or "").strip()
    if proxy_url and "://" not in proxy_url:
        return f"http://{proxy_url}"
    return proxy_url


def _jina_reader_proxies() -> dict[str, str] | None:
    https_proxy = _with_proxy_scheme(
        os.getenv("JINA_READER_HTTPS_PROXY")
        or os.getenv("JINA_READER_PROXY")
        or os.getenv("JINA_HTTPS_PROXY")
        or ""
    )
    http_proxy = _with_proxy_scheme(
        os.getenv("JINA_READER_HTTP_PROXY")
        or os.getenv("JINA_READER_PROXY")
        or os.getenv("JINA_HTTP_PROXY")
        or https_proxy
    )
    proxies = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    return proxies or None


def _request_ignoring_env_proxy(method: str, url: str, **kwargs) -> requests.Response:
    with requests.Session() as session:
        session.trust_env = False
        return session.request(method, url, **kwargs)


def _truncate_text(text: str, max_length: int, truncate_side: str = "middle") -> str:
    if max_length <= 0 or len(text) <= max_length:
        return text

    if truncate_side == "left":
        return text[:max_length] + "...(truncated)"
    if truncate_side == "right":
        return "(truncated)..." + text[-max_length:]

    half = max_length // 2
    return text[:half] + "...(truncated)..." + text[-half:]


class _TournamentGRPOBaseTool(BaseTool):
    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema]):
        super().__init__(config, tool_schema)
        self.timeout = int(config.get("timeout", DEFAULT_TIMEOUT))

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        return instance_id or str(uuid4()), ToolResponse()

    async def release(self, instance_id: str, **kwargs) -> None:
        return None

    @staticmethod
    def _wrap_tool_output(content_blocks: list[str]) -> str:
        content = "\n".join(block for block in content_blocks if block)
        return f"<tool_output>\n{content}\n</tool_output>"

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _get_search_api_url(self) -> str:
        url = self._clean_text(
            self.config.get("url", os.getenv("WEB_SEARCH_API_URL", ""))
        )
        if not url:
            raise ValueError(
                "Web search API URL is not configured. Set `url` in the tool "
                "config or export `WEB_SEARCH_API_URL`."
            )
        return url

    def _get_search_api_key(self) -> str:
        api_key_env = self._clean_text(self.config.get("api_key_env")) or "WEB_SEARCH_API_KEY"
        return _require_env_var(api_key_env)

    def _get_jina_api_key(self) -> str:
        api_key_env = self._clean_text(self.config.get("jina_api_key_env")) or "JINA_API_KEY"
        return self._clean_text(os.getenv(api_key_env))

    def _search_api_request(self, payload: dict[str, Any]) -> Any:
        response = _request_ignoring_env_proxy(
            "POST",
            self._get_search_api_url(),
            headers={"api-key": self._get_search_api_key(), "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    @classmethod
    def _normalize_result_record(cls, record: dict[str, Any]) -> dict[str, str]:
        return {
            "title": cls._clean_text(record.get("title") or record.get("name")),
            "url": cls._clean_text(record.get("url") or record.get("link") or record.get("href")),
            "snippet": cls._clean_text(
                record.get("snippet")
                or record.get("description")
                or record.get("summary")
                or record.get("text")
                or record.get("content")
                or record.get("body")
            ),
            "date": cls._clean_text(
                record.get("date")
                or record.get("publishedDate")
                or record.get("time")
                or record.get("publish_time")
                or record.get("publish_date")
            ),
        }

    @classmethod
    def _extract_search_result_records(cls, payload: Any) -> list[dict[str, str]]:
        if not isinstance(payload, dict):
            return []

        for key_path in (("search_result",), ("data", "search_result"), ("result", "search_result")):
            current = payload
            ok = True
            for key in key_path:
                if not isinstance(current, dict):
                    ok = False
                    break
                current = current.get(key)
            if not ok or not isinstance(current, list):
                continue

            records = []
            for item in current:
                if not isinstance(item, dict):
                    continue
                normalized = cls._normalize_result_record(item)
                if normalized["title"] or normalized["url"] or normalized["snippet"]:
                    records.append(normalized)
            if records:
                return records
        return []

    @classmethod
    def _extract_webpage_payload(cls, payload: Any, fallback_url: str) -> dict[str, str]:
        if isinstance(payload, dict):
            data = payload.get("data", payload)
            if isinstance(data, dict):
                return {
                    "url": cls._clean_text(data.get("url")) or fallback_url,
                    "title": cls._clean_text(data.get("title")),
                    "content": cls._clean_text(
                        data.get("content")
                        or data.get("text")
                        or data.get("markdown")
                        or data.get("body")
                        or data.get("result")
                    ),
                }

        return {"url": fallback_url, "title": "", "content": cls._clean_text(payload)}

    @classmethod
    def _normalize_url(cls, url: Any) -> str:
        normalized = cls._clean_text(url)
        if not normalized:
            raise ValueError("Empty URL")
        if "://" not in normalized:
            normalized = f"https://{normalized.lstrip('/')}"
        parsed = urlparse(normalized)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {normalized}")
        return normalized


class GoogleSearchTool(_TournamentGRPOBaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="google_search",
                description="General web search for relevant webpages and snippets.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "query": OpenAIFunctionPropertySchema(type="string", description="The search query."),
                    },
                    required=["query"],
                ),
            ),
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = self._clean_text(parameters.get("query"))
        if not query:
            return ToolResponse(text="Error: query is required."), 0.0, {"status": "error"}

        max_results = int(self.config.get("max_results", 10))

        def _search():
            payload = self._search_api_request(
                {
                    "search_engine": self._clean_text(self.config.get("search_engine")) or "search_prime",
                    "search_query": query,
                    "query_rewrite": str(self.config.get("query_rewrite", "false")).lower(),
                }
            )
            return self._extract_search_result_records(payload)[:max_results]

        try:
            results = await asyncio.to_thread(_search)
            blocks = []
            for result in results:
                lines = [
                    f"Title: {result['title']}",
                    f"URL: {result['url']}",
                    f"Search Snippet: {result['snippet']}",
                ]
                if result["date"]:
                    lines.append(f"Date: {result['date']}")
                blocks.append("<snippet>\n" + "\n".join(lines) + "\n</snippet>")

            if not blocks:
                blocks.append("<snippet>\nNo search results found.\n</snippet>")

            return (
                ToolResponse(text=self._wrap_tool_output(blocks)),
                0.0,
                {"status": "success", "query_count": 1, "backend": "web_search", "total_results": len(results)},
            )
        except Exception as exc:
            return (
                ToolResponse(text=f"Error performing google_search: {exc}"),
                0.0,
                {"status": "error", "error": str(exc), "query_count": 1},
            )


class BrowseWebpageTool(_TournamentGRPOBaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="browse_webpage",
                description="Open a specific URL and extract readable page text.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "url": OpenAIFunctionPropertySchema(type="string", description="The webpage URL to browse."),
                    },
                    required=["url"],
                ),
            ),
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        url = self._clean_text(parameters.get("url"))
        if not url:
            return ToolResponse(text="Error: url is required."), 0.0, {"status": "error"}

        normalized_url = self._normalize_url(url)

        def _browse() -> dict[str, str]:
            headers = {}
            jina_api_key = self._get_jina_api_key()
            if jina_api_key:
                headers["Authorization"] = f"Bearer {jina_api_key}"
            response = requests.get(
                f"https://r.jina.ai/{normalized_url}",
                headers=headers,
                timeout=self.timeout,
                proxies=_jina_reader_proxies(),
            )
            response.raise_for_status()

            content_type = self._clean_text(response.headers.get("Content-Type")).lower()
            if "json" in content_type:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                if payload is not None:
                    extracted = self._extract_webpage_payload(payload, fallback_url=normalized_url)
                    if self._clean_text(extracted.get("content")):
                        return extracted

            text_content = self._clean_text(response.text)
            if not text_content:
                raise ValueError("Jina Reader response did not include readable page content")
            return {"url": normalized_url, "title": "", "content": text_content}

        async def _browse_with_rate_limit() -> dict[str, str]:
            limiter = _get_jina_rate_limiter()
            last_error = None
            for attempt in range(JINA_MAX_RETRIES):
                await limiter.acquire()
                try:
                    return await asyncio.to_thread(_browse)
                except requests.exceptions.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 429:
                        last_error = exc
                        await asyncio.sleep(JINA_RETRY_BASE_DELAY * (2**attempt))
                    else:
                        raise
            raise last_error if last_error is not None else RuntimeError("Jina request failed")

        try:
            payload = await _browse_with_rate_limit()
            content = self._clean_text(payload.get("content")) or "No content available"
            agent_data = kwargs.get("agent_data")
            if agent_data is not None:
                max_content_length = int(getattr(agent_data, "max_tool_response_length", 0) or 0)
                truncate_side = str(getattr(agent_data, "tool_response_truncate_side", "middle"))
                content = _truncate_text(content, max_content_length, truncate_side)

            text = self._wrap_tool_output(
                [
                    "<webpage>\nTitle: {title}\nURL: {url}\nContent: {content}\n</webpage>".format(
                        title=self._clean_text(payload.get("title")) or "No title available",
                        url=self._clean_text(payload.get("url")) or normalized_url,
                        content=content,
                    )
                ]
            )
            return ToolResponse(text=text), 0.0, {"status": "success", "query_count": 1, "backend": "jina"}
        except Exception as exc:
            return (
                ToolResponse(text=f"Error performing browse_webpage: {exc}"),
                0.0,
                {"status": "error", "error": str(exc), "query_count": 1},
            )
