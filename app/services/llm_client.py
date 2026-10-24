import json
import logging
import httpx
from app.config import LLM_API_KEY, LLM_API_URL, LLM_MODEL

logger = logging.getLogger(__name__)


def complete_json(system_prompt: str, payload: dict) -> dict | None:
    if not LLM_API_URL or not LLM_MODEL:
        logger.warning("LLM request skipped because LLM_API_URL or LLM_MODEL is not configured")
        return None
    if "api.openai.com" in LLM_API_URL and not LLM_API_KEY:
        logger.warning("LLM request skipped because no LLM_API_KEY or OPENAI_API_KEY is configured")
        return None
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    try:
        response = httpx.post(LLM_API_URL, headers=headers, json={
            "model": LLM_MODEL, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        }, timeout=120)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("LLM request failed with HTTP %s: %s", response.status_code, response.text[:1000])
            return None
        content = response.json()["choices"][0]["message"]["content"]
        result = json.loads(content)
        return result if isinstance(result, dict) else None
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.exception("LLM request or JSON parsing failed (%s)", type(exc).__name__)
        return None
