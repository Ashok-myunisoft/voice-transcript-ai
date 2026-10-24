"""Speaker diarization through the configured RunPod worker."""
import base64
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from app.config import (
    RUNPOD_API_KEY,
    RUNPOD_ENDPOINT_URL,
    RUNPOD_JOB_TIMEOUT_SECONDS,
    RUNPOD_MAX_AUDIO_BYTES,
    RUNPOD_POLL_INTERVAL_SECONDS,
)

logger = logging.getLogger(__name__)


def _normalize_segments(segments: list[dict]) -> list[dict]:
    if isinstance(segments, dict):
        # Some workers wrap the list under a second result/annotation key.
        for key in ("segments", "turns", "results", "diarization"):
            value = segments.get(key)
            if isinstance(value, (dict, list)):
                segments = value
                break
    if isinstance(segments, dict):
        segments = [segments]
    if not isinstance(segments, list):
        return []

    normalized = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        try:
            start = float(segment.get("start", segment.get("start_time")))
            end = float(segment.get("end", segment.get("end_time")))
            # Numeric speaker IDs can be 0; avoid `or`, which treats 0 as missing.
            speaker = next(
                (segment[key] for key in ("speaker", "speaker_id", "label")
                 if key in segment and segment[key] is not None),
                None,
            )
            if end <= start or speaker is None:
                continue
            normalized.append({"start": start, "end": end, "speaker_id": str(speaker)})
        except (KeyError, TypeError, ValueError):
            continue
    normalized.sort(key=lambda turn: (turn["start"], turn["end"]))
    # Keep endpoint labels stable in order of first occurrence.
    label_map = {}
    for turn in normalized:
        label_map.setdefault(turn["speaker_id"], f"SPEAKER_{len(label_map):02d}")
        turn["speaker_id"] = label_map[turn["speaker_id"]]
    return normalized


def _runpod_diarize(audio_path: str, audio_url: str | None = None) -> tuple[list[dict], str]:
    if not RUNPOD_API_KEY:
        raise RuntimeError("RUNPOD_API_KEY is not configured")
    path = Path(audio_path)
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    if audio_url:
        audio_input = {"audio_url": audio_url}
    else:
        audio_bytes = path.read_bytes()
        if len(audio_bytes) > RUNPOD_MAX_AUDIO_BYTES:
            raise ValueError("Audio exceeds RunPod's inline Base64 size limit; configure PUBLIC_BASE_URL for audio_url delivery")
        audio_input = {"audio_base64": base64.b64encode(audio_bytes).decode("ascii"), "format": path.suffix.lstrip(".") or "wav"}

    base_url = RUNPOD_ENDPOINT_URL.rstrip("/")
    endpoint_host = urlsplit(base_url).hostname or "unknown host"

    def request(client: httpx.Client, method: str, url: str, **kwargs):
        try:
            return client.request(method, url, **kwargs)
        except httpx.ConnectError as exc:
            logger.error("Could not connect to RunPod API host %s: %s", endpoint_host, exc)
            raise RuntimeError(
                f"Cannot resolve or connect to RunPod API host {endpoint_host}; "
                "check RUNPOD_ENDPOINT_URL and DNS/network access from the app container"
            ) from exc

    transport = httpx.HTTPTransport(retries=3)
    with httpx.Client(timeout=httpx.Timeout(30.0, connect=15.0), transport=transport) as client:
        response = request(client, "POST", f"{base_url}/run", headers=headers, json={"input": audio_input})
        response.raise_for_status()
        job = response.json()
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError("RunPod did not return a job ID")

        deadline = time.monotonic() + RUNPOD_JOB_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            response = request(client, "GET", f"{base_url}/status/{job_id}", headers=headers)
            response.raise_for_status()
            job = response.json()
            status = str(job.get("status", "")).upper()
            if status == "COMPLETED":
                output = job.get("output")
                if isinstance(output, list) and len(output) == 1 and isinstance(output[0], dict):
                    output = output[0]
                if (not isinstance(output, (dict, list))
                        or (isinstance(output, dict) and output.get("error"))):
                    detail = output.get("error") if isinstance(output, dict) else output
                    raise RuntimeError(f"RunPod diarization worker returned an error: {str(detail)[:2000]}")
                # Accept the usual direct result plus common wrapper envelopes.
                payload = output
                for _ in range(3):
                    if isinstance(payload, list):
                        break
                    if not isinstance(payload, dict):
                        break
                    if any(key in payload for key in ("start", "start_time")):
                        break
                    nested = next((payload[key] for key in ("segments", "turns", "results", "diarization", "output")
                                   if isinstance(payload.get(key), (dict, list))), None)
                    if nested is None:
                        break
                    payload = nested
                segments = _normalize_segments(payload)
                if not segments:
                    output_keys = ", ".join(map(str, output.keys())) if isinstance(output, dict) else "list"
                    raise RuntimeError(
                        "RunPod diarization completed but returned no usable speaker segments "
                        f"(output keys: {output_keys})"
                    )
                return segments, "completed"
            if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                # Preserve the worker's traceback/error instead of hiding it behind
                # the terminal job status. RunPod may place it in `error` or output.
                detail = job.get("error")
                if not detail:
                    detail = job.get("output")
                if isinstance(detail, (dict, list)):
                    detail = json.dumps(detail, ensure_ascii=False)
                suffix = f": {str(detail)[:2000]}" if detail else ""
                raise RuntimeError(f"RunPod diarization job ended with status {status}{suffix}")
            time.sleep(RUNPOD_POLL_INTERVAL_SECONDS)
    raise TimeoutError("RunPod diarization job exceeded the configured wait time")


def diarize_audio(audio_path: str, audio_url: str | None = None) -> tuple[list[dict], str]:
    if RUNPOD_ENDPOINT_URL:
        return _runpod_diarize(audio_path, audio_url)
    return [], "unavailable"
