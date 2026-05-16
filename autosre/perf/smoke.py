"""End-to-end smoke tests for meeting-scribe backend containers.

Validates that each backend (ASR, TTS, diarization, translation) is
functional by sending a single request and checking the protocol-level
response shape. No semantic validation — just HTTP 200 + structurally
valid output.

See ``autosre perf smoke --help`` for usage.
"""

from __future__ import annotations

import base64
import io
import wave
from dataclasses import dataclass
from typing import Any

import click
import httpx

# Endpoints
ASR_URL = "http://localhost:8003"
TTS_URL = "http://localhost:8002"
DIARIZE_URL = "http://localhost:8001"
TRANSLATE_URL = "http://localhost:8010"

_TIMEOUT = 60.0  # generous for cold models


@dataclass
class SmokeResult:
    backend: str
    passed: bool
    status_code: int | None = None
    error: str | None = None
    detail: str = ""


# ── Fixtures ──────────────────────────────────────────────────


def _generate_silence_wav(duration_s: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate a silent WAV file (s16le) for smoke testing."""
    num_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * num_samples)
    return buf.getvalue()


def _generate_silence_pcm(duration_s: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate silent raw PCM (s16le) for diarization smoke test."""
    num_samples = int(sample_rate * duration_s)
    return b"\x00\x00" * num_samples


# ── Individual tests ──────────────────────────────────────────


async def _smoke_asr(client: httpx.AsyncClient) -> SmokeResult:
    """ASR: POST /v1/chat/completions with audio input."""
    wav_data = _generate_silence_wav(2.0)
    audio_b64 = base64.b64encode(wav_data).decode()

    # Get model name
    try:
        models_resp = await client.get(f"{ASR_URL}/v1/models")
        model_data = models_resp.json().get("data", [])
        model_name = model_data[0]["id"] if model_data else "Qwen/Qwen3-ASR-1.7B"
    except Exception:
        model_name = "Qwen/Qwen3-ASR-1.7B"

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "Transcribe the audio in the original spoken language. Do not translate.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    }
                ],
            },
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }

    try:
        resp = await client.post(f"{ASR_URL}/v1/chat/completions", json=payload)
        if resp.status_code != 200:
            return SmokeResult(
                "asr", passed=False, status_code=resp.status_code, error=resp.text[:500]
            )

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return SmokeResult("asr", passed=False, status_code=200, error="No choices in response")

        # Accept empty content for silence — protocol-level check only
        content = choices[0].get("message", {}).get("content")
        if content is None:
            return SmokeResult(
                "asr", passed=False, status_code=200, error="No message.content in response"
            )

        return SmokeResult(
            "asr", passed=True, status_code=200, detail=f"content_len={len(content)}"
        )
    except httpx.HTTPError as e:
        return SmokeResult("asr", passed=False, error=str(e))


async def _smoke_tts(client: httpx.AsyncClient) -> SmokeResult:
    """TTS: POST /v1/audio/speech with text input."""
    payload: dict[str, Any] = {
        "model": "qwen3-tts",
        "input": "Hello, this is a smoke test.",
        "language": "English",
        "stream": False,
        "response_format": "wav",
    }

    try:
        resp = await client.post(f"{TTS_URL}/v1/audio/speech", json=payload)
        if resp.status_code != 200:
            return SmokeResult(
                "tts", passed=False, status_code=resp.status_code, error=resp.text[:500]
            )

        body_len = len(resp.content)
        if body_len < 1000:
            return SmokeResult(
                "tts",
                passed=False,
                status_code=200,
                error=f"Response too small ({body_len} bytes) — expected audio data",
            )

        return SmokeResult("tts", passed=True, status_code=200, detail=f"audio_bytes={body_len}")
    except httpx.HTTPError as e:
        return SmokeResult("tts", passed=False, error=str(e))


async def _smoke_diarize(client: httpx.AsyncClient) -> SmokeResult:
    """Diarization: POST /v1/diarize with raw PCM."""
    pcm_data = _generate_silence_pcm(2.0)

    try:
        resp = await client.post(
            f"{DIARIZE_URL}/v1/diarize",
            content=pcm_data,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Sample-Rate": "16000",
            },
        )
        if resp.status_code != 200:
            return SmokeResult(
                "diarization", passed=False, status_code=resp.status_code, error=resp.text[:500]
            )

        data = resp.json()
        if "segments" not in data:
            return SmokeResult(
                "diarization", passed=False, status_code=200, error="No 'segments' key in response"
            )
        if "num_speakers" not in data:
            return SmokeResult(
                "diarization",
                passed=False,
                status_code=200,
                error="No 'num_speakers' key in response",
            )

        return SmokeResult(
            "diarization",
            passed=True,
            status_code=200,
            detail=f"segments={len(data['segments'])}, speakers={data['num_speakers']}",
        )
    except httpx.HTTPError as e:
        return SmokeResult("diarization", passed=False, error=str(e))


async def _smoke_translation(client: httpx.AsyncClient) -> SmokeResult:
    """Translation: POST /v1/chat/completions with JA→EN."""
    # Get model name
    try:
        models_resp = await client.get(f"{TRANSLATE_URL}/v1/models")
        model_data = models_resp.json().get("data", [])
        model_name = model_data[0]["id"] if model_data else "Qwen/Qwen3.6-35B-A3B-FP8"
    except Exception:
        model_name = "Qwen/Qwen3.6-35B-A3B-FP8"

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "Translate the following Japanese text to English. Output only the translation.",
            },
            {
                "role": "user",
                "content": "\u4f1a\u8b70\u3092\u59cb\u3081\u307e\u3057\u3087\u3046",  # 会議を始めましょう
            },
        ],
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
        # Disable thinking/reasoning to get direct output — matches
        # meeting-scribe's translation path (enable_thinking=False).
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        resp = await client.post(f"{TRANSLATE_URL}/v1/chat/completions", json=payload)
        if resp.status_code != 200:
            return SmokeResult(
                "translation", passed=False, status_code=resp.status_code, error=resp.text[:500]
            )

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return SmokeResult(
                "translation", passed=False, status_code=200, error="No choices in response"
            )

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            return SmokeResult(
                "translation", passed=False, status_code=200, error="Empty translation content"
            )

        return SmokeResult(
            "translation", passed=True, status_code=200, detail=f"content_len={len(content)}"
        )
    except httpx.HTTPError as e:
        return SmokeResult("translation", passed=False, error=str(e))


# ── Runner ────────────────────────────────────────────────────


async def run_smoke() -> list[SmokeResult]:
    """Run all smoke tests and return results."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = []
        tests = [
            ("ASR", _smoke_asr),
            ("TTS", _smoke_tts),
            ("Diarization", _smoke_diarize),
            ("Translation", _smoke_translation),
        ]
        for label, test_fn in tests:
            click.echo(f"  {label}... ", nl=False)
            result = await test_fn(client)
            if result.passed:
                click.secho(f"PASS ({result.detail})", fg="green")
            else:
                error_detail = result.error or "unknown"
                click.secho(f"FAIL ({error_detail})", fg="red")
            results.append(result)
        return results


def render_smoke_stdout(results: list[SmokeResult]) -> None:
    click.echo()
    click.secho("autosre perf smoke — summary", bold=True)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    color = "green" if passed == total else "red"
    click.secho(f"  {passed}/{total} backends passed", fg=color)
    click.echo()
