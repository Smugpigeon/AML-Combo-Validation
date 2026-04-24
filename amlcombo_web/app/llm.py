"""Minimal BYOK LLM client — calls user's OpenAI / Anthropic key.

The ONLY LLM feature in the MVP is "generate a plain-English summary of
this patient's clinical report" — a ~500-word narrative the researcher
can paste into their notes. Uses the user's own API key so:
  (a) the researcher pays for their own inference,
  (b) we never log the LLM prompt (contains PHI-adjacent content).

All LLM calls go through this module so we can swap providers or add
new ones (Gemini, Together, etc.) without touching callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


_DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
}


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None


def summarize_report(
    report_markdown: str,
    api_key: str,
    provider: str,
    model: Optional[str] = None,
    max_tokens: int = 800,
) -> LLMResponse:
    """Use the researcher's LLM key to generate a plain-English summary.

    The prompt is fixed and deliberately conservative — we ask the model
    to paraphrase, NOT to add clinical recommendations beyond what's
    already in the kit's output.
    """
    provider = provider.lower()
    if provider not in _DEFAULT_MODEL_BY_PROVIDER:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    model = model or _DEFAULT_MODEL_BY_PROVIDER[provider]

    system_prompt = (
        "You are a helpful assistant summarizing AML precision-medicine "
        "reports for the researcher who authored them. Paraphrase the "
        "findings faithfully in plain English (English + simplified "
        "Chinese bilingual is fine). Do NOT invent or extrapolate new "
        "clinical recommendations. If the report already says 'research "
        "use only', preserve that disclaimer. Keep it under 400 words."
    )
    user_prompt = (
        "Summarize the following AML patient report in 3 paragraphs — "
        "(1) patient snapshot + ELN risk, (2) top treatment rationale, "
        "(3) key caveats / next steps. Quote the top regimen name verbatim.\n\n"
        "---\n" + report_markdown + "\n---"
    )

    if provider == "openai":
        return _call_openai(api_key, model, system_prompt, user_prompt,
                             max_tokens)
    elif provider == "anthropic":
        return _call_anthropic(api_key, model, system_prompt, user_prompt,
                                max_tokens)
    raise ValueError(provider)  # unreachable


def _call_openai(api_key: str, model: str, system: str, user: str,
                  max_tokens: int) -> LLMResponse:
    import httpx

    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    payload = resp.json()
    return LLMResponse(
        text=payload["choices"][0]["message"]["content"],
        model=model, provider="openai",
        tokens_in=payload.get("usage", {}).get("prompt_tokens"),
        tokens_out=payload.get("usage", {}).get("completion_tokens"),
    )


def _call_anthropic(api_key: str, model: str, system: str, user: str,
                     max_tokens: int) -> LLMResponse:
    import httpx

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    payload = resp.json()
    # Anthropic returns content as list of content blocks
    text = "".join(b.get("text", "") for b in payload.get("content", [])
                    if b.get("type") == "text")
    return LLMResponse(
        text=text, model=model, provider="anthropic",
        tokens_in=payload.get("usage", {}).get("input_tokens"),
        tokens_out=payload.get("usage", {}).get("output_tokens"),
    )
