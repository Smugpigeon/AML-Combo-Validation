"""Lightweight i18n for templates + error messages.

Load all JSON locale files in `app/locales/*.json` on import. Resolve
translation via dot-notation keys like `t("landing.hero_title")`. Missing
keys fall back to English, then to the raw key (so the template still
renders but surfaces the gap).

Language detection order (each step can override the previous):
  1. `?lang=en|zh` query param (lets users bookmark a forced language)
  2. `lang` cookie (persisted by /set-lang/{lang})
  3. `Accept-Language` header first primary tag
  4. default: "zh" (target audience is mainland researchers)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from fastapi import Request


_LOCALES_DIR = Path(__file__).resolve().parent / "locales"

# Loaded on module import. Keys are language codes ("en", "zh").
_STORE: dict[str, dict] = {}


def _load_all() -> None:
    """Read every *.json under locales/ into _STORE."""
    _STORE.clear()
    for path in sorted(_LOCALES_DIR.glob("*.json")):
        try:
            _STORE[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            # Don't crash the whole app if one locale file is malformed.
            _STORE[path.stem] = {}
            print(f"[i18n] Failed to load {path.name}: {e}")


_load_all()

SUPPORTED_LANGS: list[str] = list(_STORE.keys()) or ["en"]
DEFAULT_LANG: str = "zh" if "zh" in _STORE else "en"


def _walk(d: dict, parts: Iterable[str]) -> Any:
    cur: Any = d
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
    return cur


def t(key: str, lang: str = DEFAULT_LANG, **fmt: Any) -> str:
    """Translate a dot-notation key.

    >>> t("nav.home", "en")
    'Home'
    >>> t("nav.home", "zh")
    '首页'
    >>> t("missing.key", "en")
    'missing.key'
    """
    parts = key.split(".")
    val = _walk(_STORE.get(lang, {}), parts)
    if not isinstance(val, str):
        # Fallback to English
        val = _walk(_STORE.get("en", {}), parts)
    if not isinstance(val, str):
        return key
    if fmt:
        try:
            return val.format(**fmt)
        except (KeyError, IndexError, ValueError):
            return val
    return val


def detect_lang(request: Request) -> str:
    """Determine the user's language for this request."""
    # 1. Query param override
    ql = request.query_params.get("lang")
    if ql in SUPPORTED_LANGS:
        return ql

    # 2. Cookie
    cl = request.cookies.get("lang")
    if cl in SUPPORTED_LANGS:
        return cl

    # 3. Accept-Language header
    accept = request.headers.get("accept-language", "").lower()
    # Primary language tag only (e.g. "zh-CN" → "zh", "en-US" → "en")
    if accept:
        first = accept.split(",")[0].strip().split("-")[0]
        if first in SUPPORTED_LANGS:
            return first

    return DEFAULT_LANG


def make_template_context(request: Request, lang: str | None = None,
                            **extra: Any) -> dict:
    """Build the common context dict all templates need.

    Pass this into TemplateResponse (in addition to `request`).
    """
    if lang is None:
        lang = detect_lang(request)
    return {
        "lang": lang,
        "t": lambda k, **kw: t(k, lang, **kw),
        "supported_langs": SUPPORTED_LANGS,
        **extra,
    }
