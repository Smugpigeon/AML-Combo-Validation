"""BYOK LLM client — calls user's OpenAI / Anthropic key.

Two LLM features live here:
  - `summarize_report()`    — plain-English narrative of a finished report
  - `parse_clinical_text()` — smart-paste: extract structured AML patient
                              data from free-text clinical paste

Both use the user's own API key so:
  (a) the researcher pays for their own inference,
  (b) we never log the LLM prompt (contains PHI-adjacent content).

All LLM calls go through this module so we can swap providers or add
new ones (Gemini, Together, etc.) without touching callers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


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


# ---------------------------------------------------------------------------
# Smart paste — extract structured AML patient data from free-text clinical
# paste (Excel cells / report text / hospital dump / 中文 病历)
# ---------------------------------------------------------------------------


_PARSE_SYSTEM_PROMPT = """You are a clinical-data extraction assistant for an AML precision-medicine platform. Extract structured patient data from the free-form clinical text the user pastes.

Return ONLY a valid JSON object with exactly this top-level shape:
{
  "parsed": {
    "patient_label":        string,      // use existing study/sample ID if present; else leave as null
    "age":                  number|null, // years
    "sex":                  "male"|"female"|null,
    "is_initial_diagnosis": bool|null,
    "is_relapse":           bool|null,
    "prior_mds":            bool|null,
    "karyotype_text":       string|null, // preserve ISCN verbatim e.g. "46,XX,t(8;21)(q22;q22)[20]"
    "fusions":              string[],    // e.g. ["PML-RARA"]
    "mutations": [                       // array; empty if none detected
      {
        "gene":          string,         // HUGO symbol, UPPERCASE
        "variant_type":  string|null,    // "missense" | "frameshift" | "nonsense" | "ITD" | "TKD" | "splice" | "deletion" | "insertion" | null
        "vaf":           number|null,    // 0-1 fraction; if the source says "45%" store 0.45
        "is_ITD":        bool,           // FLT3 only — true if ITD mentioned
        "is_TKD":        bool,           // FLT3 only — true if TKD / D835 / I836 mentioned
        "allelic_ratio": number|null,    // FLT3-ITD allelic ratio if reported, e.g. 0.62
        "is_biallelic":  bool,           // CEBPA only — true if biallelic / both alleles
        "is_bzip":       bool,           // CEBPA only — true if bZIP / leucine zipper / b-ZIP region / in-frame bZIP mutation mentioned (per ELN 2022 / WHO 2022)
        "is_multi_hit":  bool,           // TP53 only — true if multi-hit / biallelic TP53 / 2 distinct TP53 mutations / TP53 + del(17p) mentioned (per ELN 2022)
        "protein_codon": string|null     // protein-level annotation if mentioned, e.g. "R882H" (DNMT3A), "R132H" (IDH1), "R140Q"/"R172K" (IDH2), "D835Y" (FLT3-TKD), "F691L" (FLT3 gatekeeper), "W288Cfs*12" (NPM1), "D816V" (KIT). Preserve verbatim. null if not in text.
      }
    ],
    "wbc":           number|null,        // ×10^9/L
    "platelet":     number|null,         // ×10^9/L
    "hemoglobin":    number|null,        // g/dL (if reported as g/L divide by 10)
    "ldh":           number|null,        // U/L
    "alt":           number|null,        // U/L
    "ast":           number|null,        // U/L
    "blast_pct_bm":  number|null,        // %
    "blast_pct_pb":  number|null         // %
  },
  "confidence": {
    // per-field confidence 0.0-1.0. Use dot notation for nested keys, e.g.
    // "mutations.0.gene": 0.99, "karyotype_text": 0.85.
    // 1.0 = literal match, 0.9 = clear inference, 0.7 = plausible guess,
    // <0.5 = should not have been set (prefer null instead).
  },
  "warnings": [
    // short strings flagging ambiguities, e.g.
    // "FLT3-ITD allelic ratio not explicitly stated"
    // "Two possible karyotype snippets detected — used the more recent one"
  ]
}

Extraction rules:
1. Do NOT invent data. If uncertain → null. An empty array is fine for mutations/fusions.
2. Gene symbols MUST be HUGO uppercase (FLT3, NPM1, TP53, DNMT3A, IDH1, IDH2, KMT2A, CEBPA, RUNX1, ASXL1, TET2, ...). If user writes lowercase or alias → normalize.
3. FLT3-ITD: if text contains "ITD", "internal tandem", "内部串联", or "AR =" / "allelic ratio" → set variant_type="ITD", is_ITD=true.
4. FLT3-TKD: "TKD", "D835Y/F/V", "I836", "tyrosine kinase domain" → is_TKD=true.
5. Allelic ratio: accept "AR=0.62", "AR 0.6", "allelic ratio 0.45", "高负荷/high-burden" (0.5+ implied but keep null if not quantified), ratio forms "0.62/1". Always store as decimal.
6. VAF: "45%" → 0.45; "VAF 0.42" → 0.42; "突变频率 40%" → 0.40.
7. CEBPA fields (per ELN 2022 / WHO 2022):
   - is_biallelic: set true when text says "biallelic" / "two mutations" / "双等位" / "both alleles".
   - is_bzip:      set true when text mentions "bZIP" / "leucine zipper" / "b-ZIP region" / "in-frame bZIP" / specific bZIP-domain residues. ELN 2022 made bZIP single-allele Favorable, so this is independent of biallelic.
7b. TP53 is_multi_hit: set true when text mentions "multi-hit TP53" / "biallelic TP53" / "two distinct TP53 mutations" / "TP53 + del(17p)" / TP53 with VAF ≥ 0.5. Per ELN 2022, multi-hit is an independent Adverse subtype.
7c. protein_codon (NEW): if the text gives a protein-level annotation (e.g. "DNMT3A R882H", "IDH1 R132C", "FLT3 D835Y", "KIT D816V", "NPM1 W288Cfs*12") preserve the codon part verbatim ("R882H" not "DNMT3A R882H"). Skip leading gene symbol. If text only mentions ITD or TKD with no specific codon, leave protein_codon null. Hotspots that matter:
    - DNMT3A R882* (canonical AML hotspot, independent adverse)
    - IDH1 R132* (FDA-approved Ivosidenib target)
    - IDH2 R140* / R172* (FDA-approved Enasidenib target)
    - FLT3 D835* / I836 (Type-1 FLT3i preferred), F691L (Gilt-resistance gatekeeper)
    - NPM1 exon 12 frameshift (W288Cfs, TCTG insertion)
    - KIT D816V / N822 (CBF-AML adverse modifier)
8. Sex: "男"/"M"/"male" → "male"; "女"/"F"/"female" → "female".
9. Disease stage:
   - "初诊"/"de novo"/"newly diagnosed"/"primary" → is_initial_diagnosis=true
   - "复发"/"relapse"/"refractory"/"R/R" → is_relapse=true, is_initial_diagnosis=false
   - "既往 MDS"/"prior MDS"/"MDS 转化"/"secondary AML from MDS" → prior_mds=true
10. Karyotype: preserve original verbatim. Strip only irrelevant prefixes like "核型:" or "Karyotype:".
11. Fusions: uppercase with hyphen, e.g. "PML-RARA" (not "PML/RARA"). Common ones: PML-RARA, RUNX1-RUNX1T1, CBFB-MYH11, KMT2A-MLLT3, BCR-ABL1.
12. Labs unit conversion:
    - WBC in "10^9/L" or "K/μL" → same number. "/mm³" ÷ 1000.
    - Hgb in g/L → divide by 10 to g/dL.
    - Platelet in "10^9/L" or "K/μL" → same number.
13. If you see multiple patients, only extract the first clearly identifiable one and add a warning.
14. Patient label: if text has explicit "ID", "MRN-like code", "study ID", or "Patient: XXX-123", use it. If not present, leave null — the frontend will generate one. DO NOT invent labels.
15. Output JSON must parse with standard json.loads. No markdown fences, no commentary.

Be thorough but conservative: err on the side of null + a warning, not hallucination."""


def _extract_first_json_object(text: str) -> Optional[dict]:
    """Try hard to pull a JSON object out of LLM output.

    LLMs sometimes wrap JSON in ```json ... ``` fences or prepend chatter
    despite instructions. This walks the string to find the first balanced
    { ... } block.
    """
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    # Fast path: whole string is JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Slow path: find first balanced brace block
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1
                    continue
    return None


@dataclass
class ParsedClinicalText:
    parsed: dict            # matches PatientInputJSON-ish schema
    confidence: dict        # dot-notation key → 0..1
    warnings: list[str]
    model: str
    provider: str
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    raw_response_excerpt: Optional[str] = None   # for debugging; not logged


def parse_clinical_text(
    raw_text: str,
    api_key: str,
    provider: str,
    model: Optional[str] = None,
    max_tokens: int = 2000,
    focus_patient: Optional[str] = None,
) -> ParsedClinicalText:
    """Use the user's LLM to extract structured AML patient data.

    Args:
      raw_text: Free-form clinical text (may be CSV/TSV; may contain many
        patients in rows).
      api_key / provider / model: BYOK LLM config.
      focus_patient: optional "MRN 12345" / "patient A-001" / "row 3" /
        "姓名: 张三" hint. If the paste contains multiple patients,
        tells the LLM which one to extract; otherwise ignored.

    The LLM is instructed to return a JSON object with {parsed, confidence,
    warnings}. We robustly extract the first JSON object from its response
    and validate minimally — heavy validation happens on the FastAPI endpoint
    via Pydantic.
    """
    if not raw_text or len(raw_text.strip()) < 10:
        raise ValueError("raw_text is empty or too short to parse")
    # Basic safety cap to keep prompt costs predictable
    if len(raw_text) > 20000:
        raw_text = raw_text[:20000] + "\n\n[... truncated at 20k chars ...]"

    provider = provider.lower()
    if provider not in _DEFAULT_MODEL_BY_PROVIDER:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    model = model or _DEFAULT_MODEL_BY_PROVIDER[provider]

    focus_hint = ""
    if focus_patient and focus_patient.strip():
        focus_hint = (
            f"\n\nFOCUS PATIENT: The user has indicated the target patient "
            f"is identified by: {focus_patient.strip()!r}. If this text "
            f"contains multiple patients (e.g., a CSV table with many rows), "
            f"extract ONLY that patient. If you cannot confidently find the "
            f"requested patient, return an empty parsed object and add a "
            f"warning listing the identifiers you did find.\n"
        )

    user_prompt = (
        "Clinical text/CSV to parse (may be multilingual):\n\n"
        f"---\n{raw_text}\n---"
        f"{focus_hint}\n\n"
        "Respond with the JSON object only."
    )

    if provider == "openai":
        resp = _call_openai(api_key, model, _PARSE_SYSTEM_PROMPT,
                             user_prompt, max_tokens)
    elif provider == "anthropic":
        resp = _call_anthropic(api_key, model, _PARSE_SYSTEM_PROMPT,
                                user_prompt, max_tokens)
    else:
        raise ValueError(provider)

    obj = _extract_first_json_object(resp.text)
    if not isinstance(obj, dict) or "parsed" not in obj:
        raise ValueError(
            f"LLM returned unparseable output. First 200 chars: "
            f"{resp.text[:200]!r}"
        )

    parsed = obj.get("parsed") or {}
    confidence = obj.get("confidence") or {}
    warnings = obj.get("warnings") or []
    if not isinstance(warnings, list):
        warnings = [str(warnings)]
    # Coerce non-string warnings to string
    warnings = [str(w) for w in warnings if w][:20]

    return ParsedClinicalText(
        parsed=parsed if isinstance(parsed, dict) else {},
        confidence=confidence if isinstance(confidence, dict) else {},
        warnings=warnings,
        model=resp.model, provider=resp.provider,
        tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
        raw_response_excerpt=resp.text[:500],
    )
