"""Drug name normalization and cross-vocabulary alignment.

Problem: BeatAML uses drug names like "Quizartinib (AC220)" while DrugComb
uses "Quizartinib" or "AC220". When we build the combo predictor
(Week 3), we need to map DrugComb drug pairs into the BeatAML drug-id
space so the Baseline A single-drug features can be paired with
DrugComb synergy labels.

Strategy (stdlib only, no rapidfuzz dep):
  1. Normalize to a canonical token (lowercase, strip punct, fold
     parenthesized aliases into their own entries, strip stereochemistry
     prefixes, strip trailing version numbers).
  2. Exact match on normalized token.
  3. Use difflib.get_close_matches for fuzzy fallback (cutoff 0.85).
  4. Manual alias dictionary for known tricky cases — this is the truth-of-
     last-resort that gets curated as we encounter mismatches.

Known alignment pain points encountered in AML drug vocabs:
  - "Quizartinib (AC220)"   ↔ "Quizartinib" or "AC220"
  - "17-AAG (Tanespimycin)" ↔ "17-AAG"  or "Tanespimycin"
  - "Dovitinib (CHIR-258)"  ↔ "CHIR-258"
  - "(R)-Crizotinib"        ↔ "Crizotinib"
  - "Azacytidine"           ↔ "Azacitidine" (spelling variant)
  - "7+3 (Cytarabine, Daunorubicin)" — combination regimen, not a single drug, skip
"""

from __future__ import annotations

import re
import unicodedata
from difflib import get_close_matches
from typing import Iterable

# ---------------------------------------------------------------------------
# Manual alias table (the curator's last resort)
# ---------------------------------------------------------------------------

# Maps DrugComb-normalized name → BeatAML-normalized canonical name.
# Only used when exact + fuzzy both fail. Fill in as we find mismatches.
MANUAL_ALIASES: dict[str, str] = {
    # BeatAML spells it "Azacytidine" (y), DrugComb typically "Azacitidine" (i)
    "azacitidine": "azacytidine",
    "abt-199": "venetoclax",
    "gdc-0199": "venetoclax",
    "abt-263": "navitoclax",
    "ac220": "quizartinib",
    "chir-258": "dovitinib",
    "cp-690550": "tofacitinib",
    "tg101348": "fedratinib",
    "agi-120": "ivosidenib",
    "ag-120": "ivosidenib",
    "ag-221": "enasidenib",
    "idh-305": "ivosidenib",  # approximate — different compound but same mechanism class
    "tanespimycin": "17-aag",
    "hsp990": "hsp990",
    "nvp-hsp990": "hsp990",
    "selumetinib": "azd6244",
    "ravoxertinib": "gdc-0994",
    "bi-2536": "bi-2536",
    "zd6474": "vandetanib",
    "nsc-625487": "linifanib",
    "abt-869": "linifanib",
    "sb-525334": "sb-525334",
    "pf-2341066": "crizotinib",
    "mln8237": "alisertib",
    "gsk1120212": "trametinib",
    "mek162": "binimetinib",
    "ki-20227": "ki20227",
}

# Tokens in drug names that don't add information — safely stripped for matching.
# Patterns are applied in order.
_STRIP_PATTERNS = [
    r"\(\s*R\s*\)",          # (R)-isomer
    r"\(\s*S\s*\)",          # (S)-isomer
    r"\(\s*±\s*\)",          # racemic
    r"\b(?:mono|di|tri)?hydrochloride\b",
    r"\bsodium\b",
    r"\bcalcium\b",
    r"\bpotassium\b",
    r"\bmaleate\b",
    r"\bmesylate\b",
    r"\bsulfate\b",
    r"\bphosphate\b",
    r"\bcitrate\b",
    r"\btartrate\b",
    r"\bditartrate\b",
    r"\bditosylate\b",
    r"\btosylate\b",
    r"\bfumarate\b",
    r"\bsuccinate\b",
    r"\bacetate\b",
    r"\bhydrate\b",
    r"\banhydrous\b",
    r"\b\(TN\)\b",           # trade-name marker some DrugComb rows use
]


def _strip_nonalpha(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s)


def normalize_drug_name(name: str) -> str:
    """Canonicalize a raw drug name to a comparable token.

    - NFD unicode normalize
    - lowercase
    - remove stereochemistry / salt / hydrate suffixes
    - collapse whitespace
    - trim trailing "(XYZ)" if it looks like a code AND the stem alone is not empty
    """
    if name is None:
        return ""
    s = unicodedata.normalize("NFD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.strip().lower()

    # Strip anything after a dash-hyphen if followed by 'based' 'ligand' etc.
    for pat in _STRIP_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Strip trailing "(XYZ)" — keep only main stem IF the stem has content.
    # E.g. "quizartinib (ac220)" → "quizartinib"
    #      "(r)-crizotinib"      → "-crizotinib" from strip above → "crizotinib"
    m = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", s)
    if m:
        stem = m.group(1).strip()
        code = m.group(2).strip()
        if stem and len(stem) >= 3:
            # Keep stem
            s = stem
        elif code:
            # Use code if stem too short
            s = code

    s = s.strip(" -_()")
    return s


def parenthesized_aliases(name: str) -> list[str]:
    """For names like "Quizartinib (AC220)", return BOTH ["quizartinib", "ac220"]
    so we can index both as possible lookup keys.
    """
    aliases = {normalize_drug_name(name)}
    m = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", str(name).strip())
    if m:
        stem = m.group(1).strip()
        code = m.group(2).strip()
        if stem:
            aliases.add(normalize_drug_name(stem))
        if code:
            aliases.add(normalize_drug_name(code))
    return sorted(a for a in aliases if a)


# ---------------------------------------------------------------------------
# Aligner
# ---------------------------------------------------------------------------


class DrugNameAligner:
    """Align an arbitrary drug name (DrugComb/CCLE/etc.) to a BeatAML drug id.

    Lookup cascade:
      1. Exact match on normalized form
      2. Parenthesized alias match (BeatAML entry has alias "A (B)", index both)
      3. Manual alias table
      4. difflib fuzzy match at cutoff 0.87
      5. Return None (caller logs and drops)
    """

    def __init__(
        self,
        beataml_drug_names: Iterable[str],
        fuzzy_cutoff: float = 0.87,
        manual_aliases: dict[str, str] | None = None,
    ):
        self.fuzzy_cutoff = fuzzy_cutoff
        self.manual_aliases = {**MANUAL_ALIASES, **(manual_aliases or {})}

        # Build reverse index: normalized_alias → canonical BeatAML name
        self.norm_to_canonical: dict[str, str] = {}
        self.canonical_names = list(beataml_drug_names)

        for canonical in self.canonical_names:
            for alias in parenthesized_aliases(canonical):
                # First registration wins — catches ambiguity early
                self.norm_to_canonical.setdefault(alias, canonical)

        self._all_norms = list(self.norm_to_canonical.keys())

    def align(self, raw_name: str) -> tuple[str | None, str]:
        """Align one external name. Returns (canonical_beataml_name | None, source_of_match).

        source_of_match ∈ {'exact', 'paren_alias', 'manual', 'fuzzy', 'no_match'}.
        """
        if raw_name is None or not str(raw_name).strip():
            return None, "no_match"

        # 1 & 2: exact / paren_alias
        for alias in parenthesized_aliases(raw_name):
            if alias in self.norm_to_canonical:
                canonical = self.norm_to_canonical[alias]
                # Was this the main normalization or a paren alias on the DrugComb side?
                main_norm = normalize_drug_name(raw_name)
                src = "exact" if alias == main_norm else "paren_alias"
                return canonical, src

        # 3: manual aliases
        main_norm = normalize_drug_name(raw_name)
        if main_norm in self.manual_aliases:
            mapped = self.manual_aliases[main_norm]
            if mapped in self.norm_to_canonical:
                return self.norm_to_canonical[mapped], "manual"

        # 4: fuzzy
        matches = get_close_matches(main_norm, self._all_norms, n=1, cutoff=self.fuzzy_cutoff)
        if matches:
            return self.norm_to_canonical[matches[0]], "fuzzy"

        # 5: give up
        return None, "no_match"

    def align_many(self, raw_names: Iterable[str]) -> dict[str, tuple[str | None, str]]:
        """Vectorize align over a collection. Returns {raw: (canonical|None, source)}."""
        out = {}
        for r in raw_names:
            out[r] = self.align(r)
        return out
