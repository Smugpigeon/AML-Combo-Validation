"""Parse AML karyotype ISCN strings into structured flags.

Conservative parser — aims to correctly flag:
  - complex karyotype (≥3 abnormalities per ICC 2022)
  - monosomy 5 / monosomy 7 / del(5q) / del(7q)
  - del(17p) / TP53-associated del(17)
  - t(8;21), inv(16), t(15;17) (for core-binding factor + APL recognition)
  - Normal karyotype

Handles common inputs like:
  "46,XX,t(8;21)(q22;q22)[20]"
  "47,XY,+8,del(7q)(q22q36)[15]/46,XY[5]"
  "46,XX[cp3]/45,X,-Y[18]"
  "complex karyotype"

Missing or "NK" (not known) → flags all zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_ABERRATION_TOKENS = (
    "t(", "del(", "inv(", "add(", "der(", "dup(", "ins(", "idic(", "dic(", "r(",
)


@dataclass(frozen=True)
class KaryotypeFlags:
    complex: int = 0                # ≥ 3 distinct abnormalities
    monosomy_5_or_7: int = 0
    del_5q: int = 0
    del_7q: int = 0
    del_17p: int = 0
    trisomy_8: int = 0
    t_8_21: int = 0                 # RUNX1-RUNX1T1
    inv_16: int = 0                 # CBFB-MYH11
    t_15_17: int = 0                # PML-RARA
    t_9_11: int = 0                 # KMT2A-MLLT3
    normal: int = 0
    as_dict_values: tuple = ()      # legacy

    def as_dict(self) -> dict[str, int]:
        return {
            "karyo_complex": self.complex,
            "karyo_monosomy_5_or_7": self.monosomy_5_or_7,
            "karyo_del_5q": self.del_5q,
            "karyo_del_7q": self.del_7q,
            "karyo_del_17p": self.del_17p,
            "karyo_trisomy_8": self.trisomy_8,
            "karyo_t_8_21": self.t_8_21,
            "karyo_inv_16": self.inv_16,
            "karyo_t_15_17": self.t_15_17,
            "karyo_t_9_11": self.t_9_11,
            "karyo_normal": self.normal,
        }


def parse_karyotype(text: str | None) -> KaryotypeFlags:
    if not text or not text.strip() or str(text).lower() in {"nk", "nan", "none", "not done"}:
        return KaryotypeFlags()

    s = str(text)
    s_low = s.lower()

    # --- Count aberrations
    # Conservative: sum occurrences of structural aberration prefixes across all clones
    n_aberrations = sum(s_low.count(tok) for tok in _ABERRATION_TOKENS)
    # Also count numerical aberrations like "+8", "-7" (but not "-Y" alone = sex chromosome loss)
    #   +N, -N where N in 1..22
    num_aberrations = len(re.findall(r"[+\-][12]?\d(?=[,\[\s\]]|$)", s))
    total_aberrations = n_aberrations + num_aberrations

    flags = {
        "complex": int(total_aberrations >= 3),
        "monosomy_5_or_7": int(
            bool(re.search(r"-5(?![0-9])|-7(?![0-9])|monosomy\s*5|monosomy\s*7",
                            s, re.IGNORECASE))
        ),
        "del_5q": int(bool(re.search(r"del\(5", s, re.IGNORECASE))),
        "del_7q": int(bool(re.search(r"del\(7", s, re.IGNORECASE))),
        "del_17p": int(bool(re.search(r"del\(17", s, re.IGNORECASE))),
        "trisomy_8": int(bool(re.search(r"\+8(?![0-9])|trisomy\s*8", s, re.IGNORECASE))),
        "t_8_21": int(bool(re.search(r"t\(8;21\)", s, re.IGNORECASE))),
        "inv_16": int(bool(re.search(r"inv\(16\)|t\(16;16\)", s, re.IGNORECASE))),
        "t_15_17": int(bool(re.search(r"t\(15;17\)", s, re.IGNORECASE))),
        "t_9_11": int(bool(re.search(r"t\(9;11\)", s, re.IGNORECASE))),
    }
    # Normal karyotype: something like "46,XX" or "46,XY" with no aberrations
    flags["normal"] = int(
        bool(re.match(r"^\s*46,\s*x[xy]\s*(\[|\s|$)", s_low)) and total_aberrations == 0
    )
    return KaryotypeFlags(**flags)
