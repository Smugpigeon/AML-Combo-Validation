"""Participating-site enrollment registry.

A simple JSON-backed registry of sites participating in the prospective
validation study. Each site has:

  - site_id           : e.g. "PUMC_IRB_2026_HEM_42" (institution + IRB)
  - institution       : e.g. "Peking Union Medical College Hematology"
  - country           : ISO-3166 alpha-2
  - principal_investigator: PI name + credentials
  - irb_protocol      : IRB protocol number + approval date
  - target_n          : target enrollment for this site
  - enrolled_at       : ISO date when site joined
  - active            : boolean (False if withdrawn)

The registry is hand-edited (small N, low frequency) and version-controlled
in git. Each new patient's lock_prediction() call MUST reference a site_id
present + active in the registry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class Site:
    site_id: str
    institution: str
    country: str
    principal_investigator: str
    irb_protocol: str
    irb_approval_date: str        # ISO date
    target_n: int
    enrolled_at: str              # ISO date
    active: bool = True
    notes: str = ""


_DEFAULT_REGISTRY = Path(__file__).parents[3] / "data" / "prospective" / "site_registry.json"


def _load_registry(path: Optional[Path] = None) -> list[Site]:
    p = path or _DEFAULT_REGISTRY
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Site(**s) for s in raw.get("sites", [])]


def _save_registry(sites: list[Site], path: Optional[Path] = None) -> None:
    p = path or _DEFAULT_REGISTRY
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": "1.0",
            "last_updated": datetime.now(tz=timezone.utc).isoformat(),
            "sites": [asdict(s) for s in sites],
        }, f, ensure_ascii=False, indent=2)


def list_active_sites(path: Optional[Path] = None) -> list[Site]:
    return [s for s in _load_registry(path) if s.active]


def get_site(site_id: str, path: Optional[Path] = None) -> Optional[Site]:
    for s in _load_registry(path):
        if s.site_id == site_id:
            return s
    return None


def add_site(site: Site, path: Optional[Path] = None) -> None:
    """Add a new participating site. Idempotent on site_id."""
    sites = _load_registry(path)
    if any(s.site_id == site.site_id for s in sites):
        raise ValueError(f"Site {site.site_id!r} already in registry")
    sites.append(site)
    _save_registry(sites, path)


def deactivate_site(site_id: str, reason: str = "",
                     path: Optional[Path] = None) -> None:
    """Mark a site as withdrawn (does NOT delete history)."""
    sites = _load_registry(path)
    found = False
    for s in sites:
        if s.site_id == site_id:
            s.active = False
            s.notes = (s.notes + "; " if s.notes else "") + f"DEACTIVATED: {reason}"
            found = True
            break
    if not found:
        raise ValueError(f"Site {site_id!r} not in registry")
    _save_registry(sites, path)


def assert_site_active(site_id: str, path: Optional[Path] = None) -> Site:
    """Raise if site_id is unknown or inactive — call before lock_prediction."""
    s = get_site(site_id, path)
    if s is None:
        raise ValueError(
            f"Site {site_id!r} not in prospective-validation registry. "
            f"Add via combo_val.prospective.registry.add_site() AFTER "
            f"IRB approval is in hand."
        )
    if not s.active:
        raise ValueError(
            f"Site {site_id!r} is deactivated: {s.notes}"
        )
    return s
