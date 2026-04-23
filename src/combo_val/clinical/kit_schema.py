"""Kit input schema — the contract a new AML patient's data must satisfy.

Designed to match what a real clinical workup produces for an AML patient:
  - Mutation panel (NGS report): gene + variant type + VAF; FLT3-ITD calls
  - Cytogenetics: karyotype string, any fusion calls
  - CBC / chemistry: WBC, platelet, Hb, LDH, ALT, AST, albumin, creatinine
  - Demographics + disease state

RNA-Seq is provided separately as a gene_symbol → count Series.

No field is required; missing fields are imputed with BeatAML medians (or
zero for binary) so the kit degrades gracefully on incomplete intake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MutationCall:
    """One entry in an NGS mutation panel report."""
    gene: str                              # HUGO symbol
    variant_type: Optional[str] = None     # "missense" | "frameshift" | "nonsense" | "splice" | "ITD" | "TKD" | ...
    vaf: Optional[float] = None            # allele fraction [0, 1]
    is_ITD: bool = False                   # FLT3 specifically
    is_TKD: bool = False                   # FLT3 specifically
    allelic_ratio: Optional[float] = None  # FLT3-ITD allelic ratio (ITD reads / WT reads)
    is_biallelic: bool = False             # CEBPA specifically


@dataclass
class KitInput:
    """Full input for one new patient going through the kit."""
    patient_id: str
    # --- Mutation panel ---
    mutations: list[MutationCall] = field(default_factory=list)
    # --- Cytogenetics ---
    karyotype_text: Optional[str] = None    # e.g. "46,XX,t(8;21)(q22;q22)[20]"
    fusions: list[str] = field(default_factory=list)   # e.g. ["PML-RARA", "KMT2A-MLLT3"]
    # --- Labs ---
    wbc: Optional[float] = None             # 10^9/L
    platelet: Optional[float] = None        # 10^9/L
    hemoglobin: Optional[float] = None      # g/dL
    ldh: Optional[float] = None             # U/L
    alt: Optional[float] = None             # U/L
    ast: Optional[float] = None             # U/L
    albumin: Optional[float] = None         # g/dL
    creatinine: Optional[float] = None      # mg/dL
    blast_pct_bm: Optional[float] = None    # %
    blast_pct_pb: Optional[float] = None    # %
    # --- Demographics / disease state ---
    age: Optional[float] = None
    sex: Optional[str] = None               # "male" | "female" | None
    is_relapse: Optional[bool] = None
    prior_mds: Optional[bool] = None
    prior_chemo: Optional[bool] = None
    is_initial_diagnosis: Optional[bool] = None
    eln2017: Optional[str] = None           # precomputed; if None we compute from karyotype+mutations
    # --- Intent (for kit report framing; not a model feature) ---
    intent_comment: Optional[str] = None


@dataclass
class KitOutput:
    """Structured output the kit reports back to the clinical team."""
    patient_id: str
    predicted_eln2017: str                  # computed or passed through
    top_combinations: list[dict]            # [{rank, drug1, drug2, predicted_auc, mech_score, …}]
    top_single_drugs: list[dict]            # [{rank, drug, predicted_auc}]
    top_regimens: list[dict]                # Route C: trial-evidence-based regimens w/ published CR/OS
    driver_flags: dict                      # {FLT3_ITD: bool, NPM1: bool, IDH2: bool, ...}
    fitness_flag: str                       # "fit_for_intensive" | "unfit"
    cautions: list[str]                     # e.g. "TLS risk with Venetoclax; baseline LDH elevated"
    confidence_notes: list[str]             # e.g. "RNA-Seq gene coverage 4/5000 below training support"
