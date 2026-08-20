"""Hierarchical single-cell state atlas for the AML virtual-cell pilot.

The module deliberately models observed baseline state. It does not infer a
causal post-treatment transcriptome and it never treats expression clusters as
genetically proven clones. The outputs are intended to provide a reproducible
state representation for later perturbation models and ex-vivo validation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_mutual_info_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

CELL_TYPE_MARKERS: dict[str, tuple[str, ...]] = {
    "T_cell": ("CD3D", "CD3E", "TRAC", "LCK", "IL7R", "LTB"),
    "NK_cell": ("NKG7", "GNLY", "PRF1", "KLRD1", "CTSW", "GZMB"),
    "B_cell": ("CD79A", "MS4A1", "CD37", "CD74", "CD22", "CD19"),
    "plasma_cell": ("MZB1", "JCHAIN", "SDC1", "IGKC", "DERL3"),
    "monocyte_macrophage": (
        "LST1",
        "CTSS",
        "FCER1G",
        "TYROBP",
        "LILRB1",
        "CTSD",
        "S100A8",
        "S100A9",
    ),
    "dendritic_cell": ("FCER1A", "CD1C", "CLEC10A", "CST3", "HLA-DRA"),
    "granulocytic": ("CSF3R", "FCGR3B", "ELANE", "MPO", "AZU1", "PRTN3"),
    "erythroid": ("HBB", "HBA1", "HBA2", "GYPA", "AHSP", "ALAS2"),
    "megakaryocyte": ("PPBP", "PF4", "GP9", "ITGA2B", "NRGN"),
    "mast_cell": ("TPSAB1", "TPSB2", "KIT", "CPA3", "MS4A2"),
    "stem_progenitor": ("CD34", "SPINK2", "HOPX", "GATA2", "KIT", "MPL", "MEIS1"),
}


PROGRAM_MARKERS: dict[str, tuple[str, ...]] = {
    "stem_progenitor": (
        "CD34",
        "SPINK2",
        "HOPX",
        "GATA2",
        "KIT",
        "MPL",
        "MEIS1",
        "HOXA9",
    ),
    "proliferation": ("MKI67", "TOP2A", "PCNA", "TYMS", "UBE2C", "CENPF", "CDK1"),
    "anti_apoptotic_escape": ("BCL2", "MCL1", "BCL2L1", "BCL2A1", "XIAP", "BIRC5"),
    "apoptosis_priming": ("BCL2L11", "PMAIP1", "BBC3", "BAX", "BAK1", "BID"),
    "interferon_response": ("ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "OAS1", "STAT1"),
    "inflammatory_jak_stat": ("IL6R", "JAK1", "JAK2", "STAT3", "SOCS3", "CXCL8", "FOS", "JUN"),
    "hypoxia": ("HIF1A", "VEGFA", "CA9", "LDHA", "SLC2A1", "BNIP3"),
    "glycolysis": ("HK2", "PFKP", "ALDOA", "GAPDH", "PGK1", "ENO1", "PKM", "LDHA"),
    "oxidative_phosphorylation": (
        "NDUFS1",
        "NDUFA9",
        "SDHA",
        "UQCRC1",
        "COX5A",
        "ATP5F1A",
        "TOMM20",
    ),
    "dna_damage_repair": ("RAD51", "BRCA1", "BRCA2", "CHEK1", "ATR", "FANCD2", "MSH2", "PARP1"),
    "mapk_feedback": ("DUSP6", "EGR1", "FOS", "JUN", "SPRY2", "ETV4", "ETV5"),
    "oxidative_stress": ("NQO1", "HMOX1", "GCLC", "GCLM", "TXNRD1", "SOD2"),
    "p53_stress": ("CDKN1A", "GADD45A", "MDM2", "BBC3", "BAX", "DDB2"),
    "myeloid_differentiation": ("LYZ", "CTSS", "FCER1G", "TYROBP", "LST1", "CSF1R"),
}


CONFIDENT_NORMAL_LINEAGES = frozenset(
    {"T_cell", "NK_cell", "B_cell", "plasma_cell", "erythroid", "megakaryocyte"}
)


@dataclass(frozen=True)
class StateAtlasConfig:
    """Reproducible settings for cross-patient expression-state prototypes."""

    n_components: int = 30
    n_states: int = 12
    random_state: int = 20260820
    batch_size: int = 2048
    stability_repeats: int = 4


@dataclass
class StateAtlasResult:
    """Fitted cross-patient state representation and assignment diagnostics."""

    coordinates: np.ndarray
    standardized_coordinates: np.ndarray
    state_ids: np.ndarray
    assignment_confidence: np.ndarray
    nearest_distance: np.ndarray
    raw_state_labels: np.ndarray
    stable_label_mapping: dict[int, str]
    initialization_stability: np.ndarray
    initialization_ami: tuple[float, ...]
    svd: TruncatedSVD
    scaler: StandardScaler
    kmeans: MiniBatchKMeans


@dataclass
class GeneSetReference:
    """Frozen cohort statistics for comparable marker/program inference."""

    present_genes: tuple[str, ...]
    gene_median: np.ndarray
    gene_scale: np.ndarray
    gene_sets: dict[str, tuple[str, ...]]
    minimum_genes: int = 2


@dataclass
class ReferenceShiftModel:
    """Frozen small-cohort center/scale for descriptive shift monitoring."""

    feature_columns: tuple[str, ...]
    feature_median: np.ndarray
    feature_scale: np.ndarray
    usable_features: np.ndarray
    reference_distances: np.ndarray


def _canonical_gene(value: object) -> str:
    return str(value).strip().upper()


def normalize_log1p_to_10k(matrix: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    """Re-normalize an existing log1p matrix to 10,000 counts per cell.

    STATE preprocessing can normalize each source sample to its median library
    size. CellTypist and the module scorer expect log1p values at a 10,000-count
    target. Reversing log1p preserves the within-cell proportions and avoids
    needing to retain a second raw-count object in memory.
    """

    if sparse.issparse(matrix):
        linear = matrix.tocsr(copy=True).astype(np.float32)
        linear.data = np.expm1(linear.data)
    else:
        dense = np.asarray(matrix, dtype=np.float32)
        linear = sparse.csr_matrix(np.expm1(dense))
    totals = np.asarray(linear.sum(axis=1)).ravel()
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError("Every cell must have a positive finite library size")
    normalized = sparse.diags(10000.0 / totals) @ linear
    normalized = normalized.tocsr().astype(np.float32)
    normalized.data = np.log1p(normalized.data)
    return normalized


def robust_zscore_columns(values: np.ndarray, clip: float = 8.0) -> np.ndarray:
    """Robustly standardize genes across the reference cohort."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Expected a two-dimensional cell-by-gene matrix")
    median = np.median(array, axis=0)
    mad = np.median(np.abs(array - median), axis=0) * 1.4826
    standard = np.std(array, axis=0)
    scale = np.where(mad >= 1e-4, mad, standard)
    scale = np.where(scale >= 1e-4, scale, 1.0)
    zscore = (array - median) / scale
    return np.clip(zscore, -clip, clip).astype(np.float32)


def _fit_robust_gene_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    median = np.median(array, axis=0)
    mad = np.median(np.abs(array - median), axis=0) * 1.4826
    standard = np.std(array, axis=0)
    scale = np.where(mad >= 1e-4, mad, standard)
    scale = np.where(scale >= 1e-4, scale, 1.0)
    return median.astype(np.float32), scale.astype(np.float32)


def _extract_reference_genes(
    log_expression: sparse.spmatrix | np.ndarray,
    gene_names: Sequence[object],
    requested_genes: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    lookup: dict[str, int] = {}
    for index, gene in enumerate(gene_names):
        lookup.setdefault(_canonical_gene(gene), index)
    present = [gene for gene in requested_genes if gene in lookup]
    if not present:
        raise ValueError("None of the requested marker genes are present")
    selected = log_expression[:, [lookup[gene] for gene in present]]
    if sparse.issparse(selected):
        selected = selected.toarray()
    return np.asarray(selected, dtype=np.float32), present


def _scores_from_reference_z(
    selected_z: np.ndarray,
    present_genes: Sequence[str],
    gene_sets: Mapping[str, Sequence[str]],
    minimum_genes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_lookup = {gene: index for index, gene in enumerate(present_genes)}
    scores: dict[str, np.ndarray] = {}
    coverage_rows: list[dict[str, object]] = []
    for name, genes in gene_sets.items():
        canonical = [_canonical_gene(gene) for gene in genes]
        available = [gene for gene in canonical if gene in selected_lookup]
        missing = [gene for gene in canonical if gene not in selected_lookup]
        if len(available) < minimum_genes:
            score = np.full(selected_z.shape[0], np.nan, dtype=np.float32)
        else:
            columns = [selected_lookup[gene] for gene in available]
            score = selected_z[:, columns].mean(axis=1).astype(np.float32)
        scores[name] = score
        coverage_rows.append(
            {
                "module": name,
                "requested_genes": len(canonical),
                "available_genes": len(available),
                "coverage_fraction": len(available) / max(len(canonical), 1),
                "genes_used": ";".join(available),
                "genes_missing": ";".join(missing),
                "score_available": len(available) >= minimum_genes,
            }
        )
    return pd.DataFrame(scores), pd.DataFrame(coverage_rows)


def fit_gene_set_reference(
    log_expression: sparse.spmatrix | np.ndarray,
    gene_names: Sequence[object],
    gene_sets: Mapping[str, Sequence[str]],
    *,
    minimum_genes: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, GeneSetReference]:
    """Fit and return a frozen gene-wise reference for later patients."""

    canonical_sets = {
        str(name): tuple(_canonical_gene(gene) for gene in genes)
        for name, genes in gene_sets.items()
    }
    requested = sorted({gene for genes in canonical_sets.values() for gene in genes})
    selected, present = _extract_reference_genes(log_expression, gene_names, requested)
    median, scale = _fit_robust_gene_statistics(selected)
    selected_z = np.clip((selected - median) / scale, -8.0, 8.0).astype(np.float32)
    scores, coverage = _scores_from_reference_z(
        selected_z, present, canonical_sets, minimum_genes
    )
    reference = GeneSetReference(
        present_genes=tuple(present),
        gene_median=median,
        gene_scale=scale,
        gene_sets=canonical_sets,
        minimum_genes=minimum_genes,
    )
    return scores, coverage, reference


def transform_gene_set_reference(
    log_expression: sparse.spmatrix | np.ndarray,
    gene_names: Sequence[object],
    reference: GeneSetReference,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score new cells on the exact frozen scale used by the reference cohort."""

    selected, present = _extract_reference_genes(
        log_expression, gene_names, reference.present_genes
    )
    reference_lookup = {gene: index for index, gene in enumerate(reference.present_genes)}
    median = np.asarray(
        [reference.gene_median[reference_lookup[gene]] for gene in present], dtype=np.float32
    )
    scale = np.asarray(
        [reference.gene_scale[reference_lookup[gene]] for gene in present], dtype=np.float32
    )
    selected_z = np.clip((selected - median) / scale, -8.0, 8.0).astype(np.float32)
    return _scores_from_reference_z(
        selected_z,
        present,
        reference.gene_sets,
        reference.minimum_genes,
    )


def score_gene_sets(
    log_expression: sparse.spmatrix | np.ndarray,
    gene_names: Sequence[object],
    gene_sets: Mapping[str, Sequence[str]],
    *,
    minimum_genes: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score curated modules using robust gene-wise expression z-scores.

    The score is descriptive relative activity within the supplied cohort. It
    is not pathway flux and should not be interpreted as direct phosphoprotein
    activation.
    """

    scores, coverage, _ = fit_gene_set_reference(
        log_expression,
        gene_names,
        gene_sets,
        minimum_genes=minimum_genes,
    )
    return scores, coverage


def _softmax(values: np.ndarray, temperature: float = 0.75) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / max(temperature, 1e-6)
    scaled = scaled - np.nanmax(scaled, axis=1, keepdims=True)
    exp = np.exp(np.nan_to_num(scaled, nan=-50.0))
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def classify_marker_lineage(
    cell_type_scores: pd.DataFrame,
    *,
    minimum_confidence: float = 0.30,
    minimum_margin: float = 0.04,
) -> pd.DataFrame:
    """Convert marker-module scores into broad lineage calls with uncertainty."""

    if cell_type_scores.empty:
        raise ValueError("Cell-type score table cannot be empty")
    columns = list(cell_type_scores.columns)
    probabilities = _softmax(cell_type_scores.to_numpy(dtype=float))
    order = np.argsort(probabilities, axis=1)
    top_index = order[:, -1]
    second_index = order[:, -2] if len(columns) > 1 else top_index
    top_probability = probabilities[np.arange(len(probabilities)), top_index]
    second_probability = probabilities[np.arange(len(probabilities)), second_index]
    margin = top_probability - second_probability
    labels = np.asarray([columns[index] for index in top_index], dtype=object)
    ambiguous = (top_probability < minimum_confidence) | (margin < minimum_margin)
    labels[ambiguous] = "ambiguous"
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)
    entropy /= math.log(max(len(columns), 2))
    return pd.DataFrame(
        {
            "marker_lineage": labels,
            "marker_lineage_top": [columns[index] for index in top_index],
            "marker_lineage_confidence": top_probability,
            "marker_lineage_margin": margin,
            "marker_lineage_entropy": entropy,
        },
        index=cell_type_scores.index,
    )


def map_celltypist_label(label: object) -> str:
    """Map CellTypist fine immune labels to broad auditable lineages."""

    text = str(label).strip().lower()
    if not text or text == "nan":
        return "unavailable"
    if any(token in text for token in ("platelet", "megakaryo")):
        return "megakaryocyte"
    if any(token in text for token in ("eryth", "red blood")):
        return "erythroid"
    if any(token in text for token in ("plasma", "plasmablast")):
        return "plasma_cell"
    if any(token in text for token in ("natural killer", " nk", "nk ", "nk cell")) or text.startswith("nk"):
        return "NK_cell"
    if any(token in text for token in (" t cell", "t cell", "helper t", "cytotoxic t", "regulatory t", "tem/", "tcm/", "naive t")):
        return "T_cell"
    if any(token in text for token in (" b cell", "b cell", "naive b", "memory b")):
        return "B_cell"
    if any(token in text for token in ("dendritic", "dc1", "dc2", "dc3", "pdc")):
        return "dendritic_cell"
    if any(token in text for token in ("monocyte", "macrophage", "mono-mac")):
        return "monocyte_macrophage"
    if any(token in text for token in ("neutroph", "granul", "myelocyte")):
        return "granulocytic"
    if "mast" in text:
        return "mast_cell"
    if any(token in text for token in ("progenitor", "stem cell", "hsc", "mpp")):
        return "stem_progenitor"
    return "other_or_unresolved"


def combine_lineage_annotations(
    marker_calls: pd.DataFrame,
    celltypist_labels: Sequence[object] | None = None,
    celltypist_confidence: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Combine independent lineage evidence without hiding disagreements."""

    output = marker_calls.copy()
    n_cells = len(output)
    if celltypist_labels is None:
        fine = np.full(n_cells, "unavailable", dtype=object)
        broad = np.full(n_cells, "unavailable", dtype=object)
        ct_confidence = np.zeros(n_cells, dtype=float)
    else:
        fine = np.asarray(celltypist_labels, dtype=object)
        if len(fine) != n_cells:
            raise ValueError("CellTypist labels do not align with marker calls")
        broad = np.asarray([map_celltypist_label(label) for label in fine], dtype=object)
        if celltypist_confidence is None:
            ct_confidence = np.zeros(n_cells, dtype=float)
        else:
            ct_confidence = np.asarray(celltypist_confidence, dtype=float)
            if len(ct_confidence) != n_cells:
                raise ValueError("CellTypist confidence does not align with labels")

    consensus: list[str] = []
    status: list[str] = []
    confidence: list[float] = []
    for index, marker_row in enumerate(output.itertuples()):
        marker = str(marker_row.marker_lineage)
        marker_top = str(marker_row.marker_lineage_top)
        marker_conf = float(marker_row.marker_lineage_confidence)
        ct_label = str(broad[index])
        ct_conf = float(ct_confidence[index])
        ct_usable = ct_label not in {"unavailable", "other_or_unresolved"} and ct_conf >= 0.50
        marker_usable = marker != "ambiguous" and marker_conf >= 0.30
        if ct_usable and marker_usable and ct_label == marker:
            consensus.append(ct_label)
            status.append("concordant")
            confidence.append(1.0 - (1.0 - ct_conf) * (1.0 - marker_conf))
        elif ct_usable and marker_usable and ct_label != marker:
            consensus.append("ambiguous")
            status.append("discordant")
            confidence.append(min(0.49, max(ct_conf, marker_conf) * 0.5))
        elif ct_usable:
            consensus.append(ct_label)
            status.append("celltypist_supported")
            confidence.append(ct_conf)
        elif marker_usable:
            consensus.append(marker)
            status.append("marker_supported")
            confidence.append(marker_conf)
        else:
            consensus.append(marker_top if marker_conf >= 0.25 else "ambiguous")
            status.append("low_confidence")
            confidence.append(max(marker_conf, ct_conf) * 0.5)

    output["celltypist_fine_label"] = fine
    output["celltypist_broad_lineage"] = broad
    output["celltypist_confidence"] = ct_confidence
    output["consensus_lineage"] = consensus
    output["lineage_evidence_status"] = status
    output["lineage_confidence"] = np.clip(confidence, 0.0, 1.0)
    return output


def transcriptomic_suspicion(
    program_scores: pd.DataFrame,
    lineages: Sequence[object],
    lineage_confidence: Sequence[float],
) -> pd.DataFrame:
    """Build a non-diagnostic AML-like state index with explicit evidence labels."""

    required = {
        "stem_progenitor",
        "proliferation",
        "anti_apoptotic_escape",
        "dna_damage_repair",
        "myeloid_differentiation",
    }
    missing = required - set(program_scores.columns)
    if missing:
        raise ValueError(f"Missing programs required for suspicion index: {sorted(missing)}")
    score = (
        0.55 * program_scores["stem_progenitor"].fillna(0).to_numpy(dtype=float)
        + 0.20 * program_scores["proliferation"].fillna(0).to_numpy(dtype=float)
        + 0.15 * program_scores["anti_apoptotic_escape"].fillna(0).to_numpy(dtype=float)
        + 0.10 * program_scores["dna_damage_repair"].fillna(0).to_numpy(dtype=float)
        + 0.10 * program_scores["myeloid_differentiation"].fillna(0).to_numpy(dtype=float)
    )
    lineage_array = np.asarray(lineages, dtype=object)
    lineage_conf = np.asarray(lineage_confidence, dtype=float)
    normal_anchor = np.asarray(
        [lineage in CONFIDENT_NORMAL_LINEAGES for lineage in lineage_array], dtype=bool
    ) & (lineage_conf >= 0.55)
    score = score - normal_anchor.astype(float) * 2.0
    index = 1.0 / (1.0 + np.exp(-np.clip(score, -12.0, 12.0)))
    evidence = np.full(len(index), "indeterminate_expression_state", dtype=object)
    evidence[normal_anchor] = "confident_normal_lineage_anchor"
    high = (~normal_anchor) & (index >= 0.70)
    intermediate = (~normal_anchor) & (index >= 0.55) & ~high
    evidence[high] = "high_transcriptomic_suspicion"
    evidence[intermediate] = "intermediate_transcriptomic_suspicion"
    return pd.DataFrame(
        {
            "transcriptomic_suspicion_index": index,
            "transcriptomic_suspicion_evidence": evidence,
            "confident_normal_lineage_anchor": normal_anchor,
        },
        index=program_scores.index,
    )


def calibrate_probability_to_prevalence(
    probability: Sequence[float], target_fraction: float
) -> np.ndarray:
    """Shift logits so the mean probability matches a known prevalence prior."""

    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    target = float(target_fraction)
    if not 0.0 <= target <= 1.0:
        raise ValueError("Target prevalence must be in [0, 1]")
    if target <= 0.0:
        return np.zeros_like(values)
    if target >= 1.0:
        return np.ones_like(values)
    logits = np.log(values / (1.0 - values))
    low, high = -20.0, 20.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        shifted = 1.0 / (1.0 + np.exp(-np.clip(logits + midpoint, -30.0, 30.0)))
        if float(shifted.mean()) < target:
            low = midpoint
        else:
            high = midpoint
    offset = (low + high) / 2.0
    return 1.0 / (1.0 + np.exp(-np.clip(logits + offset, -30.0, 30.0)))


def _initialization_stability(
    standardized: np.ndarray,
    primary_labels: np.ndarray,
    config: StateAtlasConfig,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Measure sensitivity to clustering initialization without relabeling artifacts."""

    if config.stability_repeats <= 0:
        return np.ones(len(primary_labels), dtype=np.float32), tuple()
    matched = np.zeros(len(primary_labels), dtype=np.float32)
    ami_values: list[float] = []
    label_order = np.arange(config.n_states)
    for repeat in range(config.stability_repeats):
        alternative_model = MiniBatchKMeans(
            n_clusters=config.n_states,
            random_state=config.random_state + 1009 * (repeat + 1),
            batch_size=config.batch_size,
            n_init=5,
            reassignment_ratio=0.01,
        )
        alternative = alternative_model.fit_predict(standardized)
        contingency = confusion_matrix(
            primary_labels, alternative, labels=label_order
        )
        primary_index, alternative_index = linear_sum_assignment(-contingency)
        mapping = {
            int(alternative_label): int(primary_label)
            for primary_label, alternative_label in zip(
                primary_index, alternative_index, strict=True
            )
        }
        aligned = np.asarray([mapping[int(label)] for label in alternative], dtype=int)
        matched += (aligned == primary_labels).astype(np.float32)
        ami_values.append(float(adjusted_mutual_info_score(primary_labels, alternative)))
    return matched / config.stability_repeats, tuple(ami_values)


def fit_state_atlas(
    hvg_matrix: sparse.spmatrix | np.ndarray,
    config: StateAtlasConfig,
) -> StateAtlasResult:
    """Fit reproducible cross-patient transcriptomic state prototypes."""

    n_cells, n_features = hvg_matrix.shape
    if n_cells < config.n_states * 5:
        raise ValueError("Too few cells for the requested number of state prototypes")
    n_components = min(config.n_components, n_features - 1, n_cells - 1)
    if n_components < 2:
        raise ValueError("At least two state components are required")
    svd = TruncatedSVD(n_components=n_components, random_state=config.random_state)
    coordinates = svd.fit_transform(hvg_matrix).astype(np.float32)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(coordinates).astype(np.float32)
    kmeans = MiniBatchKMeans(
        n_clusters=config.n_states,
        random_state=config.random_state,
        batch_size=config.batch_size,
        n_init=10,
        reassignment_ratio=0.01,
    )
    raw_labels = kmeans.fit_predict(standardized)
    initialization_stability, initialization_ami = _initialization_stability(
        standardized, raw_labels, config
    )
    distances = kmeans.transform(standardized)
    order = np.argsort(distances, axis=1)
    nearest = distances[np.arange(n_cells), order[:, 0]]
    second = distances[np.arange(n_cells), order[:, 1]]
    confidence = np.clip(1.0 - nearest / np.maximum(second, 1e-8), 0.0, 1.0)

    cluster_sizes = np.bincount(raw_labels, minlength=config.n_states)
    stable_order = np.argsort(-cluster_sizes)
    stable_label_mapping = {
        int(old): f"VS{new:02d}" for new, old in enumerate(stable_order, start=1)
    }
    state_ids = np.asarray([stable_label_mapping[int(label)] for label in raw_labels])
    return StateAtlasResult(
        coordinates=coordinates,
        standardized_coordinates=standardized,
        state_ids=state_ids,
        assignment_confidence=confidence.astype(np.float32),
        nearest_distance=nearest.astype(np.float32),
        raw_state_labels=raw_labels.astype(np.int32),
        stable_label_mapping=stable_label_mapping,
        initialization_stability=initialization_stability,
        initialization_ami=initialization_ami,
        svd=svd,
        scaler=scaler,
        kmeans=kmeans,
    )


def transform_state_atlas(
    hvg_matrix: sparse.spmatrix | np.ndarray,
    svd: TruncatedSVD,
    scaler: StandardScaler,
    kmeans: MiniBatchKMeans,
    stable_label_mapping: Mapping[int, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign new cells to frozen state prototypes on the reference scale."""

    coordinates = svd.transform(hvg_matrix).astype(np.float32)
    standardized = scaler.transform(coordinates).astype(np.float32)
    raw_labels = kmeans.predict(standardized)
    missing = sorted(set(int(label) for label in raw_labels) - set(stable_label_mapping))
    if missing:
        raise ValueError(f"State label mapping is incomplete: {missing}")
    state_ids = np.asarray([stable_label_mapping[int(label)] for label in raw_labels])
    distances = kmeans.transform(standardized)
    order = np.argsort(distances, axis=1)
    nearest = distances[np.arange(len(raw_labels)), order[:, 0]]
    second = distances[np.arange(len(raw_labels)), order[:, 1]]
    confidence = np.clip(1.0 - nearest / np.maximum(second, 1e-8), 0.0, 1.0)
    return coordinates, state_ids, confidence.astype(np.float32), nearest.astype(np.float32)


def normalized_entropy(values: Sequence[object]) -> float:
    series = pd.Series(values, dtype="object").dropna()
    if series.empty:
        return float("nan")
    frequencies = series.value_counts(normalize=True).to_numpy(dtype=float)
    if len(frequencies) <= 1:
        return 0.0
    entropy = -float(np.sum(frequencies * np.log(frequencies)))
    return entropy / math.log(len(frequencies))


def _top_fraction_string(series: pd.Series, limit: int = 3) -> str:
    frequency = series.value_counts(normalize=True).head(limit)
    return ";".join(f"{name}:{fraction:.3f}" for name, fraction in frequency.items())


def summarize_patient_states(
    annotations: pd.DataFrame,
    program_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create patient summaries, long program statistics, and numeric vectors."""

    required = {
        "sample_id",
        "consensus_lineage",
        "lineage_confidence",
        "virtual_state_id",
        "state_assignment_confidence",
        "transcriptomic_suspicion_index",
        "confident_normal_lineage_anchor",
        "qc_pass",
    }
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"Annotation table is missing columns: {sorted(missing)}")
    program_thresholds = {
        column: float(annotations[column].quantile(0.75)) for column in program_columns
    }
    patient_rows: list[dict[str, object]] = []
    program_rows: list[dict[str, object]] = []
    vector_rows: list[dict[str, object]] = []
    lineage_levels = sorted(annotations["consensus_lineage"].dropna().unique())
    state_levels = sorted(annotations["virtual_state_id"].dropna().unique())
    for sample_id, group in annotations.groupby("sample_id", sort=True):
        lineage_fraction = group["consensus_lineage"].value_counts(normalize=True)
        state_fraction = group["virtual_state_id"].value_counts(normalize=True)
        row: dict[str, object] = {
            "patient_id": sample_id,
            "n_cells": int(len(group)),
            "qc_pass_fraction": float(group["qc_pass"].mean()),
            "mean_lineage_confidence": float(group["lineage_confidence"].mean()),
            "mean_state_assignment_confidence": float(
                group["state_assignment_confidence"].mean()
            ),
            "lineage_entropy_normalized": normalized_entropy(group["consensus_lineage"]),
            "state_entropy_normalized": normalized_entropy(group["virtual_state_id"]),
            "effective_state_count": float(
                np.exp(
                    -np.sum(
                        state_fraction.to_numpy(dtype=float)
                        * np.log(np.maximum(state_fraction.to_numpy(dtype=float), 1e-12))
                    )
                )
            ),
            "confident_normal_anchor_fraction": float(
                group["confident_normal_lineage_anchor"].mean()
            ),
            "high_transcriptomic_suspicion_fraction": float(
                (group["transcriptomic_suspicion_index"] >= 0.70).mean()
            ),
            "mean_transcriptomic_suspicion_index": float(
                group["transcriptomic_suspicion_index"].mean()
            ),
            "top_lineages": _top_fraction_string(group["consensus_lineage"]),
            "top_virtual_states": _top_fraction_string(group["virtual_state_id"]),
        }
        if "state_initialization_stability" in group:
            row["mean_state_initialization_stability"] = float(
                group["state_initialization_stability"].mean()
            )
        if "blast_prior_probability" in group:
            finite = pd.to_numeric(group["blast_prior_probability"], errors="coerce")
            row["mean_blast_prior_probability"] = (
                float(finite.mean()) if finite.notna().any() else float("nan")
            )
        patient_rows.append(row)

        vector: dict[str, object] = {"patient_id": sample_id}
        for lineage in lineage_levels:
            vector[f"lineage_fraction::{lineage}"] = float(lineage_fraction.get(lineage, 0.0))
        for state in state_levels:
            vector[f"state_fraction::{state}"] = float(state_fraction.get(state, 0.0))
        vector["heterogeneity::lineage_entropy"] = row["lineage_entropy_normalized"]
        vector["heterogeneity::state_entropy"] = row["state_entropy_normalized"]
        vector["heterogeneity::effective_states"] = row["effective_state_count"]
        if "mean_state_initialization_stability" in row:
            vector["heterogeneity::state_initialization_stability"] = row[
                "mean_state_initialization_stability"
            ]
        vector["suspicion::mean"] = row["mean_transcriptomic_suspicion_index"]

        for program in program_columns:
            values = pd.to_numeric(group[program], errors="coerce").dropna()
            threshold = program_thresholds[program]
            if values.empty:
                stats = {
                    "mean": float("nan"),
                    "median": float("nan"),
                    "q10": float("nan"),
                    "q90": float("nan"),
                    "fraction_above_reference_q75": float("nan"),
                }
            else:
                stats = {
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "q10": float(values.quantile(0.10)),
                    "q90": float(values.quantile(0.90)),
                    "fraction_above_reference_q75": float((values >= threshold).mean()),
                }
            program_rows.append(
                {
                    "patient_id": sample_id,
                    "program": program,
                    "reference_q75": threshold,
                    **stats,
                }
            )
            vector[f"program_mean::{program}"] = stats["mean"]
            vector[f"program_q90::{program}"] = stats["q90"]
        vector_rows.append(vector)
    summaries = pd.DataFrame(patient_rows).sort_values("patient_id").reset_index(drop=True)
    programs = pd.DataFrame(program_rows).sort_values(["patient_id", "program"])
    vectors = pd.DataFrame(vector_rows).sort_values("patient_id").reset_index(drop=True)
    return summaries, programs, vectors


def fit_reference_shift_model(
    vectors: pd.DataFrame,
) -> tuple[pd.DataFrame, ReferenceShiftModel]:
    """Fit a frozen descriptive shift reference from patient vectors."""

    if "patient_id" not in vectors:
        raise ValueError("Patient vectors require patient_id")
    numeric_columns = [column for column in vectors.columns if column != "patient_id"]
    numeric = vectors[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median()).fillna(0.0)
    values = numeric.to_numpy(dtype=float)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0) * 1.4826
    standard = np.std(values, axis=0)
    scale = np.where(mad >= 1e-5, mad, standard)
    usable = scale >= 1e-5
    if not np.any(usable):
        distance = np.zeros(len(values), dtype=float)
    else:
        zscore = (values[:, usable] - median[usable]) / scale[usable]
        distance = np.sqrt(np.mean(np.square(np.clip(zscore, -10.0, 10.0)), axis=1))
    rank = pd.Series(distance).rank(method="average", pct=True).to_numpy(dtype=float)
    output = vectors.copy()
    output["reference_shift_distance"] = distance
    output["reference_shift_percentile"] = rank
    output["reference_shift_flag"] = np.where(rank >= 0.90, "reference_edge", "in_reference_range")
    model = ReferenceShiftModel(
        feature_columns=tuple(numeric_columns),
        feature_median=median,
        feature_scale=scale,
        usable_features=usable,
        reference_distances=np.sort(distance),
    )
    return output, model


def transform_reference_shift(
    vectors: pd.DataFrame,
    model: ReferenceShiftModel,
) -> pd.DataFrame:
    """Evaluate new patient vectors against a frozen small reference cohort."""

    if "patient_id" not in vectors:
        raise ValueError("Patient vectors require patient_id")
    numeric = vectors.reindex(columns=model.feature_columns).apply(
        pd.to_numeric, errors="coerce"
    )
    for index, column in enumerate(model.feature_columns):
        numeric[column] = numeric[column].fillna(float(model.feature_median[index]))
    values = numeric.to_numpy(dtype=float)
    if not np.any(model.usable_features):
        distance = np.zeros(len(values), dtype=float)
    else:
        zscore = (
            values[:, model.usable_features]
            - model.feature_median[model.usable_features]
        ) / model.feature_scale[model.usable_features]
        distance = np.sqrt(np.mean(np.square(np.clip(zscore, -10.0, 10.0)), axis=1))
    reference = np.asarray(model.reference_distances, dtype=float)
    percentile = np.asarray(
        [
            (np.searchsorted(reference, value, side="right") + 1) / (len(reference) + 1)
            for value in distance
        ],
        dtype=float,
    )
    output = vectors.copy()
    output["reference_shift_distance"] = distance
    output["reference_shift_percentile"] = percentile
    output["reference_shift_flag"] = np.where(
        percentile >= 0.90, "reference_edge", "in_reference_range"
    )
    return output


def reference_shift_contributions(
    vectors: pd.DataFrame,
    model: ReferenceShiftModel,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Identify the patient-vector features driving descriptive cohort shift."""

    if "patient_id" not in vectors:
        raise ValueError("Patient vectors require patient_id")
    numeric = vectors.reindex(columns=model.feature_columns).apply(
        pd.to_numeric, errors="coerce"
    )
    for index, column in enumerate(model.feature_columns):
        numeric[column] = numeric[column].fillna(float(model.feature_median[index]))
    values = numeric.to_numpy(dtype=float)
    zscore = np.zeros_like(values, dtype=float)
    zscore[:, model.usable_features] = (
        values[:, model.usable_features] - model.feature_median[model.usable_features]
    ) / model.feature_scale[model.usable_features]
    rows: list[dict[str, object]] = []
    for patient_index, patient_id in enumerate(vectors["patient_id"].astype(str)):
        order = np.argsort(np.abs(zscore[patient_index]))[::-1][:top_n]
        for rank, feature_index in enumerate(order, start=1):
            rows.append(
                {
                    "patient_id": patient_id,
                    "rank": rank,
                    "feature": model.feature_columns[feature_index],
                    "patient_value": float(values[patient_index, feature_index]),
                    "reference_median": float(model.feature_median[feature_index]),
                    "reference_scale": float(model.feature_scale[feature_index]),
                    "robust_z": float(zscore[patient_index, feature_index]),
                    "absolute_robust_z": float(abs(zscore[patient_index, feature_index])),
                }
            )
    return pd.DataFrame(rows)


def add_reference_shift_metrics(vectors: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible helper returning within-reference shift metrics."""

    output, _ = fit_reference_shift_model(vectors)
    return output


def summarize_state_prototypes(
    annotations: pd.DataFrame,
    program_columns: Sequence[str],
) -> pd.DataFrame:
    """Summarize cross-patient expression-state prototypes."""

    rows: list[dict[str, object]] = []
    total_patients = max(annotations["sample_id"].nunique(), 1)
    for state_id, group in annotations.groupby("virtual_state_id", sort=True):
        lineage_frequency = group["consensus_lineage"].value_counts(normalize=True)
        program_means = {
            program: float(pd.to_numeric(group[program], errors="coerce").mean())
            for program in program_columns
        }
        top_programs = sorted(program_means, key=program_means.get, reverse=True)[:3]
        rows.append(
            {
                "virtual_state_id": state_id,
                "n_cells": int(len(group)),
                "n_patients_represented": int(group["sample_id"].nunique()),
                "patient_coverage_fraction": float(
                    group["sample_id"].nunique() / total_patients
                ),
                "dominant_lineage": str(lineage_frequency.index[0]),
                "dominant_lineage_fraction": float(lineage_frequency.iloc[0]),
                "mean_assignment_confidence": float(
                    group["state_assignment_confidence"].mean()
                ),
                "mean_initialization_stability": (
                    float(group["state_initialization_stability"].mean())
                    if "state_initialization_stability" in group
                    else float("nan")
                ),
                "mean_transcriptomic_suspicion_index": float(
                    group["transcriptomic_suspicion_index"].mean()
                ),
                "top_programs": ";".join(
                    f"{program}:{program_means[program]:.3f}" for program in top_programs
                ),
                **{f"program_mean::{key}": value for key, value in program_means.items()},
            }
        )
    return pd.DataFrame(rows).sort_values("virtual_state_id").reset_index(drop=True)
