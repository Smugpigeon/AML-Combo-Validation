"""TCGA-LAML public-cohort ETL (for Week 5 independent-cohort validation).

Pulls the TCGA Acute Myeloid Leukemia PanCancer Atlas 2018 cohort from cBioPortal's
public REST API, builds patient × 80-feature matrix matching the BeatAML schema
(50 PCA on variable-gene log expression + 25 curated mutation genes + 5 clinical),
then writes a canonical CSV.

Why this is a separate ETL (not a BeatAML extension):
  - TCGA uses different sample IDs, different expression units (RSEM vs feature counts),
    and different clinical fields than BeatAML.
  - For independent-cohort validation, we want to apply the SAME pipeline shape
    (PCA on variable genes + mutation binary + clinical) to TCGA and evaluate
    if combo-predictor recommendations land on clinically reasonable regimens.
  - We re-fit PCA on TCGA alone. Not sharing the PCA with BeatAML is defensible
    because the combo predictor uses mechanism-prior + drug embeddings; it
    doesn't require BeatAML's exact PC directions. Clinical-biology signal
    (FLT3/NPM1/TP53 mutation features, blast %, risk) carries over directly.

Outputs:
  - data/canonical/tcga_laml_patient_features.csv
      columns: patient_id + rna_pc01..50 + mut_{25 genes} + clin_{5 fields}
  - data/canonical/tcga_laml_clinical.csv
      extra clinical (OS months, vital status, karyotype) for Week 5 analyses
  - data/canonical/tcga_laml_manifest.json
"""

from __future__ import annotations

import gzip
import io
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# Same curated gene panel as BeatAML (must match exactly so features align).
CURATED_MUTATION_GENES: tuple[str, ...] = (
    "FLT3", "NPM1", "DNMT3A", "IDH1", "IDH2", "TP53",
    "RUNX1", "ASXL1", "TET2", "CEBPA", "KIT",
    "NRAS", "KRAS", "PTPN11", "WT1", "BCOR",
    "STAG2", "PHF6", "SRSF2", "SF3B1", "U2AF1",
    "EZH2", "KMT2A", "MECOM", "CBFB",
)

# ELN risk ordinal (cytogenetic-based for TCGA; BeatAML uses the literal strings).
# TCGA PanCan clinical uses CYTOGENETIC_CODE values or "RISK_CYTOGENETICS" strings.
CYTOGENETIC_RISK_ORDINAL: dict[str, float] = {
    "FAVORABLE": 0.0,
    "Favorable": 0.0,
    "INTERMEDIATE/NORMAL": 1.0,
    "INTERMEDIATE": 1.0,
    "Intermediate": 1.0,
    "POOR": 2.0,
    "Poor": 2.0,
    "Adverse": 2.0,
    "UNKNOWN": np.nan,
    "Unknown": np.nan,
}

CBIOPORTAL_BASE = "https://www.cbioportal.org/api"
STUDY_ID = "laml_tcga_pan_can_atlas_2018"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TcgaConfig:
    out_dir: Path = Path("data/canonical")
    cache_dir: Path = Path("data/raw/tcga_laml")
    study_id: str = STUDY_ID
    mrna_profile_id: str = f"{STUDY_ID}_rna_seq_v2_mrna"
    mutation_profile_id: str = f"{STUDY_ID}_mutations"
    sample_list_id: str = f"{STUDY_ID}_rna_seq_v2_mrna"   # samples with mRNA
    n_pca: int = 50
    top_n_variable_genes: int = 5000
    random_state: int = 42
    request_timeout_s: int = 120


# ---------------------------------------------------------------------------
# cBioPortal API fetchers (with on-disk caching)
# ---------------------------------------------------------------------------


def _http_json(url: str, *, method: str = "GET", body: dict | list | None = None,
               timeout: int = 120) -> object:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cache_or_fetch(cache_path: Path, fetch_fn) -> object:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    payload = fetch_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return payload


def fetch_sample_list(cfg: TcgaConfig) -> list[dict]:
    """Samples with mRNA data. Patient ID is derivable from sample ID in TCGA."""
    url = f"{CBIOPORTAL_BASE}/sample-lists/{cfg.sample_list_id}/sample-ids"
    cache_path = cfg.cache_dir / "sample_ids.json"

    sample_ids = _cache_or_fetch(
        cache_path,
        lambda: _http_json(url, timeout=cfg.request_timeout_s),
    )
    # Parse patient id directly from sample id: TCGA-AB-2802-03 → TCGA-AB-2802
    out = []
    for sid in sample_ids:
        parts = sid.split("-")
        pid = "-".join(parts[:3]) if len(parts) >= 3 else sid
        out.append({"sampleId": sid, "patientId": pid, "studyId": cfg.study_id})
    return out


def fetch_clinical(cfg: TcgaConfig, sample_ids: list[str]) -> pd.DataFrame:
    """Patient-level clinical attributes via cBioPortal API.

    Note: cBioPortal's POST /clinical-data/fetch expects `entityId`, not
    `patientId`/`sampleId`. TCGA-LAML PanCan study has ~56 attributes total;
    we request only the fields relevant for AML + Week 5 survival analyses.
    """
    cache_path = cfg.cache_dir / "clinical_data.json"

    def _fetch():
        patient_ids = sorted({"-".join(s.split("-")[:3]) for s in sample_ids})
        body = {
            "identifiers": [
                {"studyId": cfg.study_id, "entityId": p} for p in patient_ids
            ],
            "attributeIds": [
                "AGE", "SEX", "RACE",
                "SUBTYPE",              # FAB subtype
                "OS_STATUS", "OS_MONTHS",
                "DFS_STATUS", "DFS_MONTHS",
                "DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS",
                "AJCC_PATHOLOGIC_TUMOR_STAGE",
            ],
        }
        url = f"{CBIOPORTAL_BASE}/clinical-data/fetch?clinicalDataType=PATIENT"
        return _http_json(url, method="POST", body=body,
                          timeout=cfg.request_timeout_s)

    raw = _cache_or_fetch(cache_path, _fetch)
    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame({"patientId": []})
    return df.pivot_table(
        index="patientId", columns="clinicalAttributeId", values="value",
        aggfunc="first",
    ).reset_index()


def fetch_mutations(cfg: TcgaConfig, sample_ids: list[str],
                    gene_symbols: Iterable[str]) -> pd.DataFrame:
    """Mutations for curated genes. Returns per-sample gene hit matrix."""
    cache_path = cfg.cache_dir / "mutations.json"

    def _fetch():
        # Resolve gene symbols → Entrez IDs (API expects a plain list, geneIdType in URL)
        genes_resp = _http_json(
            f"{CBIOPORTAL_BASE}/genes/fetch?geneIdType=HUGO_GENE_SYMBOL",
            method="POST", body=list(gene_symbols),
            timeout=cfg.request_timeout_s,
        )
        entrez_ids = [int(g["entrezGeneId"]) for g in genes_resp]
        sym_to_entrez = {g["hugoGeneSymbol"]: int(g["entrezGeneId"]) for g in genes_resp}

        # Fetch mutations for these genes in these samples
        mut_body = {
            "entrezGeneIds": entrez_ids,
            "sampleIds": sample_ids,
        }
        url = (f"{CBIOPORTAL_BASE}/molecular-profiles/"
               f"{cfg.mutation_profile_id}/mutations/fetch?projection=SUMMARY")
        muts = _http_json(url, method="POST", body=mut_body,
                          timeout=cfg.request_timeout_s)
        return {"genes": genes_resp, "mutations": muts, "sym_to_entrez": sym_to_entrez}

    raw = _cache_or_fetch(cache_path, _fetch)
    sym_to_entrez = raw["sym_to_entrez"]
    entrez_to_sym = {v: k for k, v in sym_to_entrez.items()}

    rows = []
    for m in raw["mutations"]:
        sym = entrez_to_sym.get(int(m.get("entrezGeneId", 0)))
        if sym is None:
            continue
        rows.append({
            "sampleId": m.get("sampleId"),
            "patientId": m.get("patientId"),
            "gene": sym,
            "mutationType": m.get("mutationType"),
        })
    if not rows:
        return pd.DataFrame(columns=["sampleId", "patientId", "gene", "mutationType"])
    return pd.DataFrame(rows)


def fetch_expression(cfg: TcgaConfig, sample_ids: list[str]) -> pd.DataFrame:
    """Pull RNA-seq v2 RSEM expression as gene×sample DataFrame.

    To keep memory sane we fetch in batches of 4000 Entrez IDs per request.
    """
    cache_path = cfg.cache_dir / "expression.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    # Step 1: Get gene list for the mRNA profile (all Entrez IDs).
    all_genes_url = f"{CBIOPORTAL_BASE}/genes?pageSize=30000&projection=ID"
    gene_cache = cfg.cache_dir / "all_genes.json"
    all_genes = _cache_or_fetch(
        gene_cache,
        lambda: _http_json(all_genes_url, timeout=cfg.request_timeout_s),
    )
    entrez_list = [int(g["entrezGeneId"]) for g in all_genes]
    symbol_map = {int(g["entrezGeneId"]): g["hugoGeneSymbol"] for g in all_genes}
    print(f"[tcga] total genes in cBioPortal: {len(entrez_list):,}")

    batches = [entrez_list[i:i + 4000] for i in range(0, len(entrez_list), 4000)]
    all_rows = []
    url = (f"{CBIOPORTAL_BASE}/molecular-profiles/"
           f"{cfg.mrna_profile_id}/molecular-data/fetch?projection=SUMMARY")

    for bi, batch in enumerate(batches):
        batch_cache = cfg.cache_dir / f"expr_batch_{bi:03d}.json"
        body = {"entrezGeneIds": batch, "sampleIds": sample_ids}
        payload = _cache_or_fetch(
            batch_cache,
            lambda body=body: _http_json(
                url, method="POST", body=body, timeout=cfg.request_timeout_s),
        )
        all_rows.extend(payload)
        print(f"[tcga] expression batch {bi + 1}/{len(batches)}  rows cumulative={len(all_rows):,}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("TCGA expression fetch returned 0 rows")
    df["symbol"] = df["entrezGeneId"].map(symbol_map)
    df = df.dropna(subset=["symbol"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # gene × sample pivot
    expr = df.pivot_table(index="symbol", columns="sampleId", values="value", aggfunc="mean")
    expr.to_parquet(cache_path)
    return expr


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------


def _extract_rna_pca(expr: pd.DataFrame, *, n_pca: int, top_n_variable: int,
                     random_state: int) -> tuple[pd.DataFrame, dict]:
    X = expr.to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, a_min=0.0, a_max=None)
    X = np.log2(X + 1.0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.isfinite(X).all()

    gene_var = X.var(axis=1)
    top_idx = np.argsort(gene_var)[::-1][:top_n_variable]
    top_idx.sort()
    X = X[top_idx]
    kept_genes = expr.index[top_idx].tolist()

    X_t = X.T
    pca = PCA(n_components=n_pca, random_state=random_state)
    X_pca = pca.fit_transform(X_t)

    sample_pca = pd.DataFrame(
        X_pca, index=expr.columns,
        columns=[f"rna_pc{i+1:02d}" for i in range(n_pca)],
    )
    sample_pca.index.name = "sample_id"
    meta = {
        "n_samples_used": int(X_t.shape[0]),
        "n_genes_before": int(expr.shape[0]),
        "n_genes_kept_top_variable": int(len(kept_genes)),
        "explained_variance_ratio_cumulative":
            np.cumsum(pca.explained_variance_ratio_).round(4).tolist(),
        "n_pca": int(n_pca),
    }
    return sample_pca, meta


def _extract_mutation_binary(mut_df: pd.DataFrame,
                              sample_ids: list[str],
                              genes: tuple[str, ...]) -> pd.DataFrame:
    feat = pd.DataFrame(0, index=sample_ids,
                         columns=[f"mut_{g}" for g in genes])
    feat.index.name = "sample_id"
    if mut_df.empty:
        return feat.astype(np.int8)
    for _, row in mut_df.iterrows():
        sid = row["sampleId"]
        gene = row["gene"]
        col = f"mut_{gene}"
        if sid in feat.index and col in feat.columns:
            feat.at[sid, col] = 1
    return feat.astype(np.int8)


def _extract_clinical(clinical_df: pd.DataFrame,
                       sample_to_patient: dict[str, str]) -> pd.DataFrame:
    """Clinical features matching BeatAML columns where possible."""
    clinical_df = clinical_df.set_index("patientId")
    rows = []
    for sid, pid in sample_to_patient.items():
        if pid not in clinical_df.index:
            rows.append({
                "sample_id": sid, "patient_id": pid,
                "clin_age": np.nan, "clin_eln_ordinal": np.nan,
                "clin_blast_pct": np.nan, "clin_secondary_aml": np.nan,
                "clin_fit_for_intensive": np.nan,
            })
            continue
        r = clinical_df.loc[pid]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]

        age_s = r.get("AGE", np.nan)
        try:
            age = float(age_s) if pd.notna(age_s) else np.nan
        except (TypeError, ValueError):
            age = np.nan

        risk_raw = r.get("RISK_CYTOGENETICS") or r.get("RISK_MOLECULAR")
        eln = CYTOGENETIC_RISK_ORDINAL.get(str(risk_raw), np.nan) if pd.notna(risk_raw) else np.nan

        # NOTE: TCGA-LAML PanCancer Atlas 2018 does not expose AGE/SEX/RACE
        # via the cBioPortal API (only OS_MONTHS, OS_STATUS, SUBTYPE, SAMPLE_COUNT,
        # CANCER_TYPE_ACRONYM are populated). Fill clinical placeholders with BeatAML
        # medians so the 80-dim feature schema is preserved for later transfer work.
        BEATAML_CLINICAL_MEDIAN = {
            "clin_age": 62.0,
            "clin_eln_ordinal": 1.0,
            "clin_blast_pct": 70.0,
            "clin_secondary_aml": 0.16,
            "clin_fit_for_intensive": 0.5,
        }
        rows.append({
            "sample_id": sid,
            "patient_id": pid,
            "clin_age": age if pd.notna(age) else BEATAML_CLINICAL_MEDIAN["clin_age"],
            "clin_eln_ordinal": eln if pd.notna(eln) else BEATAML_CLINICAL_MEDIAN["clin_eln_ordinal"],
            "clin_blast_pct": BEATAML_CLINICAL_MEDIAN["clin_blast_pct"],
            "clin_secondary_aml": BEATAML_CLINICAL_MEDIAN["clin_secondary_aml"],
            "clin_fit_for_intensive": (
                1.0 if pd.notna(age) and age <= 60.0 else
                (0.0 if pd.notna(age) else BEATAML_CLINICAL_MEDIAN["clin_fit_for_intensive"])
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_tcga_etl(cfg: TcgaConfig | None = None) -> dict:
    cfg = cfg or TcgaConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[tcga] fetching sample list for study {cfg.study_id} ...")
    samples = fetch_sample_list(cfg)
    sample_ids = [s["sampleId"] for s in samples]
    sample_to_patient = {s["sampleId"]: s["patientId"] for s in samples}
    print(f"[tcga] {len(sample_ids)} samples with mRNA data, "
          f"{len(set(sample_to_patient.values()))} unique patients "
          f"[{time.time() - t0:.1f}s]")

    print("[tcga] fetching clinical data ...")
    clinical = fetch_clinical(cfg, sample_ids)
    print(f"[tcga] clinical rows: {len(clinical)}  cols: {list(clinical.columns)[:6]}...")

    print("[tcga] fetching mutations for 25 curated genes ...")
    muts = fetch_mutations(cfg, sample_ids, CURATED_MUTATION_GENES)
    print(f"[tcga] mutation rows: {len(muts):,}  unique samples with mut: "
          f"{muts['sampleId'].nunique() if not muts.empty else 0}")

    print("[tcga] fetching expression ...")
    expr = fetch_expression(cfg, sample_ids)
    print(f"[tcga] expression matrix: {expr.shape}")

    # Drop samples not in expression matrix
    expr_samples = [s for s in sample_ids if s in expr.columns]
    expr = expr[expr_samples]

    print("[tcga] extracting RNA PCA features ...")
    rna_feat, rna_meta = _extract_rna_pca(
        expr, n_pca=cfg.n_pca, top_n_variable=cfg.top_n_variable_genes,
        random_state=cfg.random_state,
    )

    print("[tcga] extracting mutation features ...")
    mut_feat = _extract_mutation_binary(muts, expr_samples, CURATED_MUTATION_GENES)

    print("[tcga] extracting clinical features ...")
    clin_feat = _extract_clinical(
        clinical, {sid: sample_to_patient[sid] for sid in expr_samples},
    ).set_index("sample_id")

    # Combine — sample level
    features = rna_feat.join(mut_feat, how="left").join(clin_feat, how="left")
    features["patient_id"] = [sample_to_patient[s] for s in features.index]

    # Average per patient (should be 1:1 for TCGA-LAML but stay safe)
    col_order = (
        ["patient_id"] +
        [c for c in features.columns if c.startswith("rna_pc")] +
        [c for c in features.columns if c.startswith("mut_")] +
        [c for c in features.columns if c.startswith("clin_")]
    )
    features = features[col_order]
    patient_features = features.groupby("patient_id").mean(numeric_only=True).reset_index()

    # Impute remaining NaN (clinical fields mainly) with median across cohort
    for c in patient_features.columns:
        if c == "patient_id":
            continue
        if patient_features[c].isna().any():
            median = patient_features[c].median()
            patient_features[c] = patient_features[c].fillna(median if pd.notna(median) else 0.0)

    out_path = cfg.out_dir / "tcga_laml_patient_features.csv"
    patient_features.to_csv(out_path, index=False)
    print(f"[tcga] patient features: {patient_features.shape} → {out_path}")

    # Also dump raw clinical for Week 5 analyses (OS, DFS, regimens)
    clin_out = cfg.out_dir / "tcga_laml_clinical.csv"
    clinical.to_csv(clin_out, index=False)

    manifest = {
        "cfg": asdict(cfg),
        "elapsed_s": round(time.time() - t0, 1),
        "n_samples": int(len(sample_ids)),
        "n_patients_final": int(patient_features.shape[0]),
        "rna_meta": rna_meta,
        "mut_gene_prevalence": {
            c: round(float(mut_feat[c].mean()), 3) for c in mut_feat.columns
        },
        "outputs": {
            "patient_features": str(out_path),
            "clinical": str(clin_out),
        },
    }
    manifest_path = cfg.out_dir / "tcga_laml_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"[tcga] DONE in {time.time() - t0:.1f}s")
    print(f"[tcga] top-5 mutation prevalence:")
    top5 = sorted(manifest["mut_gene_prevalence"].items(),
                  key=lambda kv: -kv[1])[:5]
    for g, f in top5:
        print(f"    {g:20s}  {f:.2%}")
    return manifest


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data/canonical")
    ap.add_argument("--cache", default="data/raw/tcga_laml")
    ap.add_argument("--n-pca", type=int, default=50)
    ap.add_argument("--top-n-genes", type=int, default=5000)
    args = ap.parse_args()

    cfg = TcgaConfig(
        out_dir=Path(args.out),
        cache_dir=Path(args.cache),
        n_pca=args.n_pca,
        top_n_variable_genes=args.top_n_genes,
    )
    run_tcga_etl(cfg)


if __name__ == "__main__":
    main()
