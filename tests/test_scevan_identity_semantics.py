from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _load_statuses(path: Path) -> dict[str, dict[str, object]]:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_pyscevan_identity.py"
    spec = importlib.util.spec_from_file_location("run_pyscevan_identity", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_statuses(path)


def _identity_inputs(patient: str) -> pd.DataFrame:
    barcodes = [f"b{index}" for index in range(10)]
    healthy = [False] * 8 + [True] * 2
    return pd.DataFrame(
        {
            "sample_id": patient,
            "cell_barcode": barcodes,
            "seurat_cluster": ["0"] * 10,
            "sctype_classification": ["AML"] * 8 + ["T cells"] * 2,
            "known_normal_reference": healthy,
            "sctype_malignant_healthy": ["malignant"] * 8 + ["healthy"] * 2,
        }
    )


def test_scevan_semantics_and_python_r_parity_audit(tmp_path: Path) -> None:
    identity_dir = tmp_path / "identity"
    legacy_dir = tmp_path / "legacy"
    r_dir = tmp_path / "r"
    py_dir = tmp_path / "py"
    out_dir = tmp_path / "audit"
    for directory in (identity_dir, legacy_dir, r_dir, py_dir):
        directory.mkdir()
    for patient in ("p1", "p2"):
        identity = _identity_inputs(patient)
        identity.to_csv(identity_dir / f"{patient}_identity_inputs.csv", index=False)
        expected = identity["known_normal_reference"].map(
            {True: "healthy", False: "malignant"}
        )
        pd.DataFrame(
            {
                "cell_barcode": identity["cell_barcode"],
                "SCEVAN_output": expected,
            }
        ).to_csv(legacy_dir / f"{patient}_identity_calls.csv", index=False)
        pd.DataFrame(
            {
                "cell_barcode": identity["cell_barcode"],
                "SCEVAN_output": expected,
            }
        ).to_csv(r_dir / f"{patient}_scevan_r_classification.csv", index=False)
        pd.DataFrame(
            {
                "cell_barcode": identity["cell_barcode"],
                "SCEVAN_output": expected,
            }
        ).to_csv(py_dir / f"{patient}_scevan_python_classification.csv", index=False)

    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "audit_scevan_identity_semantics.py"),
            "--legacy-call-dir",
            str(legacy_dir),
            "--identity-input-dir",
            str(identity_dir),
            "--r-reference-dir",
            str(r_dir),
            "--python-call-dir",
            str(py_dir),
            "--reference-patients",
            "p1,p2",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )
    summary = json.loads(
        (out_dir / "scevan_semantics_and_parity_summary.json").read_text()
    )
    assert summary["released_code_semantics_confirmed"]
    assert summary["python_reimplementation_parity_gate_pass"]


def test_identity_modes_remain_explicitly_separate(tmp_path: Path) -> None:
    identity_dir = tmp_path / "identity"
    legacy_dir = tmp_path / "legacy"
    copykat_dir = tmp_path / "copykat"
    py_dir = tmp_path / "py"
    out_dir = tmp_path / "modes"
    for directory in (identity_dir, legacy_dir, copykat_dir, py_dir):
        directory.mkdir()
    for patient in ("p1", "p2"):
        identity = _identity_inputs(patient)
        identity.to_csv(identity_dir / f"{patient}_identity_inputs.csv", index=False)
        copykat = ["malignant"] * 7 + ["healthy"] * 3
        if patient == "p1":
            pd.DataFrame(
                {
                    "cell_barcode": identity["cell_barcode"],
                    "copyKat_output": copykat,
                }
            ).to_csv(legacy_dir / f"{patient}_identity_calls.csv", index=False)
        else:
            pd.DataFrame(
                {
                    "cell_barcode": identity["cell_barcode"],
                    "copyKat_output": copykat,
                }
            ).to_csv(copykat_dir / f"{patient}_copykat_calls.csv", index=False)
        true_scevan = ["healthy"] + ["malignant"] * 7 + ["healthy"] * 2
        pd.DataFrame(
            {
                "cell_barcode": identity["cell_barcode"],
                "SCEVAN_output": true_scevan,
            }
        ).to_csv(py_dir / f"{patient}_scevan_python_classification.csv", index=False)

    parity_path = tmp_path / "parity.json"
    parity_path.write_text(
        json.dumps({"python_reimplementation_parity_gate_pass": True}),
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "assemble_sctherapy_identity_modes.py"),
            "--identity-input-dir",
            str(identity_dir),
            "--legacy-call-dir",
            str(legacy_dir),
            "--copykat-dir",
            str(copykat_dir),
            "--python-scevan-dir",
            str(py_dir),
            "--parity-summary",
            str(parity_path),
            "--patients",
            "p1,p2",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )
    released = pd.read_csv(
        out_dir / "released_code_effective" / "p1_identity_calls.csv"
    )
    corrected = pd.read_csv(
        out_dir / "true_scevan_corrected" / "p1_identity_calls.csv"
    )
    assert released.loc[0, "SCEVAN_output"] == "malignant"
    assert corrected.loc[0, "SCEVAN_output"] == "healthy"
    assert released.loc[0, "scevan_call_source"] != corrected.loc[
        0, "scevan_call_source"
    ]


def test_pyscevan_status_file_is_incremental(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    status_path = out_dir / "run_status.json"
    status_path.write_text(
        json.dumps({"p1": {"patient_id": "p1", "status": "complete"}}),
        encoding="utf-8",
    )
    statuses = _load_statuses(status_path)
    statuses["p2"] = {"patient_id": "p2", "status": "complete"}
    status_path.write_text(json.dumps(statuses), encoding="utf-8")
    assert set(_load_statuses(status_path)) == {"p1", "p2"}
