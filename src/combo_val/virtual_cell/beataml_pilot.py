"""BeatAML virtual patient-state pilot.

The model is deliberately narrower than a full single-cell digital twin. It
encodes one de-identified BeatAML patient into a shared latent state and learns
how a categorical drug perturbation changes the expected ex-vivo AUC. Patient
splits, preprocessing statistics, and drug target statistics are fitted from
the training partition only.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings persisted with every checkpoint."""

    n_features: int
    n_drugs: int
    hidden_dim: int = 128
    latent_dim: int = 32
    head_dim: int = 96
    dropout: float = 0.15


@dataclass(frozen=True)
class PatientSplit:
    """Disjoint patient identifiers used for leakage-safe evaluation."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self) -> None:
        train = set(self.train)
        validation = set(self.validation)
        test = set(self.test)
        if train & validation or train & test or validation & test:
            raise ValueError("Patient split leakage detected")
        if not train or not validation or not test:
            raise ValueError("Train, validation, and test patient sets must be non-empty")


@dataclass
class PreparedBeatAML:
    """Prepared matrices and train-only statistics."""

    patient_ids: list[str]
    feature_columns: list[str]
    drug_ids: list[str]
    features_scaled: np.ndarray
    responses: pd.DataFrame
    split: PatientSplit
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    drug_mean: np.ndarray
    drug_scale: np.ndarray
    drug_train_count: np.ndarray


class BeatAMLVirtualCellModel(nn.Module):
    """Composable patient-state and drug-perturbation response model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.patient_encoder = nn.Sequential(
            nn.Linear(config.n_features, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.Tanh(),
        )
        self.drug_embedding = nn.Embedding(config.n_drugs, config.latent_dim)
        self.composition_norm = nn.LayerNorm(config.latent_dim)
        self.response_head = nn.Sequential(
            nn.Linear(config.latent_dim * 4, config.head_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_dim, config.head_dim // 2),
            nn.GELU(),
            nn.Linear(config.head_dim // 2, 1),
        )

    def encode_patient(self, patient_features: torch.Tensor) -> torch.Tensor:
        """Return a drug-independent patient baseline state."""

        return self.patient_encoder(patient_features)

    def forward(
        self, patient_features: torch.Tensor, drug_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patient_state = self.encode_patient(patient_features)
        drug_state = self.drug_embedding(drug_index)
        composed = self.composition_norm(patient_state + drug_state)
        interaction = torch.cat(
            [patient_state, drug_state, composed, patient_state * drug_state], dim=1
        )
        response_z = self.response_head(interaction).squeeze(1)
        return response_z, patient_state


def normalize_identifier(value: object) -> str:
    """Normalize numeric dbGaP subject IDs without changing alphanumeric IDs."""

    if pd.isna(value):
        raise ValueError("Patient identifier cannot be missing")
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_patient_split(
    patient_ids: Sequence[str], seed: int = 42, train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> PatientSplit:
    """Split unique patient IDs, never individual patient-drug rows."""

    unique = np.asarray(sorted(set(patient_ids)), dtype=object)
    if len(unique) < 10:
        raise ValueError("At least 10 patients are required for a three-way split")
    rng = np.random.default_rng(seed)
    shuffled = unique[rng.permutation(len(unique))]
    n_train = max(1, int(round(len(unique) * train_fraction)))
    n_validation = max(1, int(round(len(unique) * validation_fraction)))
    if n_train + n_validation >= len(unique):
        n_train = len(unique) - 2
        n_validation = 1
    split = PatientSplit(
        train=tuple(str(x) for x in shuffled[:n_train]),
        validation=tuple(str(x) for x in shuffled[n_train : n_train + n_validation]),
        test=tuple(str(x) for x in shuffled[n_train + n_validation :]),
    )
    split.validate()
    return split


def load_and_prepare_beataml(
    feature_csv: Path,
    response_csv: Path,
    split_seed: int = 42,
) -> PreparedBeatAML:
    """Load canonical BeatAML tables and fit all statistics on training patients."""

    features = pd.read_csv(feature_csv)
    responses = pd.read_csv(response_csv)
    required_features = {"patient_id"}
    required_responses = {"patient_id", "drug_id", "auc"}
    if not required_features.issubset(features.columns):
        raise ValueError(f"Feature table requires columns: {sorted(required_features)}")
    if not required_responses.issubset(responses.columns):
        raise ValueError(f"Response table requires columns: {sorted(required_responses)}")

    features = features.copy()
    responses = responses.copy()
    features["patient_id"] = features["patient_id"].map(normalize_identifier)
    responses["patient_id"] = responses["patient_id"].map(normalize_identifier)
    responses["drug_id"] = responses["drug_id"].astype(str).str.strip()
    responses["auc"] = pd.to_numeric(responses["auc"], errors="coerce")
    responses = responses.dropna(subset=["auc"])

    if features["patient_id"].duplicated().any():
        duplicates = features.loc[features["patient_id"].duplicated(), "patient_id"].tolist()
        raise ValueError(f"Feature table has duplicate patient IDs: {duplicates[:5]}")

    feature_columns = [c for c in features.columns if c != "patient_id"]
    numeric = features[feature_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Feature table contains missing/non-numeric values: {bad[:10]}")

    response_patients = set(responses["patient_id"])
    model_patients = [p for p in features["patient_id"] if p in response_patients]
    if not model_patients:
        raise ValueError("No patient IDs overlap between feature and response tables")
    split = make_patient_split(model_patients, seed=split_seed)

    patient_ids = features["patient_id"].tolist()
    patient_index = {patient_id: idx for idx, patient_id in enumerate(patient_ids)}
    responses = responses[responses["patient_id"].isin(patient_index)].copy()
    drug_ids = sorted(responses["drug_id"].unique().tolist())
    drug_index = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
    responses["patient_index"] = responses["patient_id"].map(patient_index).astype(int)
    responses["drug_index"] = responses["drug_id"].map(drug_index).astype(int)
    split_lookup = {
        **{p: "train" for p in split.train},
        **{p: "validation" for p in split.validation},
        **{p: "test" for p in split.test},
    }
    responses["split"] = responses["patient_id"].map(split_lookup)
    responses = responses.dropna(subset=["split"]).reset_index(drop=True)

    train_patient_rows = [patient_index[p] for p in split.train]
    raw_features = numeric.to_numpy(dtype=np.float32)
    feature_mean = raw_features[train_patient_rows].mean(axis=0)
    feature_scale = raw_features[train_patient_rows].std(axis=0)
    feature_scale = np.where(feature_scale < 1e-6, 1.0, feature_scale)
    features_scaled = ((raw_features - feature_mean) / feature_scale).astype(np.float32)

    train_responses = responses[responses["split"] == "train"]
    global_mean = float(train_responses["auc"].mean())
    global_scale = float(train_responses["auc"].std(ddof=0))
    if not math.isfinite(global_scale) or global_scale < 1e-6:
        global_scale = 1.0
    grouped = train_responses.groupby("drug_id")["auc"].agg(["mean", "std", "count"])
    drug_mean = np.full(len(drug_ids), global_mean, dtype=np.float32)
    drug_scale = np.full(len(drug_ids), global_scale, dtype=np.float32)
    drug_train_count = np.zeros(len(drug_ids), dtype=np.int64)
    for drug_id, idx in drug_index.items():
        if drug_id not in grouped.index:
            continue
        row = grouped.loc[drug_id]
        drug_mean[idx] = float(row["mean"])
        scale = float(row["std"])
        drug_scale[idx] = scale if math.isfinite(scale) and scale >= 1e-6 else global_scale
        drug_train_count[idx] = int(row["count"])

    response_drug_index = responses["drug_index"].to_numpy(dtype=int)
    responses["target_z"] = (
        responses["auc"].to_numpy(dtype=np.float32) - drug_mean[response_drug_index]
    ) / drug_scale[response_drug_index]
    responses["target_z"] = responses["target_z"].clip(-6.0, 6.0)

    return PreparedBeatAML(
        patient_ids=patient_ids,
        feature_columns=feature_columns,
        drug_ids=drug_ids,
        features_scaled=features_scaled,
        responses=responses,
        split=split,
        feature_mean=feature_mean.astype(np.float32),
        feature_scale=feature_scale.astype(np.float32),
        drug_mean=drug_mean,
        drug_scale=drug_scale,
        drug_train_count=drug_train_count,
    )


def _tensor_dataset(frame: pd.DataFrame) -> TensorDataset:
    return TensorDataset(
        torch.as_tensor(frame["patient_index"].to_numpy(copy=True), dtype=torch.long),
        torch.as_tensor(frame["drug_index"].to_numpy(copy=True), dtype=torch.long),
        torch.as_tensor(frame["target_z"].to_numpy(copy=True), dtype=torch.float32),
    )


def _mean_loss(
    model: BeatAMLVirtualCellModel,
    features: torch.Tensor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for patient_index, drug_index, target in loader:
            patient_index = patient_index.to(device)
            drug_index = drug_index.to(device)
            target = target.to(device)
            prediction, _ = model(features[patient_index], drug_index)
            loss = criterion(prediction, target)
            total += float(loss) * len(target)
            count += len(target)
    return total / max(count, 1)


def train_one_model(
    prepared: PreparedBeatAML,
    seed: int,
    device: torch.device,
    max_epochs: int = 100,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    patience: int = 12,
) -> tuple[BeatAMLVirtualCellModel, dict[str, object]]:
    """Train one ensemble member with early stopping on validation Huber loss."""

    seed_everything(seed)
    config = ModelConfig(
        n_features=len(prepared.feature_columns),
        n_drugs=len(prepared.drug_ids),
    )
    model = BeatAMLVirtualCellModel(config).to(device)
    features = torch.as_tensor(prepared.features_scaled, dtype=torch.float32, device=device)
    train_frame = prepared.responses[prepared.responses["split"] == "train"]
    validation_frame = prepared.responses[prepared.responses["split"] == "validation"]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        _tensor_dataset(train_frame),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        _tensor_dataset(validation_frame), batch_size=batch_size * 2, shuffle=False
    )
    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for patient_index, drug_index, target in train_loader:
            patient_index = patient_index.to(device)
            drug_index = drug_index.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, patient_state = model(features[patient_index], drug_index)
            loss = criterion(prediction, target) + 1e-4 * patient_state.square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_total += float(loss.detach()) * len(target)
            train_count += len(target)

        validation_loss = _mean_loss(
            model, features, validation_loader, criterion, device
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_total / max(train_count, 1),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    diagnostics: dict[str, object] = {
        "seed": seed,
        "best_validation_loss": best_loss,
        "epochs_trained": len(history),
        "best_epoch": int(min(history, key=lambda row: row["validation_loss"])["epoch"]),
        "history": history,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    return model, diagnostics


def predict_rows(
    model: BeatAMLVirtualCellModel,
    prepared: PreparedBeatAML,
    frame: pd.DataFrame,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Predict standardized AUC residuals for response-table rows."""

    model.eval()
    features = torch.as_tensor(prepared.features_scaled, dtype=torch.float32, device=device)
    patient_indices = torch.as_tensor(
        frame["patient_index"].to_numpy(copy=True), dtype=torch.long
    )
    drug_indices = torch.as_tensor(
        frame["drug_index"].to_numpy(copy=True), dtype=torch.long
    )
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            patient_batch = patient_indices[start : start + batch_size].to(device)
            drug_batch = drug_indices[start : start + batch_size].to(device)
            prediction, _ = model(features[patient_batch], drug_batch)
            predictions.append(prediction.cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)


def predict_all_drugs(
    model: BeatAMLVirtualCellModel,
    scaled_features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict all known drugs and return standardized residuals and latent states."""

    model.eval()
    patient_tensor = torch.as_tensor(scaled_features, dtype=torch.float32, device=device)
    n_patients = len(scaled_features)
    n_drugs = model.config.n_drugs
    with torch.no_grad():
        latent = model.encode_patient(patient_tensor)
        patient_repeated = patient_tensor.repeat_interleave(n_drugs, dim=0)
        drug_index = torch.arange(n_drugs, device=device).repeat(n_patients)
        response_z, _ = model(patient_repeated, drug_index)
    return response_z.reshape(n_patients, n_drugs).cpu().numpy(), latent.cpu().numpy()


def fit_blend_alpha(
    observed_auc: np.ndarray,
    baseline_auc: np.ndarray,
    neural_auc: np.ndarray,
) -> float:
    """Fit one shrinkage coefficient for neural residuals on validation data."""

    residual = neural_auc - baseline_auc
    denominator = float(np.dot(residual, residual))
    if denominator < 1e-12:
        return 0.0
    alpha = float(np.dot(residual, observed_auc - baseline_auc) / denominator)
    return float(np.clip(alpha, 0.0, 1.5))


def _safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    if len(x_array) < 3 or np.std(x_array) < 1e-12 or np.std(y_array) < 1e-12:
        return float("nan")
    result = spearmanr(x_array, y_array)
    return float(result.statistic)


def evaluate_predictions(
    frame: pd.DataFrame,
    prediction_column: str,
    minimum_drugs_per_patient: int = 5,
    hit_k: int = 5,
) -> dict[str, object]:
    """Compute regression and within-patient drug-ranking metrics."""

    observed = frame["auc"].to_numpy(dtype=float)
    predicted = frame[prediction_column].to_numpy(dtype=float)
    errors = predicted - observed
    patient_correlations: list[float] = []
    hit_rates: list[float] = []
    evaluable_patients = 0
    for _, group in frame.groupby("patient_id"):
        if len(group) < minimum_drugs_per_patient:
            continue
        correlation = _safe_spearman(group["auc"], group[prediction_column])
        if math.isfinite(correlation):
            patient_correlations.append(correlation)
        k = min(hit_k, len(group))
        actual_hits = set(group.nsmallest(k, "auc")["drug_id"])
        predicted_hits = set(group.nsmallest(k, prediction_column)["drug_id"])
        hit_rates.append(len(actual_hits & predicted_hits) / k)
        evaluable_patients += 1

    correlation_array = np.asarray(patient_correlations, dtype=float)
    return {
        "n_rows": int(len(frame)),
        "n_patients": int(frame["patient_id"].nunique()),
        "n_evaluable_patients": int(evaluable_patients),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "global_pearson": float(np.corrcoef(observed, predicted)[0, 1]),
        "median_patient_spearman": float(np.nanmedian(correlation_array)),
        "mean_patient_spearman": float(np.nanmean(correlation_array)),
        "patient_spearman_q25": float(np.nanquantile(correlation_array, 0.25)),
        "patient_spearman_q75": float(np.nanquantile(correlation_array, 0.75)),
        "patient_spearman_positive_fraction": float(np.mean(correlation_array > 0)),
        f"mean_top_{hit_k}_overlap": float(np.mean(hit_rates)),
    }


def checkpoint_payload(
    model: BeatAMLVirtualCellModel,
    prepared: PreparedBeatAML,
    diagnostics: dict[str, object],
    blend_alpha: float | None = None,
) -> dict[str, object]:
    """Build a self-contained, inference-ready checkpoint dictionary."""

    return {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "feature_columns": prepared.feature_columns,
        "drug_ids": prepared.drug_ids,
        "feature_mean": prepared.feature_mean,
        "feature_scale": prepared.feature_scale,
        "drug_mean": prepared.drug_mean,
        "drug_scale": prepared.drug_scale,
        "drug_train_count": prepared.drug_train_count,
        "patient_split": {
            "train": list(prepared.split.train),
            "validation": list(prepared.split.validation),
            "test": list(prepared.split.test),
        },
        "training_diagnostics": diagnostics,
        "blend_alpha": blend_alpha,
        "scope": "BeatAML baseline patient state plus single-drug ex-vivo AUC",
    }


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[BeatAMLVirtualCellModel, dict[str, object]]:
    """Load a pilot checkpoint without requiring training code."""

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**payload["model_config"])
    model = BeatAMLVirtualCellModel(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload
