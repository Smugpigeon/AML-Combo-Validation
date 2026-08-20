"""Patient-state and perturbation models for AML research."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__all__ = ["BeatAMLVirtualCellModel", "ModelConfig"]

if TYPE_CHECKING:
    from .beataml_pilot import BeatAMLVirtualCellModel, ModelConfig


def __getattr__(name: str) -> Any:
    """Load the Torch pilot lazily so lightweight firewall tools stay lightweight."""
    if name in {"BeatAMLVirtualCellModel", "ModelConfig"}:
        from .beataml_pilot import BeatAMLVirtualCellModel, ModelConfig

        return {
            "BeatAMLVirtualCellModel": BeatAMLVirtualCellModel,
            "ModelConfig": ModelConfig,
        }[name]
    raise AttributeError(name)
