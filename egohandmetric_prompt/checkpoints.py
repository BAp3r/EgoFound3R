from __future__ import annotations


from pathlib import Path
from typing import Any
import torch
from .configs import (
    MarkerModelConfig,
    marker_architecture_descriptor,
    marker_architecture_hash,
)


MARKER_CHECKPOINT_SCHEMA_VERSION = "egofound3r_marker_v1"


MARKER_CHECKPOINT_NAMESPACE = "egofound3r.marker.v1"


_CHECKPOINT_METADATA_KEYS = (
    "checkpoint_schema_version",
    "checkpoint_namespace",
    "temporal_architecture",
    "temporal_architecture_hash",
)


ROOT_GEOMETRY_RUNTIME_CONTRACT = {
    "version": "fp32_geometry",
    "geometry_dtype": "float32_minimum",
    "reported_confidence": "anchor_times",
}


def _manifest_difference(
    saved_descriptor: dict[str, object],
    expected_descriptor: dict[str, object],
) -> list[str]:
    return sorted(
        key
        for key in set(saved_descriptor) | set(expected_descriptor)
        if saved_descriptor.get(key) != expected_descriptor.get(key)
    )


def validate_marker_checkpoint_contract(
    payload: dict[str, Any],
    model_config: MarkerModelConfig,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if "model" not in payload:
        raise ValueError(
            "the marker checkpoint requires a full payload with immutable metadata; "
            "weights-only or partial loading is forbidden."
        )
    if not isinstance(payload["model"], dict):
        raise ValueError("marker checkpoint model state must be a dictionary")
    missing_metadata = [
        key for key in _CHECKPOINT_METADATA_KEYS if key not in payload
    ]
    if missing_metadata:
        raise ValueError(
            f"marker checkpoint is missing immutable metadata: {missing_metadata}"
        )
    if (
        payload.get("checkpoint_schema_version")
        != MARKER_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(
            "checkpoint_schema_version does not match the released "
            f"topology; expected {MARKER_CHECKPOINT_SCHEMA_VERSION!r}."
        )
    if (
        payload.get("checkpoint_namespace")
        != MARKER_CHECKPOINT_NAMESPACE
    ):
        raise ValueError(
            "marker checkpoint namespace does not match the current topology"
        )
    expected_descriptor = marker_architecture_descriptor(model_config)
    saved_descriptor = payload.get("temporal_architecture")
    if not isinstance(saved_descriptor, dict):
        raise ValueError(
            "marker checkpoint temporal architecture manifest is invalid"
        )
    differences = _manifest_difference(saved_descriptor, expected_descriptor)
    if differences:
        raise ValueError(
            "marker checkpoint temporal topology does not match the current "
            f"config: {differences}"
        )
    if payload.get("temporal_architecture_hash") != marker_architecture_hash(
        model_config
    ):
        raise ValueError(
            "marker checkpoint temporal architecture hash does not match"
        )
    geometry_contract = payload.get("root_geometry_contract")
    if geometry_contract is not None and geometry_contract != ROOT_GEOMETRY_RUNTIME_CONTRACT:
        raise ValueError("checkpoint Root geometry contract is unknown or incompatible")


def load_marker_model_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    model_config: MarkerModelConfig | None = None,
    align_model_floating_dtype: bool = False,
) -> int:
    if type(align_model_floating_dtype) is not bool:
        raise TypeError("align_model_floating_dtype must be a bool")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload is malformed: {checkpoint_path}")
    if model_config is not None:
        validate_marker_checkpoint_contract(payload, model_config)
    raw_weights: Any = payload["model"] if "model" in payload else payload
    if not isinstance(raw_weights, dict):
        raise ValueError(f"checkpoint carries no loadable model weights: {checkpoint_path}")
    if any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in raw_weights.items()
    ):
        raise ValueError("checkpoint model state must be a string -> tensor mapping")

    if align_model_floating_dtype:
        checkpoint_floating_dtypes = {
            value.dtype for value in raw_weights.values() if value.is_floating_point()
        }
        if len(checkpoint_floating_dtypes) > 1:
            formatted = sorted(str(dtype) for dtype in checkpoint_floating_dtypes)
            raise ValueError(
                "checkpoint model state has mixed floating dtypes and cannot define "
                f"one inference compute dtype: {formatted}"
            )
        if checkpoint_floating_dtypes:
            model.to(dtype=next(iter(checkpoint_floating_dtypes)))

    state_dict = model.state_dict()
    expected_weights = {
        name: value
        for name, value in state_dict.items()
        if not name.startswith("backbone.")
    }
    unexpected_keys = sorted(set(raw_weights) - set(expected_weights))
    missing_keys = sorted(set(expected_weights) - set(raw_weights))
    if unexpected_keys or missing_keys:
        details: list[str] = []
        if missing_keys:
            details.append(f"missing required non-backbone state keys: {missing_keys[:20]}")
        if unexpected_keys:
            details.append(f"unexpected checkpoint state keys: {unexpected_keys[:20]}")
        raise ValueError("; ".join(details))

    mismatched_shapes = [
        f"{name}: checkpoint={tuple(raw_weights[name].shape)} model={tuple(expected.shape)}"
        for name, expected in expected_weights.items()
        if tuple(raw_weights[name].shape) != tuple(expected.shape)
    ]
    if mismatched_shapes:
        raise ValueError(f"checkpoint state shape mismatch: {mismatched_shapes[:20]}")
    mismatched_dtypes = [
        f"{name}: checkpoint={raw_weights[name].dtype} model={expected.dtype}"
        for name, expected in expected_weights.items()
        if raw_weights[name].dtype != expected.dtype
    ]
    if mismatched_dtypes:
        raise ValueError(f"checkpoint state dtype mismatch: {mismatched_dtypes[:20]}")

    updated_state = dict(state_dict)
    updated_state.update(raw_weights)
    model.load_state_dict(updated_state)
    return len(raw_weights)


def marker_model_floating_dtype(model: torch.nn.Module) -> torch.dtype:
    floating_dtypes = {
        tensor.dtype
        for name, tensor in model.named_parameters()
        if tensor.is_floating_point()
        if not (
            name.startswith(("backbone.camera_head.", "backbone.dense_head."))
            and not tensor.requires_grad
        )
    }
    if not floating_dtypes:
        floating_dtypes = {
            tensor.dtype for tensor in model.buffers() if tensor.is_floating_point()
        }
    if not floating_dtypes:
        return torch.get_default_dtype()
    if len(floating_dtypes) != 1:
        formatted = sorted(str(dtype) for dtype in floating_dtypes)
        raise ValueError(f"marker model has mixed floating dtypes: {formatted}")
    return next(iter(floating_dtypes))
