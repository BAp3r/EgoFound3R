from __future__ import annotations


import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


VGGT_OMEGA_AGGREGATOR_LAYER_COUNT = 24


MARKER_TEMPORAL_ARCHITECTURE_ID = "egofound3r_multirate"


_MARKER_TOPOLOGY_FIELDS = (
    "temporal_architecture_id",
    "global_stride",
    "global_anchor_eval_phase",
    "global_prompt_pool_hidden_dim",
    "global_bridge_dim",
    "global_bridge_heads",
    "global_bridge_layers",
)


_MARKER_TOPOLOGY = {
    "temporal_architecture_id": MARKER_TEMPORAL_ARCHITECTURE_ID,
    "global_stride": 5,
    "global_anchor_eval_phase": 2,
    "global_prompt_pool_hidden_dim": 256,
    "global_bridge_dim": 256,
    "global_bridge_heads": 8,
    "global_bridge_layers": 2,
}


@dataclass(slots=True)
class ProjectInfoConfig:
    default_name: str = "egofound3r"
    seed: int = 0


@dataclass(slots=True)
class MarkerModelConfig:
    dino_dim: int = 1024
    aggregator_dim: int = 2048
    prompt_dim: int = 768
    hand_feature_dim: int = 768
    detection_hidden_dim: int = 256
    in_view_hidden_dim: int = 256
    num_markers: int = 195
    marker_query_count: int = 195
    prompt_injection_layer: int = 18
    adapter_num_layers: int = 6
    adapter_num_heads: int = 12
    adapter_mlp_ratio: int = 4
    temporal_prompt_layers: int = 2
    temporal_prompt_heads: int = 12
    temporal_prompt_mlp_ratio: int = 4
    root_head_num_layers: int = 1
    joint_head_num_layers: int = 5
    vertex_head_num_layers: int = 5
    enable_vertex_face_residual: bool = True
    vertex_face_residual_layers: list[int] = field(default_factory=lambda: [0, 2])
    vertex_face_residual_gate_init: float = 0.2
    vertex_face_residual_strength: float = 0.5
    hand_head_num_heads: int = 12
    hand_head_mlp_ratio: int = 4
    keypoint_peak_threshold: float = 0.5

    temporal_architecture_id: str = MARKER_TEMPORAL_ARCHITECTURE_ID
    global_stride: int = 5
    global_anchor_eval_phase: int = 2
    global_prompt_pool_hidden_dim: int = 256
    global_bridge_dim: int = 256
    global_bridge_heads: int = 8
    global_bridge_layers: int = 2

    def __post_init__(self) -> None:
        _validate_architecture_config(self)

@dataclass(slots=True)
class MarkerRuntimeConfig:
    image_height: int = 256
    image_width: int = 256
    vggt_checkpoint_path: str = ""
    depth_confidence_max_value: float = 64.0
    metric_scale_min_value: float = 0.01
    metric_scale_max_value: float = 100.0


@dataclass(slots=True)
class ProjectConfig:
    project: ProjectInfoConfig
    marker_model: MarkerModelConfig
    marker_runtime: MarkerRuntimeConfig


def active_marker_model_config(project_config: ProjectConfig) -> MarkerModelConfig:
    if not isinstance(project_config, ProjectConfig):
        raise TypeError("project_config must be a ProjectConfig")
    return project_config.marker_model


def _matches_fixed_topology_value(actual: object, expected: object) -> bool:
    if type(expected) is bool:
        return type(actual) is bool and actual is expected
    if type(expected) is int:
        return type(actual) is int and actual == expected
    if type(expected) is float:
        return type(actual) is float and math.isfinite(actual) and actual == expected
    return type(actual) is type(expected) and actual == expected


def _validate_architecture_config(config: MarkerModelConfig) -> None:
    if not isinstance(config, MarkerModelConfig):
        raise TypeError("config must be a MarkerModelConfig")
    if config.temporal_architecture_id != MARKER_TEMPORAL_ARCHITECTURE_ID:
        raise ValueError(
            "marker_model.temporal_architecture_id must be "
            f"{MARKER_TEMPORAL_ARCHITECTURE_ID!r}"
        )
    for field_name, expected in _MARKER_TOPOLOGY.items():
        actual = getattr(config, field_name)
        if not _matches_fixed_topology_value(actual, expected):
            raise ValueError(
                "marker_model topology is immutable: "
                f"{field_name} must be {expected!r}, got {actual!r}"
            )


def marker_architecture_descriptor(config: MarkerModelConfig) -> dict[str, object]:
    _validate_architecture_config(config)
    return {
        field_name: getattr(config, field_name)
        for field_name in _MARKER_TOPOLOGY_FIELDS
    }


def marker_architecture_hash(config: MarkerModelConfig) -> str:
    descriptor = marker_architecture_descriptor(config)
    encoded = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()




def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config_path() -> Path:
    return _package_root() / "configs" / "default.toml"


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _merge_nested_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _from_section(section_cls, section_data: dict | None):
    base = asdict(section_cls())
    if section_data:
        base.update(section_data)
    valid_keys = {field.name for field in fields(section_cls)}
    unknown_keys = set(base) - valid_keys
    if unknown_keys:
        raise ValueError(f"{section_cls.__name__} contains unknown fields: {sorted(unknown_keys)}")
    return section_cls(**base)


def _validate_marker_model_config(config: MarkerModelConfig) -> None:
    _validate_architecture_config(config)
    if not 1 <= config.prompt_injection_layer <= VGGT_OMEGA_AGGREGATOR_LAYER_COUNT:
        raise ValueError(
            "marker_model.prompt_injection_layer must be inside the frozen VGGT aggregator depth"
        )
    if float(config.keypoint_peak_threshold) != 0.5:
        raise ValueError("marker_model.keypoint_peak_threshold is fixed at 0.5")
    for name in (
        "dino_dim",
        "aggregator_dim",
        "prompt_dim",
        "hand_feature_dim",
        "detection_hidden_dim",
        "in_view_hidden_dim",
        "num_markers",
        "marker_query_count",
        "adapter_num_layers",
        "adapter_num_heads",
        "adapter_mlp_ratio",
        "temporal_prompt_layers",
        "temporal_prompt_heads",
        "temporal_prompt_mlp_ratio",
        "root_head_num_layers",
        "joint_head_num_layers",
        "vertex_head_num_layers",
        "hand_head_num_heads",
        "hand_head_mlp_ratio",
    ):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"marker_model.{name} must be positive")
    if int(config.marker_query_count) != int(config.num_markers):
        raise ValueError("marker_model.marker_query_count must equal num_markers")
    residual_layers = config.vertex_face_residual_layers
    if any(type(layer) is not int for layer in residual_layers):
        raise ValueError("marker_model.vertex_face_residual_layers must contain integers")
    if len(set(residual_layers)) != len(residual_layers):
        raise ValueError("marker_model.vertex_face_residual_layers must not contain duplicates")
    if config.enable_vertex_face_residual and any(
        layer < 0 or layer >= int(config.vertex_head_num_layers) for layer in residual_layers
    ):
        raise ValueError("marker_model.vertex_face_residual_layers must index vertex decoder blocks")
    if not math.isfinite(float(config.vertex_face_residual_gate_init)):
        raise ValueError("marker_model.vertex_face_residual_gate_init must be finite")
    if not math.isfinite(float(config.vertex_face_residual_strength)) or not 0.0 <= float(
        config.vertex_face_residual_strength
    ) <= 1.0:
        raise ValueError("marker_model.vertex_face_residual_strength must be finite and in [0, 1]")


def _validate_marker_runtime_config(config: MarkerRuntimeConfig) -> None:
    if int(config.image_height) <= 0 or int(config.image_width) <= 0:
        raise ValueError("marker_runtime image_height and image_width must be positive")
    maximum = float(config.depth_confidence_max_value)
    if not math.isfinite(maximum) or maximum <= 1.0 + 1e-6 or maximum > 1e30:
        raise ValueError(
            "marker_runtime.depth_confidence_max_value must be finite and lie in "
            "(1 + 1e-6, 1e30]"
        )
    minimum = float(config.metric_scale_min_value)
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("marker_runtime.metric_scale_min_value must be finite and positive")
    maximum_scale = float(config.metric_scale_max_value)
    if not math.isfinite(maximum_scale) or maximum_scale <= minimum:
        raise ValueError(
            "marker_runtime.metric_scale_max_value must be finite and greater than "
            "metric_scale_min_value"
        )


def load_project_config(config_path: str | Path | None = None) -> ProjectConfig:
    raw = _load_toml(default_config_path())
    if config_path is not None:
        override = _load_toml(Path(config_path))
        raw = _merge_nested_dict(raw, override)
    marker_model = _from_section(MarkerModelConfig, raw.get("marker_model"))
    _validate_marker_model_config(marker_model)
    marker_runtime = _from_section(MarkerRuntimeConfig, raw.get("marker_runtime"))
    _validate_marker_runtime_config(marker_runtime)
    return ProjectConfig(
        project=_from_section(ProjectInfoConfig, raw.get("project")),
        marker_model=marker_model,
        marker_runtime=marker_runtime,
    )
