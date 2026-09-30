from __future__ import annotations


from egohandmetric_prompt.configs import (
    MarkerModelConfig,
    ProjectConfig,
)
from egohandmetric_prompt.models.aggregator_rate_adapter import (
    FinalHandPromptBridge,
    FinalMetricPromptBridge,
    HandGlobalTokenBridge,
    HandPromptBlockPooler,
    MetricFeatureBlockPooler,
)
from egohandmetric_prompt.models.camera_temporal_refiner import (
    DeterministicTemporalInterpolator,
)
from egohandmetric_prompt.data.marker_mesh import marker_faces_for_count
from egohandmetric_prompt.data.marker_vertex_joint_map import (
    marker_vertex_joint_weights_for_count,
)
from egohandmetric_prompt.models import (
    HandHead,
    HandInViewHead,
    HandKeypointLocalizationHead,
    HandPromptAdapter,
    HandViewFusionMLP,
    MarkerModel,
    MetricHead,
    TemporalHandPromptAdapter,
    VggtOmegaFrozenWrapper,
    ViewMemoryProjector,
)
from egohandmetric_prompt.models.marker_model import MultiRateMarkerComponents


def _build_hand_components(model_config: MarkerModelConfig) -> dict[str, object]:
    return {
        "hand_in_view_head": HandInViewHead(
            dino_dim=model_config.dino_dim,
            hidden_dim=model_config.in_view_hidden_dim,
        ),
        "hand_view_fusion": HandViewFusionMLP(
            prompt_dim=model_config.prompt_dim,
            hand_feature_dim=model_config.hand_feature_dim,
        ),
        "view_memory_projector": ViewMemoryProjector(
            view_feature_dim=model_config.hand_feature_dim,
            hand_feature_dim=model_config.hand_feature_dim,
        ),
    }


def _build_multirate_components(
    model_config: MarkerModelConfig,
) -> MultiRateMarkerComponents:
    bridge_dim = model_config.global_bridge_dim
    pool_hidden_dim = model_config.global_prompt_pool_hidden_dim
    return MultiRateMarkerComponents(
        hand_global_token_bridge=HandGlobalTokenBridge(
            query_dim=model_config.dino_dim,
            global_token_dim=model_config.aggregator_dim,
            output_dim=model_config.aggregator_dim,
            bridge_dim=bridge_dim,
            num_heads=model_config.global_bridge_heads,
            num_layers=model_config.global_bridge_layers,
        ),
        hand_prompt_pooler=HandPromptBlockPooler(
            prompt_dim=model_config.prompt_dim,
            hidden_dim=pool_hidden_dim,
        ),
        metric_prompt_pooler=MetricFeatureBlockPooler(
            metric_feature_dim=model_config.prompt_dim,
            hidden_dim=pool_hidden_dim,
        ),
        metric_feature_pooler=MetricFeatureBlockPooler(
            metric_feature_dim=model_config.hand_feature_dim,
            hidden_dim=pool_hidden_dim,
        ),
        final_hand_prompt_bridge=FinalHandPromptBridge(
            query_dim=model_config.prompt_dim,
            bridge_dim=bridge_dim,
            num_heads=model_config.global_bridge_heads,
            num_layers=model_config.global_bridge_layers,
        ),
        final_metric_prompt_bridge=FinalMetricPromptBridge(
            query_dim=model_config.prompt_dim,
            bridge_dim=bridge_dim,
            num_heads=model_config.global_bridge_heads,
            num_layers=model_config.global_bridge_layers,
        ),
        temporal_interpolator=DeterministicTemporalInterpolator(),
    )


def _vertex_face_residual_kwargs(model_config: MarkerModelConfig) -> dict[str, object]:
    if not model_config.enable_vertex_face_residual:
        return {
            "marker_faces": None,
            "enable_vertex_face_residual": False,
            "vertex_face_residual_layers": (),
            "vertex_face_residual_gate_init": model_config.vertex_face_residual_gate_init,
            "vertex_face_residual_strength": model_config.vertex_face_residual_strength,
        }
    marker_faces = marker_faces_for_count(model_config.num_markers)
    if marker_faces.numel() == 0:
        return {
            "marker_faces": None,
            "enable_vertex_face_residual": False,
            "vertex_face_residual_layers": (),
            "vertex_face_residual_gate_init": model_config.vertex_face_residual_gate_init,
            "vertex_face_residual_strength": model_config.vertex_face_residual_strength,
        }
    return {
        "marker_faces": marker_faces,
        "enable_vertex_face_residual": True,
        "vertex_face_residual_layers": model_config.vertex_face_residual_layers,
        "vertex_face_residual_gate_init": model_config.vertex_face_residual_gate_init,
        "vertex_face_residual_strength": model_config.vertex_face_residual_strength,
    }






def build_runtime_marker_model(project_config: ProjectConfig) -> MarkerModel:
    runtime_config = project_config.marker_runtime
    if not runtime_config.vggt_checkpoint_path:
        raise ValueError("marker_runtime.vggt_checkpoint_path must not be empty")
    backbone = VggtOmegaFrozenWrapper.from_vendor(
        checkpoint_path=runtime_config.vggt_checkpoint_path,
        image_height=runtime_config.image_height,
        image_width=runtime_config.image_width,
        mid_layer_index=project_config.marker_model.prompt_injection_layer - 1,
        depth_confidence_max_value=runtime_config.depth_confidence_max_value,
    )
    keypoint_head = HandKeypointLocalizationHead(
        in_channels=project_config.marker_model.dino_dim,
        hidden_channels=project_config.marker_model.detection_hidden_dim,
    )
    hand_adapter = HandPromptAdapter(
        dino_dim=project_config.marker_model.dino_dim,
        agg_dim=project_config.marker_model.aggregator_dim,
        prompt_dim=project_config.marker_model.prompt_dim,
        hand_feature_dim=project_config.marker_model.hand_feature_dim,
        num_layers=project_config.marker_model.adapter_num_layers,
        num_heads=project_config.marker_model.adapter_num_heads,
        mlp_ratio=project_config.marker_model.adapter_mlp_ratio,
    )
    temporal_adapter = TemporalHandPromptAdapter(
        prompt_dim=project_config.marker_model.prompt_dim,
        num_layers=project_config.marker_model.temporal_prompt_layers,
        num_heads=project_config.marker_model.temporal_prompt_heads,
        mlp_ratio=project_config.marker_model.temporal_prompt_mlp_ratio,
    )
    hand_head = HandHead(
        prompt_dim=project_config.marker_model.prompt_dim,
        num_markers=project_config.marker_model.num_markers,
        num_queries=project_config.marker_model.marker_query_count,
        hand_feature_dim=hand_adapter.hand_feature_dim,
        root_num_layers=project_config.marker_model.root_head_num_layers,
        joint_num_layers=project_config.marker_model.joint_head_num_layers,
        vertex_num_layers=project_config.marker_model.vertex_head_num_layers,
        num_heads=project_config.marker_model.hand_head_num_heads,
        mlp_ratio=project_config.marker_model.hand_head_mlp_ratio,
        vertex_joint_weights=marker_vertex_joint_weights_for_count(project_config.marker_model.num_markers),
        **_vertex_face_residual_kwargs(project_config.marker_model),
    )
    metric_head = MetricHead(
        prompt_dim=project_config.marker_model.prompt_dim,
        hand_feature_dim=hand_adapter.hand_feature_dim,
        metric_scale_min_value=runtime_config.metric_scale_min_value,
        metric_scale_max_value=runtime_config.metric_scale_max_value,
    )
    multirate_components = _build_multirate_components(project_config.marker_model)
    return MarkerModel(
        backbone,
        keypoint_head,
        hand_adapter,
        temporal_adapter,
        hand_head,
        metric_head=metric_head,
        keypoint_peak_threshold=project_config.marker_model.keypoint_peak_threshold,
        multirate_components=multirate_components,
        **_build_hand_components(project_config.marker_model),
    )
