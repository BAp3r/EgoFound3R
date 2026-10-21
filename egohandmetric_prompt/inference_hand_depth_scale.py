import argparse
from dataclasses import asdict, dataclass
import math
import time
import numpy as np
import torch
from scipy.ndimage import map_coordinates, maximum_filter, minimum_filter
from .inference_contracts import reconstruct_multirate_metric_scene_outputs
from .inference_root_z import _weighted_median
from .models.camera_temporal_refiner import (
    DeterministicTemporalInterpolator,
    matrix_to_rotation_6d,
)


@dataclass(frozen=True)
class HandDepthScaleOptions:
    enabled: bool = True
    min_confidence: float = .25
    visibility_threshold: float = .7
    max_change: float = 2.
    overlap_propagation: bool = True

    def validate(self):
        if not math.isfinite(self.min_confidence) or not 0 < self.min_confidence <= 1:
            raise ValueError("hand-depth scale confidence must be in (0,1]")
        if not math.isfinite(self.visibility_threshold) or not 0 < self.visibility_threshold <= 1:
            raise ValueError("hand-depth visibility must be in (0,1]")
        if not math.isfinite(self.max_change) or self.max_change < 1:
            raise ValueError("hand-depth maximum scale change must be finite and >=1")


def add_hand_depth_scale_arguments(parser):
    parser.add_argument("--hand-depth-scale", action=argparse.BooleanOptionalAction, default=True,
                        help="Fuse reliable metric Hand/raw Depth scale with MetricHead (default on).")
    parser.add_argument("--hand-depth-scale-min-confidence", type=float, default=.25)
    parser.add_argument("--hand-depth-scale-visibility", type=float, default=.7)
    parser.add_argument("--hand-depth-scale-max-change", type=float, default=2.,
                        help="Limit weak-fusion prior influence to this ratio around the Hand estimate.")
    parser.add_argument("--overlap-depth-scale", action=argparse.BooleanOptionalAction, default=True,
                        help="Resolve multi-chunk metric scales using exact-frame raw Depth overlaps.")


def hand_depth_scale_options_from_args(args):
    options = HandDepthScaleOptions(args.hand_depth_scale, args.hand_depth_scale_min_confidence,
                                    args.hand_depth_scale_visibility, args.hand_depth_scale_max_change,
                                    args.overlap_depth_scale)
    options.validate()
    return options


def spatial_log_statistics(logs, weights, cells):
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    medians = np.array([_weighted_median(logs[inverse==i], weights[inverse==i])
                        for i in range(int(inverse.max())+1)])
    median = float(np.median(medians))
    mad = float(np.median(abs(medians-median)))
    spread = float(np.diff(np.quantile(medians, [.1,.9]))[0])
    return median, mad, spread


def estimate_frame_scale(vertices, visibility, root_confidence, depth, confidence, K, options):
    row = {"valid": False, "reason": "insufficient_support", "point_count": 0, "cell_count": 0,
           "factor": None, "log_mad": None, "quality": 0.}
    if not np.isfinite(root_confidence) or root_confidence < options.min_confidence:
        row["reason"] = "low_hand_confidence"
        return row
    if depth.ndim != 2 or min(depth.shape) < 4 or confidence.shape != depth.shape:
        row["reason"] = "invalid_depth_grid"
        return row
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1], atol=1e-6) or abs(np.linalg.det(K)) < 1e-10):
        row["reason"] = "invalid_K"
        return row
    height, width = depth.shape
    xyz = np.asarray(vertices, dtype=np.float64)
    finite = np.isfinite(xyz).all(-1) & (xyz[:, 2] > 1e-4) & np.isfinite(visibility)
    finite &= visibility >= options.visibility_threshold
    xyz, vis = xyz[finite], visibility[finite]
    if not len(xyz):
        return row
    projected = xyz @ K.T
    uv = projected[:, :2] / projected[:, 2:3]
    in_frame = (uv[:, 0] >= 1) & (uv[:, 0] <= width-2) & (uv[:, 1] >= 1) & (uv[:, 1] <= height-2)
    xyz, vis, uv = xyz[in_frame], vis[in_frame], uv[in_frame]
    usable = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence > 0)
    safe = np.where(usable, depth, 0.)
    interior = minimum_filter(usable.astype(float), size=3)
    edge = maximum_filter(safe, size=3) - minimum_filter(safe, size=3)
    coords = [uv[:, 1], uv[:, 0]]
    sample = lambda value: map_coordinates(value, coords, order=1, mode="constant", cval=0.)
    sampled_depth, sampled_conf = sample(safe), sample(np.where(usable, confidence, 0.))
    mask = (sample(interior) > .999) & (sampled_depth > 0) & (sampled_conf > 0)
    mask &= sample(edge) < .08 * sampled_depth
    xyz, vis, uv = xyz[mask], vis[mask], uv[mask]
    sampled_depth, sampled_conf = sampled_depth[mask], sampled_conf[mask]
    row["point_count"] = len(xyz)
    if len(xyz) < 16:
        return row
    logs = np.log(xyz[:, 2]) - np.log(sampled_depth)
    weights = vis * np.clip(sampled_conf / max(float(np.median(sampled_conf)), 1e-8), .25, 1.)
    median = _weighted_median(logs, weights)
    mad = _weighted_median(abs(logs-median), weights)
    cell_median, cell_mad, spread = spatial_log_statistics(logs, weights, np.floor(uv/8).astype(int))
    row.update(log_mad=mad, cell_log_mad=cell_mad, cell_log_spread=spread,
               cell_log_disagreement=abs(median-cell_median))
    if abs(median-cell_median)>math.log(1.1):
        row["reason"] = "inconsistent_regions"
        return row
    if mad > .15:
        row["reason"] = "inconsistent_points"
        return row
    keep = abs(logs-median) <= max(math.log(1.1), 3*mad)
    retained_fraction = float(keep.mean())
    uv, logs, weights = uv[keep], logs[keep], weights[keep]
    cells = len(np.unique(np.floor(uv / 8).astype(int), axis=0))
    row.update(point_count=len(logs), cell_count=cells)
    if len(logs) < 16 or cells < 4 or np.linalg.norm(np.ptp(uv, axis=0) / [width, height]) < .04:
        row["reason"] = "narrow_support"
        return row
    factor = math.exp(_weighted_median(logs, weights))
    quality = min(1., len(logs)/32) * math.exp(-(mad/.1)**2) * min(1., root_confidence/.5)
    row.update(retained_fraction=retained_fraction, spatial_quality=retained_fraction*math.exp(-(spread/.2)**2))
    row.update(valid=True, reason="accepted", factor=factor, quality=quality)
    return row


def fuse_scale_observations(prior, prior_valid, observations, options):
    options.validate()
    prior_valid = bool(prior_valid and math.isfinite(prior) and prior > 0)
    report = {"prior_factor": float(prior) if math.isfinite(prior) else None,
              "prior_valid": prior_valid, "hand_factor": None, "frame_count": len(observations),
              "frame_log_mad": None, "quality": 0., "hand_weight": 0., "bounded": False,
              "factor": float(prior) if prior_valid else 1., "valid": prior_valid,
              "strength": "none", "reason": "no_hand_reference",
              "prior_factor_effective": float(prior) if prior_valid else None,
              "prior_influence_limited": False, "corroborated": False, "reliable": False,
              "confidence": .1 if prior_valid else 0.,
              "log_uncertainty": math.log(2.) if prior_valid else None}
    if not observations:
        return report
    logs = np.log([o["factor"] for o in observations])
    weights = np.asarray([o["quality"] for o in observations])
    membership = abs(logs[:, None]-logs[None, :]) <= math.log(1.2)
    support = membership @ weights
    keep = membership[int(np.argmax(support))]
    if float(weights[keep].sum()) < .6 * float(weights.sum()):
        report["reason"] = "no_dominant_hand_cluster"
        return report
    kept = [o for o,k in zip(observations,keep) if k]
    span = max(o["time"] for o in kept)-min(o["time"] for o in kept)
    median = _weighted_median(logs[keep], weights[keep])
    mad = _weighted_median(abs(logs[keep]-median), weights[keep])
    hand = math.exp(median)
    mean_quality = float(np.mean(weights[keep]))
    times = np.array([o["time"] for o in kept])
    early = times <= (times.min()+times.max())/2
    temporal_difference = abs(_weighted_median(logs[keep][early],weights[keep][early])
                              -_weighted_median(logs[keep][~early],weights[keep][~early])) if (~early).any() else 0.
    side_difference = max(o.get("side_log_disagreement",0.) for o in kept)
    spatial_quality = float(np.mean([o.get("spatial_quality",1.) for o in kept]))
    spatial_spread = float(np.median([o.get("cell_log_spread",0.) for o in kept]))
    quality = min(1., len(kept)/3) * max(.5, min(1., span/.2)) * math.exp(-(mad/.15)**2) * mean_quality
    quality *= spatial_quality * math.exp(-(temporal_difference/.2)**2-(side_difference/.2)**2)
    strong = len(kept) >= 3 and span >= .2-1e-8 and mad <= .1 and mean_quality*spatial_quality >= .55
    strong &= temporal_difference <= .1 and side_difference <= .1
    corroborated = len(kept)>=2 and span>=.1-1e-8 and temporal_difference<=.1 and side_difference<=.1 and mad<=.1
    uncertainty = max(.03 if strong else .1, 1.4826*mad, spatial_spread, temporal_difference, side_difference, .15*(1-quality))
    report.update(hand_factor=hand, frame_count=len(kept), quality=quality,
                  frame_log_mad=mad, strength="strong" if strong else "weak",
                  temporal_log_disagreement=temporal_difference, side_log_disagreement=side_difference,
                  cell_log_spread=spatial_spread, corroborated=bool(corroborated),
                  log_uncertainty=uncertainty, confidence=quality*math.exp(-(uncertainty/math.log(1.25))**2),
                  reliable=bool(strong))
    if prior_valid:
        report["prior_hand_ratio"]=min(hand/prior,prior/hand)
    if strong or corroborated or not prior_valid:
        report.update(factor=hand, valid=True, hand_weight=1.,
                      prior_factor_effective=None,
                      reason="strong_hand" if strong else "weak_hand_consensus" if corroborated else "weak_hand_invalid_prior")
        return report
    weight = .8 + .2 * min(1., quality/.55)
    prior_log = math.log(prior)
    limited_prior_log = float(np.clip(prior_log, median-math.log(options.max_change), median+math.log(options.max_change)))
    factor = math.exp((1-weight)*limited_prior_log + weight*median)
    report.update(factor=factor, hand_weight=weight, bounded=limited_prior_log != prior_log,
                  prior_factor_effective=math.exp(limited_prior_log), prior_influence_limited=limited_prior_log != prior_log,
                  prior_hand_ratio=min(hand/prior, prior/hand), reason="weak_hand_fused")
    return report


def apply_hand_depth_scale(outputs, frame_map, query_map, fps, options):
    if options is None:
        return outputs, None
    options.validate()
    report = {"schema_version": "egofound3r.hand_depth_scale.v1",
              "options": asdict(options), "batches": [], "seconds": 0.}
    if not options.enabled:
        return outputs, report
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Hand-Depth scale FPS must be positive finite")
    device = outputs["raw_to_meter_pred"].device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    required = ("depth_raw_global", "depth_conf_global", "intrinsics_global", "dense_vertex_xyz",
                "dense_vertex_visibility_logits", "root_translation_reported_confidence", "root_translation_valid")
    missing = [key for key in required if key not in outputs]
    if missing:
        raise ValueError(f"Hand-Depth scale requires raw model outputs: {missing}")
    if outputs["dense_vertex_xyz"].shape[-2] != 195:
        raise ValueError("Hand-Depth scale uses original 195 markers, never 778 interpolated points")
    array = lambda tensor: tensor.detach().float().cpu().numpy()
    depth = array(outputs["depth_raw_global"])
    if depth.ndim == 5:
        depth = depth[..., 0]
    confidence = array(outputs["depth_conf_global"])
    if confidence.ndim == 5:
        confidence = confidence[..., 0]
    K, vertices = array(outputs["intrinsics_global"]), array(outputs["dense_vertex_xyz"])
    visibility = array(outputs["dense_vertex_visibility_logits"].float().sigmoid())
    quality = array(outputs["root_translation_reported_confidence"])
    accepted = outputs.get("interpolation_hand_anchor_valid_high", outputs["root_translation_valid"]).detach().cpu().numpy()
    anchors = frame_map.global_anchor_indices.detach().cpu().numpy()
    present = frame_map.global_frame_present.detach().cpu().numpy()
    times = query_map.high_anchor_indices.detach().cpu().numpy()/fps
    prior = array(outputs["raw_to_meter_pred"])
    prior_valid = outputs["metric_scale_valid"].detach().cpu().numpy()
    for batch in range(len(prior)):
        observations, groups = [], []
        for slot in np.flatnonzero(present[batch]):
            high = anchors[batch, slot]
            frame_groups = []
            for side in range(2):
                if not accepted[batch, high, side]:
                    continue
                row = estimate_frame_scale(vertices[batch, high, side], visibility[batch, high, side],
                    float(quality[batch, high, side]), depth[batch, slot], confidence[batch, slot], K[batch, slot], options)
                row.update(global_slot=int(slot), high_slot=int(high), side=side,
                           time=float(times[batch, high]))
                groups.append(row)
                if row["valid"]:
                    frame_groups.append(row)
            if frame_groups:
                logs = np.log([g["factor"] for g in frame_groups])
                if np.ptp(logs) > .3:
                    continue
                weights = np.array([g["quality"] for g in frame_groups])
                observations.append({"time": float(times[batch, high]), "factor": math.exp(_weighted_median(logs, weights)),
                                     "quality": float(weights.mean()), "side_log_disagreement": float(np.ptp(logs)),
                                     "spatial_quality": float(np.mean([g["spatial_quality"] for g in frame_groups])),
                                     "cell_log_spread": float(np.max([g["cell_log_spread"] for g in frame_groups]))})
        fused = fuse_scale_observations(float(prior[batch]), bool(prior_valid[batch]), observations, options)
        fused["groups"] = groups
        report["batches"].append(fused)
    factor = outputs["raw_to_meter_pred"].new_tensor([r["factor"] for r in report["batches"]], dtype=torch.float32)
    valid = outputs["metric_scale_valid"].new_tensor([r["valid"] for r in report["batches"]])
    changed = (factor != outputs["raw_to_meter_pred"]) | (valid != outputs["metric_scale_valid"])
    result = dict(outputs)
    if bool(changed.any()):
        scaled_inputs = dict(outputs, interpolation_scene_metric_scale_factor=factor,
                             interpolation_scene_metric_scale_valid=valid,
                             metric_value_effective=outputs["pred_scene_scale_raw"]*factor)
        result = reconstruct_multirate_metric_scene_outputs(scaled_inputs, global_frame_present=frame_map.global_frame_present)
        result["metric_value_pred"] = outputs["metric_value_pred"]
        result["raw_to_meter_pred"] = outputs["raw_to_meter_pred"]
        camera, camera_valid, _, exact = DeterministicTemporalInterpolator()._camera_base(
            result["camera_pose_metric_global"], frame_map, query_map, result["camera_pose_metric_valid_global"])
        for key in ("camera_pose_base_full", "camera_pose_interpolated_full", "camera_pose_refined_full", "camera_pose_metric_full"):
            old = outputs.get(key, camera)
            result[key] = torch.where(changed[:, None, None, None], camera, old)
        for key in ("camera_refined_valid_full", "camera_interpolation_valid_full", "camera_pose_metric_valid_full"):
            result[key] = camera_valid
        result["camera_anchor_exact_mask_full"] = exact
        high_camera = camera.gather(1, query_map.high_anchor_indices[..., None, None].expand(-1, -1, 4, 4))
        result["camera_pose_refined_high"] = high_camera
        result["camera_refined_valid_high"] = camera_valid.gather(1, query_map.high_anchor_indices)
        result["camera_anchor_exact_mask_high"] = exact.gather(1, query_map.high_anchor_indices)
        vector = torch.cat((matrix_to_rotation_6d(high_camera[..., :3, :3]), high_camera[..., :3, 3]), dim=-1)
        for key in ("camera_extrinsic_vector_base_high", "camera_extrinsic_vector_refined_high"):
            if key in outputs:
                result[key] = vector
    result["raw_to_meter_fused"] = factor
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report["seconds"] = time.perf_counter()-started
    return result, report
