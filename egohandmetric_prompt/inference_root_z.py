from __future__ import annotations


import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from .hand_upsampling import attach_mano778_output
from .inference_contracts import attach_metric_prediction_contract, file_sha256


DEPTH_INPUT_SCHEMA = "egofound3r.metric_depth_input.v1"


ROOT_Z_SCHEMA = "egofound3r.root_z_postprocessing.v1"


@dataclass(frozen=True)
class RootZOptions:
    smooth: bool = False
    depth_source: str = "none"
    depth_file: Path | None = None
    smooth_strength: float = 1.0
    depth_strength: float = 1.0
    max_relative_change: float = 0.20
    visibility_threshold: float = 0.5

    @property
    def enabled(self) -> bool:
        return self.smooth or self.depth_source != "none"

    def validate(self) -> None:
        if self.depth_source not in ("none", "pred", "external"):
            raise ValueError("Root Z depth source must be none, pred, or external")
        if (self.depth_file is not None) != (self.depth_source == "external"):
            raise ValueError("--root-z-depth-file is required only for external Depth")
        if not np.isfinite(self.smooth_strength) or self.smooth_strength <= 0:
            raise ValueError("Root Z smoothing strength must be positive finite")
        if not np.isfinite(self.depth_strength) or self.depth_strength <= 0:
            raise ValueError("Root Z depth strength must be positive finite")
        if not 0 < self.max_relative_change < 1:
            raise ValueError("Root Z maximum relative change must be in (0, 1)")
        if not 0 <= self.visibility_threshold <= 1:
            raise ValueError("marker visibility threshold must be in [0, 1]")


def add_root_z_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root-z-smooth", action=argparse.BooleanOptionalAction, default=True,
                        help="Inference-only ray-preserving H Root Z smoothing (enabled by default).")
    parser.add_argument("--root-z-depth-source", choices=("none", "pred", "external"), default="none")
    parser.add_argument("--root-z-depth-file", type=Path, help="Registered, meter-valued external Depth sidecar.")
    parser.add_argument("--root-z-smooth-strength", type=float, default=1.0)
    parser.add_argument("--root-z-depth-strength", type=float, default=1.0)
    parser.add_argument("--root-z-max-relative-change", type=float, default=0.20)
    parser.add_argument("--root-z-visibility-threshold", type=float, default=0.5)


def root_z_options_from_args(args: argparse.Namespace) -> RootZOptions:
    options = RootZOptions(
        smooth=args.root_z_smooth, depth_source=args.root_z_depth_source,
        depth_file=args.root_z_depth_file, smooth_strength=args.root_z_smooth_strength,
        depth_strength=args.root_z_depth_strength,
        max_relative_change=args.root_z_max_relative_change,
        visibility_threshold=args.root_z_visibility_threshold,
    )
    options.validate()
    if options.enabled and getattr(args, "smooth_hand_root_h", False):
        raise ValueError("H Root smoothing and Root Z post-processing cannot be combined")
    return options


def _array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().double().numpy()


def _depth_row(row: dict[str, Any]) -> dict[str, np.ndarray]:
    from scipy.ndimage import maximum_filter, minimum_filter

    depth, valid, K = row["value_metric"], row["valid"], row["K"]
    if not isinstance(depth, torch.Tensor) or depth.ndim != 2 or min(depth.shape) < 3:
        raise ValueError("Depth must be a 2D tensor with at least 3 pixels per axis")
    if not isinstance(valid, torch.Tensor) or valid.shape != depth.shape or valid.dtype != torch.bool:
        raise ValueError("Depth validity must be a matching bool tensor")
    if not isinstance(K, torch.Tensor) or K.shape != (3, 3):
        raise ValueError("Depth must include its own 3x3 pixel-grid intrinsics")
    d, mask, calibration = _array(depth), valid.cpu().numpy(), _array(K)
    if not np.isfinite(calibration).all() or calibration[0, 0] <= 0 or calibration[1, 1] <= 0:
        raise ValueError("Depth focal lengths must be finite and positive")
    if not np.allclose(calibration[2], [0, 0, 1], atol=1e-6) or abs(np.linalg.det(calibration)) < 1e-10:
        raise ValueError("Depth intrinsics must be a nonsingular pinhole K")
    if not np.isfinite(d[mask]).all() or np.any(d[mask] <= 0):
        raise ValueError("valid Depth must be positive finite camera-Z meters")
    confidence = row.get("confidence")
    if confidence is not None and (not isinstance(confidence, torch.Tensor) or confidence.shape != depth.shape):
        raise ValueError("Depth confidence must align with its pixel grid")
    conf = np.ones_like(d) if confidence is None else _array(confidence)
    mask = mask & np.isfinite(conf) & (conf > 0)
    safe = np.where(mask, d, 0.0)
    edge = maximum_filter(safe, size=3) - minimum_filter(safe, size=3)
    return {"depth": safe, "valid": mask.astype(float), "K": calibration,
            "confidence": np.where(mask, conf, 0.0), "edge": edge}


def _depth_rows(payload: dict[str, Any], options: RootZOptions) -> dict[int, dict[str, Any]]:
    if options.depth_source == "none":
        return {}
    rows: list[dict[str, Any]] = []
    if options.depth_source == "pred":
        predictions = payload["metric_predictions"]
        depth, intrinsics = predictions["depth"], predictions["intrinsics"]
        if depth.get("unit") != "meter":
            raise ValueError("predicted Depth must already be metric; raw Depth is not accepted")
        indices = depth["frame_indices"]
        if not torch.equal(indices, intrinsics["global_frame_indices"]):
            raise ValueError("predicted Depth and K must use identical source frames")
        for i, frame in enumerate(indices.tolist()):
            if not bool(depth["valid"][i] & intrinsics["K_global_valid"][i]):
                continue
            value = depth["value_metric"][i]
            rows.append({"source_frame_index": frame, "value_metric": value,
                         "valid": torch.isfinite(value) & (value > 0),
                         "K": intrinsics["K_global"][i], "confidence": depth["confidence"][i]})
    else:
        external = torch.load(options.depth_file, map_location="cpu", weights_only=True)
        if external.get("schema_version") != DEPTH_INPUT_SCHEMA or external.get("unit") != "meter":
            raise ValueError("external Depth requires the versioned meter schema")
        if external.get("coordinate_frame") != "rgb_camera":
            raise ValueError("external Depth must already be registered to the RGB camera")
        if external.get("video_sha256") != file_sha256(Path(payload["video_path"])):
            raise ValueError("external Depth belongs to a different video")
        rows = external.get("depth", [])
        if not rows:
            raise ValueError("external Depth sidecar contains no observations")
    result = {}
    for row in rows:
        frame = row["source_frame_index"]
        if type(frame) is not int or frame < 0 or frame in result:
            raise ValueError("Depth source frame indices must be unique nonnegative integers")
        result[frame] = row
    return result


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, cumulative[-1] * 0.5)]])


def fit_visible_marker_depth(
    root: np.ndarray, vertices: np.ndarray, visibility: np.ndarray,
    row: dict[str, np.ndarray], options: RootZOptions,
) -> tuple[float, float, dict[str, Any]]:
    from scipy.ndimage import map_coordinates

    raw_z = float(root[2])
    diagnostics: dict[str, Any] = {"accepted": False, "reason": "insufficient_support", "support_count": 0,
                                   "residual_before_m": None, "residual_after_m": None, "mad_m": None}
    if not np.isfinite(root).all() or raw_z <= 1e-4 or not np.isfinite(vertices).all():
        diagnostics["reason"] = "invalid_geometry"
        return raw_z, 0.0, diagnostics
    offset, ray = vertices - root, root / raw_z
    h, w = row["depth"].shape

    def sample(z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points = offset + ray * z
        homogeneous = points @ row["K"].T
        uv = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-8)
        coordinates = [uv[:, 1], uv[:, 0]]
        sampled = {key: map_coordinates(row[key], coordinates, order=1, mode="constant", cval=0.0)
                   for key in ("depth", "valid", "confidence", "edge")}
        mask = (np.isfinite(visibility) & (visibility >= options.visibility_threshold)
                & (points[:, 2] > 1e-4) & np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 1) & (uv[:, 0] <= w - 2) & (uv[:, 1] >= 1) & (uv[:, 1] <= h - 2)
                & (sampled["valid"] > 0.999) & (sampled["depth"] > 0)
                & (sampled["edge"] < 0.04) & (sampled["confidence"] > 0))
        mask &= np.abs(sampled["depth"] - points[:, 2]) <= max(0.03, raw_z * options.max_relative_change * 1.5)
        conf = sampled["confidence"][mask]
        weight = visibility[mask] * np.clip(conf / max(float(np.median(conf)) if conf.size else 1.0, 1e-8), 0.25, 1.0)
        return sampled["depth"][mask] - offset[mask, 2], weight, uv[mask], mask

    initial, initial_weights, initial_uv, initial_mask = sample(raw_z)
    diagnostics["support_count"] = int(initial.size)
    if initial.size < 8:
        return raw_z, 0.0, diagnostics
    if np.linalg.norm(np.ptp(initial_uv, axis=0) / [w, h]) < 0.03:
        diagnostics["reason"] = "narrow_support"
        return raw_z, 0.0, diagnostics
    candidate = _weighted_median(initial, initial_weights)
    mad = _weighted_median(np.abs(initial - candidate), initial_weights)
    diagnostics["mad_m"] = mad
    if mad > 0.03:
        diagnostics["reason"] = "inconsistent_depth"
        return raw_z, 0.0, diagnostics
    lower, upper = raw_z * (1 - options.max_relative_change), raw_z * (1 + options.max_relative_change)
    candidate = float(np.clip(candidate, lower, upper))
    required_support = max(8, int(np.ceil(initial.size * 0.7)))
    for _ in range(2):
        observations, weights, _, mask = sample(candidate)
        if np.count_nonzero(mask & initial_mask) < required_support:
            diagnostics["reason"] = "lost_support"
            return raw_z, 0.0, diagnostics
        candidate = float(np.clip(_weighted_median(observations, weights), lower, upper))
    observations, weights, final_uv, mask = sample(candidate)
    if np.count_nonzero(mask & initial_mask) < required_support:
        diagnostics["reason"] = "lost_support"
        return raw_z, 0.0, diagnostics
    common_before, common_after = mask[initial_mask], initial_mask[mask]
    common_weights = initial_weights[common_before]
    before = _weighted_median(np.abs(initial[common_before] - raw_z), common_weights)
    after = _weighted_median(np.abs(observations[common_after] - candidate), common_weights)
    final_mad = _weighted_median(np.abs(observations - _weighted_median(observations, weights)), weights)
    mad = max(mad, final_mad)
    diagnostics.update(residual_before_m=before, residual_after_m=after, mad_m=mad)
    if mad > 0.03 or np.linalg.norm(np.ptp(final_uv[common_after], axis=0) / [w, h]) < 0.03:
        diagnostics["reason"] = "inconsistent_depth"
        return raw_z, 0.0, diagnostics
    if after >= before or candidate + float(offset[:, 2].min()) <= 1e-4:
        diagnostics["reason"] = "no_residual_improvement"
        return raw_z, 0.0, diagnostics
    quality = min(1.0, observations.size / 32.0) / (1.0 + (mad / 0.015) ** 2)
    diagnostics.update(accepted=True, reason="accepted", support_count=int(observations.size))
    return candidate, quality, diagnostics


def smooth_log_depth(
    timestamps: np.ndarray, raw_z: np.ndarray, quality: np.ndarray,
    depth_z: np.ndarray, depth_quality: np.ndarray, options: RootZOptions,
) -> np.ndarray:
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve

    raw = np.log(raw_z)
    depth = np.log(np.maximum(depth_z, 1e-4))
    prior_weights = np.clip(quality, 0.25, 1.0)
    depth_weights = depth_quality * options.depth_strength
    size = len(raw)
    penalty = diags(np.zeros(size), format="csc")
    if options.smooth and size >= 3:
        dt = np.diff(timestamps)
        left = 0.02 / ((dt[:-1] + dt[1:]) * dt[:-1])
        right = 0.02 / ((dt[:-1] + dt[1:]) * dt[1:])
        rows = np.repeat(np.arange(size - 2), 3)
        columns = np.column_stack((np.arange(size - 2), np.arange(1, size - 1), np.arange(2, size))).ravel()
        values = np.column_stack((left, -left - right, right)).ravel()
        operator = coo_matrix((values, (rows, columns)), shape=(size - 2, size)).tocsc()
        penalty = options.smooth_strength * (operator.T @ operator)
    estimate = raw.copy()
    for _ in range(4):
        robust = np.minimum(1.0, 0.03 / np.maximum(np.abs(estimate - raw), 1e-10))
        weights = prior_weights * robust
        estimate = spsolve(diags(weights + depth_weights, format="csc") + penalty,
                          weights * raw + depth_weights * depth)
    return np.clip(estimate, raw + np.log1p(-options.max_relative_change), raw + np.log1p(options.max_relative_change))


def _high_observations(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = payload["source_frame_indices"].long().cpu().tolist()
    high_source = payload["high_frame_indices"].long().cpu().tolist()
    if len(set(source)) != len(source) or any(b <= a for a, b in zip(source, source[1:])):
        raise ValueError("Root Z requires strictly increasing source frame indices")
    lookup = {frame: i for i, frame in enumerate(source)}
    if not high_source or any(frame not in lookup for frame in high_source) or any(b <= a for a, b in zip(high_source, high_source[1:])):
        raise ValueError("H observations must be increasing and belong to the R frame axis")
    indices = np.array([lookup[frame] for frame in high_source], dtype=int)
    meta = payload.get("hand_h_root_smoothing")
    if not isinstance(meta, dict):
        raise ValueError("Root Z requires saved H observation validity and quality")
    if meta.get("enabled"):
        raise ValueError("Root Z post-processing cannot follow H Root smoothing")
    clips = meta.get("clips", [meta])
    valid = np.zeros((len(high_source), 2), dtype=bool)
    quality = np.zeros((len(high_source), 2))
    counts = np.zeros((len(high_source), 2))
    high_lookup = {frame: i for i, frame in enumerate(high_source)}
    for clip in clips:
        for i, frame in enumerate(clip["frame_indices"].tolist()):
            if frame not in high_lookup:
                continue
            j = high_lookup[frame]
            eligible = clip["valid"][i].cpu().numpy().astype(bool)
            confidence = clip.get("root_translation_reported_confidence", clip["anchor_quality"])
            q = _array(confidence[i])
            eligible &= np.isfinite(q)
            valid[j] |= eligible
            quality[j] += np.where(eligible, np.clip(q, 0, 1), 0.0)
            counts[j] += eligible
    quality /= np.maximum(counts, 1)
    return indices, valid, quality


def apply_root_z_postprocessing(payload: dict[str, Any], options: RootZOptions | None) -> None:
    if options is None:
        return
    options.validate()
    if not options.enabled:
        return
    if "root_z_postprocessing" in payload:
        raise ValueError("Root Z post-processing cannot be applied twice; start from the raw result")
    from scipy.interpolate import PchipInterpolator

    predictions = payload["metric_predictions"]
    raw_hand = predictions["hand"]
    if raw_hand.get("unit") != "meter" or raw_hand.get("coordinate_frame") != "camera":
        raise ValueError("Root Z post-processing requires camera-space metric Hand")
    roots = _array(raw_hand["root_xyz_metric"])
    joints, vertices = _array(raw_hand["joint_xyz_metric"]), _array(raw_hand["vertex_xyz_metric"])
    valid_r = raw_hand["valid"].cpu().numpy().astype(bool)
    if roots.shape != (len(valid_r), 2, 3) or vertices.shape != (len(valid_r), 2, 195, 3):
        raise ValueError("Root Z requires the original two-sided 195-marker Hand output")
    if not np.allclose(roots[valid_r], joints[:, :, 0][valid_r], atol=1e-6, rtol=0):
        raise ValueError("Root and Joint 0 disagree")
    times = _array(payload["timestamps_sec"])
    if not np.isfinite(times).all() or len(times) != len(roots) or np.any(np.diff(times) <= 0):
        raise ValueError("Root Z requires finite, strictly increasing timestamps")
    high, observed, quality = _high_observations(payload)
    valid_r &= np.isfinite(roots).all(axis=-1) & (roots[:, :, 2] > 1e-4)
    valid_r &= np.isfinite(joints).all(axis=(-1, -2)) & np.isfinite(vertices).all(axis=(-1, -2))
    observed &= valid_r[high]
    rows = _depth_rows(payload, options)
    visibility = raw_hand.get("vertex_visibility_probability")
    if options.depth_source != "none" and (not isinstance(visibility, torch.Tensor) or visibility.shape != vertices.shape[:-1]):
        raise ValueError("this depth mode requires predicted marker visibility probabilities")
    visibility_array = None if visibility is None else _array(visibility)
    raw_z = roots[:, :, 2]
    depth_z, depth_quality = raw_z[high].copy(), np.zeros_like(quality)
    depth_diagnostics = []
    for h, r in enumerate(high):
        frame = int(payload["source_frame_indices"][int(r)])
        if frame not in rows or not observed[h].any():
            continue
        depth_row = _depth_row(rows[frame])
        for side in range(2):
            if not observed[h, side]:
                continue
            candidate, weight, info = fit_visible_marker_depth(roots[r, side], vertices[r, side], visibility_array[r, side], depth_row, options)
            depth_z[h, side], depth_quality[h, side] = candidate, weight
            depth_diagnostics.append(dict(info, source_frame_index=frame, side=side, candidate_z_m=candidate, weight=weight))
    processed_z = raw_z.copy()
    high_times = times[high]
    nominal_dt = float(np.median(np.diff(high_times))) if len(high) > 1 else 0.1
    for side in range(2):
        segments: list[list[int]] = []
        for h in np.flatnonzero(observed[:, side]):
            continuous = (segments and h == segments[-1][-1] + 1
                          and high_times[h] - high_times[segments[-1][-1]] <= nominal_dt * 2.5
                          and valid_r[high[segments[-1][-1]]:high[h] + 1, side].all())
            if continuous:
                segments[-1].append(int(h))
            else:
                segments.append([int(h)])
        for segment in segments:
            s = np.array(segment)
            smooth = options.smooth and len(s) >= 3
            if not smooth and not np.any(depth_quality[s, side] > 0):
                continue
            fitted = smooth_log_depth(high_times[s], raw_z[high[s], side], quality[s, side], depth_z[s, side], depth_quality[s, side], options)
            selection = np.arange(high[s[0]], high[s[-1]] + 1)
            if len(s) == 1:
                processed_z[selection, side] = np.exp(fitted)
            elif smooth:
                processed_z[selection, side] = np.exp(PchipInterpolator(high_times[s], fitted)(times[selection]))
            else:
                accepted = np.flatnonzero(depth_quality[s, side] > 0)
                controls = {int(i): float(fitted[i] - np.log(raw_z[high[s[i]], side])) for i in accepted}
                controls.setdefault(0, 0.0)
                controls.setdefault(len(s) - 1, 0.0)
                available = np.array([i for i in range(len(s))
                                      if int(payload["source_frame_indices"][int(high[s[i]])]) in rows])
                gap = float(np.median(np.diff(high_times[s[available]]))) if len(available) > 1 else nominal_dt
                support_span = max(gap, nominal_dt)
                before = np.flatnonzero(high_times[s] <= high_times[s[accepted[0]]] - support_span)
                after = np.flatnonzero(high_times[s] >= high_times[s[accepted[-1]]] + support_span)
                if len(before):
                    controls.setdefault(int(before[-1]), 0.0)
                if len(after):
                    controls.setdefault(int(after[0]), 0.0)
                for a, b in zip(accepted[:-1], accepted[1:]):
                    if high_times[s[b]] - high_times[s[a]] > 2.5 * max(gap, nominal_dt):
                        controls.setdefault(int(a + 1), 0.0)
                        controls.setdefault(int(b - 1), 0.0)
                control_indices = np.array(sorted(controls))
                corrections = np.array([controls[int(i)] for i in control_indices])
                processed_z[selection, side] *= np.exp(PchipInterpolator(high_times[s[control_indices]], corrections)(times[selection]))
    lower = raw_z * (1 - options.max_relative_change)
    upper = raw_z * (1 + options.max_relative_change)
    clipped = (processed_z < lower) | (processed_z > upper)
    processed_z = np.clip(processed_z, lower, upper)
    scale = np.divide(processed_z, raw_z, out=np.ones_like(processed_z), where=valid_r)
    new_roots = roots * scale[..., None]
    delta = new_roots - roots
    safe = (valid_r & np.isfinite(delta).all(axis=-1)
            & ((joints[..., 2] + delta[..., 2, None]) > 1e-4).all(axis=-1)
            & ((vertices[..., 2] + delta[..., 2, None]) > 1e-4).all(axis=-1))
    delta = np.where(safe[..., None], delta, 0.0)
    applied = safe & (np.max(np.abs(delta), axis=-1) > 1e-8)
    options_dict = asdict(options)
    options_dict["depth_file"] = str(options.depth_file) if options.depth_file is not None else None
    diagnostics = {"schema_version": ROOT_Z_SCHEMA, "options": options_dict,
                   "scope": "video_global",
                   "depth_file_sha256": file_sha256(options.depth_file) if options.depth_file is not None else None,
                   "high_frame_indices": payload["high_frame_indices"], "observed_high": torch.from_numpy(observed),
                   "root_xyz_raw_metric": raw_hand["root_xyz_metric"],
                   "root_delta_metric": torch.from_numpy(delta).float(), "applied": torch.from_numpy(applied),
                   "clipped": torch.from_numpy(clipped & valid_r), "depth_observations": depth_diagnostics,
                   "depth_source_frame_count": len(rows), "applied_count": int(applied.sum()),
                   "depth_accepted_count": sum(int(row["accepted"]) for row in depth_diagnostics)}
    payload["root_z_postprocessing"] = diagnostics
    payload["hand_raw"] = raw_hand
    predictions["hand_raw"] = raw_hand
    processed = dict(raw_hand)
    if applied.any():
        processed["root_xyz_metric"] = torch.from_numpy(roots + delta).float()
        processed["joint_xyz_metric"] = torch.from_numpy(joints + delta[..., None, :]).float()
        processed["vertex_xyz_metric"] = torch.from_numpy(vertices + delta[..., None, :]).float()
        missing_fill = payload.get("hand_missing_fill_full")
        if isinstance(missing_fill, torch.Tensor) and bool(missing_fill.any()):
            from .inference_hand_anchors import _camera_space_smooth_fill

            joint_fill, vertex_fill, _, _ = _camera_space_smooth_fill(
                processed["joint_xyz_metric"].unsqueeze(0),
                processed["vertex_xyz_metric"].unsqueeze(0),
                (processed["valid"].bool() & ~missing_fill.bool()).unsqueeze(0),
            )
            refill = missing_fill.bool()
            for field, filled in (("joint", joint_fill[0]), ("vertex", vertex_fill[0])):
                key = f"{field}_xyz_metric"
                processed[key] = torch.where(refill[..., None, None], filled, processed[key])
            processed["root_xyz_metric"] = processed["joint_xyz_metric"][..., 0, :]
            diagnostics["completion_rebuilt_full"] = refill
    payload["hand_postprocessed"] = processed
    predictions["hand_postprocessed"] = processed
    predictions["hand"] = processed
    diagnostics["root_xyz_processed_metric"] = processed["root_xyz_metric"]
    if "completion_rebuilt_full" in diagnostics:
        diagnostics["root_delta_metric"] = processed["root_xyz_metric"] - raw_hand["root_xyz_metric"]
        diagnostics["applied"] = diagnostics["root_delta_metric"].abs().amax(dim=-1) > 1e-8
        diagnostics["applied_count"] = int(diagnostics["applied"].sum())
    attach_metric_prediction_contract(payload, predictions)
    for field in ("joint", "vertex"):
        final_geometry = processed[f"{field}_xyz_metric"]
        for key in (
            f"dense_{field}_xyz_refined_full", f"dense_{field}_xyz_interpolated_full",
            f"hand_display_{field}_xyz_full", f"{field}_xyz_display",
        ):
            if key in payload:
                payload[key] = final_geometry
    if "hand_778" in payload:
        payload["hand_778_raw"] = payload["hand_778"]
        predictions["hand_778_raw"] = payload["hand_778"]
        attach_mano778_output(payload)
