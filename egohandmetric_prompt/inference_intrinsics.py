from __future__ import annotations


import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import torch
from .inference_contracts import file_sha256


ROOT_K_INPUT_SCHEMA = "egofound3r.root_intrinsics_input.v1"


ROOT_K_OUTPUT_SCHEMA = "egofound3r.root_intrinsics.v1"


@dataclass(frozen=True)
class RootIntrinsicsInput:
    source: str
    frame_indices: torch.Tensor
    K: torch.Tensor
    valid: torch.Tensor
    file_path: str
    file_sha256: str

    def select(self, frames: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        requested = torch.as_tensor(frames, dtype=torch.long)
        positions = torch.searchsorted(self.frame_indices, requested)
        bounded = positions.clamp(max=len(self.frame_indices) - 1)
        if not bool(((positions < len(self.frame_indices)) & (self.frame_indices[bounded] == requested)).all()):
            raise ValueError("provided Root K does not cover all requested source frames")
        return self.K[positions], self.valid[positions]


def add_root_k_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root-k-source", choices=("pred", "external"), default="pred",
                        help="Calibration used by the Root solver and Hand projection, not VGGT scene scale.")
    parser.add_argument("--root-k-file", type=Path, help="External full-frame pinhole K in a versioned JSON/PT sidecar.")


def validate_root_k_arguments(args: argparse.Namespace) -> None:
    if (args.root_k_file is not None) != (args.root_k_source == "external"):
        raise ValueError("--root-k-file is required only for --root-k-source external")


def _validate_k(K: torch.Tensor, valid: torch.Tensor) -> None:
    if K.ndim != 3 or K.shape[1:] != (3, 3) or valid.shape != K.shape[:1] or valid.dtype != torch.bool:
        raise ValueError("Root K must be [N,3,3] with bool validity [N]")
    selected = K[valid]
    if not torch.isfinite(selected).all() or (selected[:, 0, 0] <= 0).any() or (selected[:, 1, 1] <= 0).any():
        raise ValueError("valid Root K requires positive finite focal lengths")
    if not torch.allclose(selected[:, 2], K.new_tensor([0., 0., 1.]).expand(len(selected), 3), atol=1e-6, rtol=0):
        raise ValueError("Root K must be a pinhole calibration matrix")
    if (selected[:, 0, 1].abs() > 1e-6).any() or (selected[:, 1, 0].abs() > 1e-6).any():
        raise ValueError("Root solver requires zero skew; rectify the input first")


def load_root_intrinsics(
    *, source: str, file_path: Path | None,
    video_path: Path, source_frame_indices: Sequence[int], image_size_hw: tuple[int, int],
) -> RootIntrinsicsInput | None:
    if source not in ("pred", "external"):
        raise ValueError("Root K source must be pred or external")
    if (file_path is not None) != (source == "external"):
        raise ValueError("external Root K requires its own calibration file")
    if source == "pred":
        return None
    indices = torch.as_tensor(source_frame_indices, dtype=torch.long)
    if not len(indices) or (indices < 0).any() or (indices[1:] <= indices[:-1]).any():
        raise ValueError("Root K requests require increasing nonnegative source frame indices")
    assert file_path is not None
    if file_path.suffix.lower() == ".json":
        data = json.loads(file_path.read_text())
    elif file_path.suffix.lower() == ".pt":
        data = torch.load(file_path, map_location="cpu", weights_only=True)
    else:
        raise ValueError("external Root K requires a versioned .json or .pt sidecar")
    if not isinstance(data, dict):
        raise ValueError("external Root K sidecar must be a dictionary")
    if (data.get("schema_version") != ROOT_K_INPUT_SCHEMA or data.get("camera_model") != "pinhole"
            or data.get("coordinate_frame") != "rgb_camera"):
        raise ValueError("external Root K must declare the pinhole RGB-camera schema")
    if data.get("video_sha256") is not None and data["video_sha256"] != file_sha256(video_path):
        raise ValueError("external Root K belongs to a different video")
    size = data.get("image_size_hw")
    if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(x) is not int or x <= 0 for x in size):
        raise ValueError("external Root K must declare its full-frame calibration image_size_hw")
    if "K" not in data:
        raise ValueError("external Root K sidecar is missing K")
    K = torch.as_tensor(data["K"], dtype=torch.float32)
    if K.shape == (3, 3):
        declared_valid = torch.as_tensor(data.get("valid", True))
        if declared_valid.dtype != torch.bool or declared_valid.numel() != 1:
            raise ValueError("static Root K validity must be one bool")
        K = K.unsqueeze(0).expand(len(indices), 3, 3).clone()
        valid = declared_valid.reshape(1).expand(len(indices)).clone()
    else:
        if "source_frame_indices" not in data:
            raise ValueError("per-frame Root K requires source_frame_indices")
        frame_values = torch.as_tensor(data["source_frame_indices"])
        if frame_values.dtype != torch.long or frame_values.ndim != 1 or not len(frame_values):
            raise ValueError("per-frame Root K requires int64 source_frame_indices")
        if (frame_values < 0).any() or (frame_values[1:] <= frame_values[:-1]).any():
            raise ValueError("per-frame Root K source indices must be unique and increasing")
        valid = torch.as_tensor(data.get("valid", [True] * len(frame_values)))
        _validate_k(K, valid)
        if len(K) != len(frame_values):
            raise ValueError("per-frame Root K and source frame indices must align")
        temporary = RootIntrinsicsInput(source, frame_values, K, valid, str(file_path), "")
        K, valid = temporary.select(source_frame_indices)
        K = K.clone()
    K[:, 0] *= float(image_size_hw[1]) / size[1]
    K[:, 1] *= float(image_size_hw[0]) / size[0]
    selected_path = file_path
    _validate_k(K, valid)
    if not bool(valid.any()):
        raise ValueError("provided Root K contains no valid calibration for this video")
    K = torch.where(valid[:, None, None], K, torch.eye(3).expand_as(K)).float()
    return RootIntrinsicsInput(source, indices, K, valid, str(selected_path.resolve()), file_sha256(selected_path))


def root_solver_kwargs(
    calibration: RootIntrinsicsInput | None, frames: Sequence[int], device: torch.device,
) -> dict[str, torch.Tensor]:
    if calibration is None:
        return {}
    K, valid = calibration.select(frames)
    return {"root_solver_intrinsics": K.unsqueeze(0).to(device=device, dtype=torch.float32),
            "root_solver_intrinsics_valid": valid.unsqueeze(0).to(device=device, dtype=torch.bool)}


def _interpolate_k(K: torch.Tensor, valid: torch.Tensor, high: torch.Tensor, full: torch.Tensor):
    left = (torch.searchsorted(high, full, right=True) - 1).clamp(0, len(high) - 1)
    right = (left + 1).clamp(max=len(high) - 1)
    alpha = ((full - high[left]).float() / (high[right] - high[left]).clamp_min(1)).clamp(0, 1)
    values = torch.lerp(K[left], K[right], alpha[:, None, None])
    mask = torch.where(alpha == 0, valid[left], torch.where(alpha == 1, valid[right], valid[left] & valid[right]))
    return torch.where(mask[:, None, None], values, torch.eye(3).expand_as(values)), mask


def build_root_intrinsics_record(
    outputs: dict[str, Any], *, high_frame_indices: torch.Tensor, source_frame_indices: Sequence[int],
    image_size_hw: tuple[int, int], calibration: RootIntrinsicsInput | None,
) -> dict[str, Any] | None:
    normalized = outputs.get("root_solver_normalized_intrinsics")
    if not isinstance(normalized, torch.Tensor):
        if calibration is not None:
            raise ValueError("model did not report the provided Root K")
        return None
    normalized = normalized.detach().cpu().float()
    high = high_frame_indices.detach().cpu().long()
    valid = outputs["root_solver_intrinsics_valid"].detach().cpu().bool()
    if normalized.shape != (len(high), 4) or valid.shape != (len(high),):
        raise ValueError("reported Root K must align with the actual H axis")
    height, width = image_size_hw
    K = torch.eye(3).repeat(len(high), 1, 1)
    K[:, 0, 0], K[:, 1, 1] = normalized[:, 0] * (width - 1), normalized[:, 1] * (height - 1)
    K[:, 0, 2], K[:, 1, 2] = normalized[:, 2] * (width - 1), normalized[:, 3] * (height - 1)
    _validate_k(K, valid)
    full = torch.as_tensor(source_frame_indices, dtype=torch.long)
    if calibration is None:
        full_K, full_valid = _interpolate_k(K, valid, high, full)
    else:
        expected, expected_valid = calibration.select(high.tolist())
        if not torch.equal(valid, expected_valid) or not torch.allclose(K[valid], expected[valid], atol=1e-4, rtol=1e-6):
            raise ValueError("actual solver K differs from the explicitly provided calibration")
        full_K, full_valid = calibration.select(source_frame_indices)
    return {"schema_version": ROOT_K_OUTPUT_SCHEMA, "source": calibration.source if calibration else "pred",
            "K_high": K, "valid_high": valid, "high_frame_indices": high,
            "K_full": full_K, "valid_full": full_valid, "frame_indices": full,
            "image_size_hw": list(image_size_hw), "pixel_frame": "model_input_resized", "unit": "pixel",
            "input_file": calibration.file_path if calibration else None,
            "input_file_sha256": calibration.file_sha256 if calibration else None,
            "full_rate_rule": "provided_exact" if calibration else "linear_actual_H_calibration"}


def merge_root_intrinsics_records(
    records: list[dict[str, Any] | None], *, source_frame_indices: Sequence[int],
    high_frame_indices: Sequence[int], calibration: RootIntrinsicsInput | None,
) -> dict[str, Any] | None:
    if all(record is None for record in records):
        return None
    if any(record is None for record in records):
        raise ValueError("some chunks did not report Root K")
    actual = [record for record in records if record is not None]
    full = torch.as_tensor(source_frame_indices, dtype=torch.long)
    if calibration is not None:
        K, valid = calibration.select(source_frame_indices)
    else:
        K = torch.eye(3).repeat(len(full), 1, 1)
        valid, filled = torch.zeros(len(full), dtype=torch.bool), torch.zeros(len(full), dtype=torch.bool)
        for record in actual:
            positions = torch.searchsorted(full, record["frame_indices"])
            if (positions >= len(full)).any() or not torch.equal(full[positions], record["frame_indices"]):
                raise ValueError("Root K chunk indices do not belong to the output video")
            overlap = filled[positions]
            alpha = torch.ones(len(positions))
            count = int(overlap.sum())
            if count:
                alpha[overlap] = 0.5 - 0.5 * torch.cos(math.pi * (torch.arange(count) + 1) / (count + 1))
            incoming = record["valid_full"]
            both = valid[positions] & incoming
            mixed = torch.lerp(K[positions], record["K_full"], alpha[:, None, None])
            selected = torch.where(both[:, None, None], mixed,
                                   torch.where(incoming[:, None, None], record["K_full"], K[positions]))
            K[positions], valid[positions], filled[positions] = selected, valid[positions] | incoming, True
        if not bool(filled.all()):
            raise ValueError("Root K chunks left uncovered frames")
    high = torch.as_tensor(high_frame_indices, dtype=torch.long)
    high_positions = torch.searchsorted(full, high)
    result = dict(actual[0], K_high=K[high_positions], valid_high=valid[high_positions],
                  high_frame_indices=high, K_full=K, valid_full=valid, frame_indices=full, clips=actual)
    result["full_rate_rule"] = "provided_exact" if calibration else "overlap_cosine_calibration_blend"
    result["high_rate_rule"] = "merged_calibration; exact_per_forward_K_is_in_clips"
    return result


def attach_root_intrinsics(payload: dict[str, Any], record: dict[str, Any] | None) -> None:
    if record is None:
        return
    payload["root_intrinsics"] = record
    payload["metric_predictions"]["root_intrinsics"] = record
    payload["hand_projection_K_full"] = record["K_full"]
    payload["hand_projection_K_valid_full"] = record["valid_full"]
