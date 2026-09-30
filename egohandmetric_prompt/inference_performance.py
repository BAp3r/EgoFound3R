from __future__ import annotations


import time
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
import torch


class ForwardMeasurement:
    def __init__(self, device: torch.device):
        self.device = device
        self.seconds = 0.0
        self.peak_allocated_bytes: int | None = None
        self.peak_reserved_bytes: int | None = None

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._started_at = time.perf_counter()
        return self

    def __exit__(self, exc_type, _value, _traceback):
        if exc_type is not None:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.seconds = time.perf_counter() - self._started_at
        if self.device.type == "cuda":
            self.peak_allocated_bytes = int(torch.cuda.max_memory_allocated(self.device))
            self.peak_reserved_bytes = int(torch.cuda.max_memory_reserved(self.device))


def build_inference_performance(
    forwards: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    model_dtype: torch.dtype,
    output_r_frames: int,
    unique_h_frames: int | None,
    unique_g_frames: int | None,
    input_r_fps: float,
    pipeline_wall_seconds: float,
    end_to_end_wall_seconds: float | None,
    overlay_enabled: bool,
) -> dict[str, Any]:
    if not forwards:
        raise ValueError("inference performance requires at least one forward")
    elapsed = np.array([float(item["forward_seconds"]) for item in forwards])
    total_seconds = float(elapsed.sum())

    def total_frames(key):
        values = [item.get(key) for item in forwards]
        return sum(int(value) for value in values) if all(value is not None for value in values) else None

    def rate(count, seconds):
        return float(count) / seconds if count is not None and seconds is not None and seconds > 0 else None

    def peak(key):
        if device.type != "cuda":
            return None
        values = [item.get(key) for item in forwards]
        return max(int(value) for value in values) if all(value is not None for value in values) else None

    allocated = peak("peak_memory_bytes")
    reserved = peak("peak_reserved_memory_bytes")
    properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    processed_h, processed_g = total_frames("T_H"), total_frames("T_G")
    return {
        "schema_version": "egofound3r.inference_performance.v1",
        "device": str(device),
        "device_name": properties.name if properties is not None else device.type,
        "device_total_memory_bytes": int(properties.total_memory) if properties is not None else None,
        "model_dtype": str(model_dtype),
        "input_r_fps": float(input_r_fps),
        "output_r_frames": int(output_r_frames),
        "unique_h_frames": unique_h_frames,
        "unique_g_frames": unique_g_frames,
        "processed_h_frames": processed_h,
        "processed_g_frames": processed_g,
        "forward_calls": len(forwards),
        "includes_first_forward": True,
        "extra_warmup_forwards": 0,
        "forward_seconds_total": total_seconds,
        "forward_seconds_first": float(elapsed[0]),
        "forward_seconds_mean": float(elapsed.mean()),
        "forward_seconds_p50": float(np.percentile(elapsed, 50)),
        "forward_seconds_p95": float(np.percentile(elapsed, 95)),
        "pipeline_wall_seconds": float(pipeline_wall_seconds),
        "end_to_end_wall_seconds": end_to_end_wall_seconds,
        "forward_output_fps": rate(output_r_frames, total_seconds),
        "forward_processed_h_fps": rate(processed_h, total_seconds),
        "forward_processed_g_fps": rate(processed_g, total_seconds),
        "pipeline_output_fps": rate(output_r_frames, pipeline_wall_seconds),
        "end_to_end_output_fps": rate(output_r_frames, end_to_end_wall_seconds),
        "end_to_end_realtime_speedup": rate(output_r_frames / input_r_fps, end_to_end_wall_seconds),
        "forward_peak_allocated_bytes": allocated,
        "forward_peak_reserved_bytes": reserved,
        "forward_peak_allocated_gib": None if allocated is None else allocated / 1024**3,
        "forward_peak_reserved_gib": None if reserved is None else reserved / 1024**3,
        "overlay_enabled": bool(overlay_enabled),
    }
