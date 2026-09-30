from __future__ import annotations


from dataclasses import dataclass
import math
from egohandmetric_prompt.multirate_contracts import (
    MAX_GLOBAL_STRIDE,
    MIN_GLOBAL_STRIDE,
    PRODUCTION_GLOBAL_STRIDE,
)


DEFAULT_R_FPS = 30.0


DEFAULT_H_FPS = 10.0


DEFAULT_CLIP_SECONDS = 2.0


DEFAULT_OVERLAP_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class TemporalRateContract:
    r_fps: float = DEFAULT_R_FPS
    h_fps: float = DEFAULT_H_FPS
    global_stride: int = PRODUCTION_GLOBAL_STRIDE

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.r_fps)) or float(self.r_fps) <= 0.0:
            raise ValueError("R fps must be finite and positive")
        if not math.isfinite(float(self.h_fps)) or float(self.h_fps) <= 0.0:
            raise ValueError("H fps must be finite and positive")
        if float(self.h_fps) > float(self.r_fps):
            raise ValueError("H fps cannot exceed R fps")
        if type(self.global_stride) is not int or not (
            MIN_GLOBAL_STRIDE <= self.global_stride <= MAX_GLOBAL_STRIDE
        ):
            raise ValueError(
                "H:G global_stride must be an integer in "
                f"[{MIN_GLOBAL_STRIDE}, {MAX_GLOBAL_STRIDE}]"
            )

    @property
    def g_fps(self) -> float:
        return float(self.h_fps) / float(self.global_stride)

    @property
    def r_to_h_ratio(self) -> float:
        return float(self.r_fps) / float(self.h_fps)

    def frame_count_for_seconds(self, seconds: float) -> int:
        if not math.isfinite(float(seconds)) or float(seconds) <= 0.0:
            raise ValueError("clip duration must be finite and positive")
        return max(int(round(float(seconds) * float(self.r_fps))), 1)

    def high_indices(self, frame_count: int) -> list[int]:
        if type(frame_count) is not int or frame_count < 1:
            raise ValueError("frame_count must be a positive integer")
        if frame_count == 1:
            return [0]
        maximum_tick = int(
            math.floor(
                (frame_count - 1) * float(self.h_fps) / float(self.r_fps)
                + 1e-9
            )
        )
        indices = {
            min(
                max(int(round(tick * float(self.r_fps) / float(self.h_fps))), 0),
                frame_count - 1,
            )
            for tick in range(maximum_tick + 1)
        }
        indices.add(0)
        indices.add(frame_count - 1)
        return sorted(indices)


def build_overlapping_r_windows(
    frame_count: int,
    *,
    clip_frames: int,
    overlap_frames: int,
) -> list[tuple[int, int]]:
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    if type(clip_frames) is not int or clip_frames < 2:
        raise ValueError("clip_frames must be an integer of at least 2")
    if type(overlap_frames) is not int or not 1 <= overlap_frames < clip_frames:
        raise ValueError("overlap_frames must satisfy 1 <= overlap < clip_frames")
    if frame_count <= clip_frames:
        return [(0, frame_count)]
    stride = clip_frames - overlap_frames
    starts = list(range(0, frame_count - clip_frames + 1, stride))
    final_start = frame_count - clip_frames
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, start + clip_frames) for start in starts]


def build_video_global_high_axis(
    rate_contract: TemporalRateContract,
    *,
    frame_count: int,
    windows: list[tuple[int, int]],
) -> list[int]:
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    high_indices = set(rate_contract.high_indices(frame_count))
    for start, end_exclusive in windows:
        if not (0 <= start < end_exclusive <= frame_count):
            raise ValueError("video windows must be non-empty and within the R axis")
        high_indices.add(int(start))
        high_indices.add(int(end_exclusive - 1))
    return sorted(high_indices)


def select_video_window_high_route(
    video_high_indices: list[int],
    *,
    r_start: int,
    r_end_exclusive: int,
    global_stride: int,
    video_global_phase: int,
) -> tuple[list[int], int, int]:
    if not video_high_indices or any(
        video_high_indices[index + 1] <= video_high_indices[index]
        for index in range(len(video_high_indices) - 1)
    ):
        raise ValueError("video_high_indices must be non-empty and strictly increasing")
    if not (0 <= r_start < r_end_exclusive):
        raise ValueError("video window must be a non-empty half-open interval")
    if type(global_stride) is not int or not MIN_GLOBAL_STRIDE <= global_stride <= MAX_GLOBAL_STRIDE:
        raise ValueError(
            f"global_stride must be an integer in [{MIN_GLOBAL_STRIDE}, {MAX_GLOBAL_STRIDE}]"
        )
    if type(video_global_phase) is not int or not 0 <= video_global_phase < global_stride:
        raise ValueError("video_global_phase must satisfy 0 <= phase < global_stride")

    selected = [
        (ordinal, frame_index)
        for ordinal, frame_index in enumerate(video_high_indices)
        if r_start <= frame_index < r_end_exclusive
    ]
    if not selected or selected[0][1] != r_start or selected[-1][1] != r_end_exclusive - 1:
        raise ValueError("video-global H axis must contain both window endpoints")
    first_ordinal = selected[0][0]
    expected_ordinals = list(range(first_ordinal, first_ordinal + len(selected)))
    if [ordinal for ordinal, _ in selected] != expected_ordinals:
        raise ValueError("window H indices must form a contiguous slice of the video-global H axis")
    local_high = [frame_index - r_start for _, frame_index in selected]
    local_phase = (video_global_phase - first_ordinal) % global_stride
    return local_high, local_phase, first_ordinal
