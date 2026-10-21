from __future__ import annotations


import math
import os
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
import cv2
import numpy as np
import torch


INPUT_RESIZE_MODES = ("keep_aspect", "stretch")


MODEL_INPUT_ALIGNMENT = 16


def add_input_resize_arguments(parser) -> None:
    parser.add_argument(
        "--input-resize-mode",
        choices=INPUT_RESIZE_MODES,
        default="keep_aspect",
        help="full-frame input mode: keep_aspect preserves the aspect ratio, stretch resizes directly to the target",
    )
    parser.add_argument("--input-height", type=int, default=None, help="target model input height; defaults to the config value")
    parser.add_argument("--input-width", type=int, default=None, help="target model input width; defaults to the config value")


def resolve_requested_input_size(
    *,
    default_size_hw: tuple[int, int],
    input_height: int | None,
    input_width: int | None,
) -> tuple[int, int]:
    height = int(default_size_hw[0] if input_height is None else input_height)
    width = int(default_size_hw[1] if input_width is None else input_width)
    if height < 1 or width < 1:
        raise ValueError("input height and width must be positive")
    return height, width


def resolve_full_frame_input_size(
    source_size_hw: tuple[int, int],
    requested_size_hw: tuple[int, int],
    *,
    mode: str,
    alignment: int = MODEL_INPUT_ALIGNMENT,
) -> tuple[tuple[int, int], torch.Tensor]:
    if mode not in INPUT_RESIZE_MODES:
        raise ValueError(f"input resize mode must be one of {INPUT_RESIZE_MODES}")
    source_height, source_width = (int(source_size_hw[0]), int(source_size_hw[1]))
    requested_height, requested_width = (int(requested_size_hw[0]), int(requested_size_hw[1]))
    if min(source_height, source_width, requested_height, requested_width) < 1:
        raise ValueError("source and requested input sizes must be positive")
    if mode == "stretch":
        target_height, target_width = requested_height, requested_width
        if target_height % alignment or target_width % alignment:
            raise ValueError(
                f"stretch input size {(target_height, target_width)} must be divisible by patch alignment {alignment}"
            )
    else:
        scale = min(requested_height / float(source_height), requested_width / float(source_width))
        target_height = max(int(round(source_height * scale / alignment)) * alignment, alignment)
        target_width = max(int(round(source_width * scale / alignment)) * alignment, alignment)
    scale_x = target_width / float(source_width)
    scale_y = target_height / float(source_height)
    affine = torch.tensor(
        [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    return (target_height, target_width), affine


def resize_full_frame_rgb(rgb: np.ndarray, target_size_hw: tuple[int, int]) -> np.ndarray:
    target_height, target_width = (int(target_size_hw[0]), int(target_size_hw[1]))
    return cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def _open_video(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"Cannot open video: {path}")
    return capture


def _video_info(path: Path, max_frames: int | None) -> tuple[int, float, tuple[int, int]]:
    capture = _open_video(path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        size = (int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        reported = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        count = int(reported) if math.isfinite(reported) and reported > 0 else 0
        if count:
            count = min(count, max_frames) if max_frames is not None else count
            sought = capture.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
            ok, _ = capture.read() if sought else (False, None)
            limited = max_frames is not None and count == max_frames
            if ok and (limited or not capture.grab()):
                return count, fps, size
    finally:
        capture.release()
    capture = _open_video(path)
    count = 0
    try:
        while (max_frames is None or count < max_frames) and capture.grab():
            count += 1
    finally:
        capture.release()
    if not count:
        raise ValueError(f"Video has no readable frames: {path}")
    return count, fps, size


class _FrameWindow(Sequence):
    def __init__(self, source: WindowedVideoFrames, indices: range):
        self.source, self.indices = source, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        selected = self.indices[index]
        if isinstance(selected, range):
            return _FrameWindow(self.source, selected)
        return self.source[selected]


class WindowedVideoFrames(Sequence):
    def __init__(
        self,
        path: Path,
        *,
        target_height: int,
        target_width: int,
        max_frames: int | None = None,
        resize_mode: str = "keep_aspect",
    ):
        if min(target_height, target_width) < 1 or (max_frames is not None and max_frames < 1):
            raise ValueError("Video dimensions and max_frames must be positive")
        self.path = Path(path)
        self.requested_image_size_hw = (int(target_height), int(target_width))
        self.frame_count, self.fps, self.source_size_hw = _video_info(self.path, max_frames)
        self.image_size_hw, self.input_affine = resolve_full_frame_input_size(
            self.source_size_hw,
            self.requested_image_size_hw,
            mode=resize_mode,
        )
        self.input_resize_mode = str(resize_mode)
        self._capture = None
        self._next_frame = 0
        self._cache: OrderedDict[int, tuple[np.ndarray, torch.Tensor | None]] = OrderedDict()
        self._capacity = 128
        self._required: set[int] | None = None
        self._bytes = 0
        self._closed = False
        self.stats = {"decoded_frames": 0, "retrieved_frames": 0, "cache_hits": 0,
                      "backward_seeks": 0, "peak_cached_frames": 0, "peak_cached_bytes": 0}

    def __len__(self):
        return self.frame_count

    def configure_cache(self, *, window_frames: int, required_indices: Sequence[int]) -> None:
        if window_frames < 1:
            raise ValueError("window_frames must be positive")
        self._capacity = int(window_frames)
        self._required = {int(index) for index in required_indices}
        self._trim()

    def _discard(self, index: int) -> None:
        rgb, tensor = self._cache.pop(index)
        self._bytes -= rgb.nbytes + (tensor.numel() * tensor.element_size() if tensor is not None else 0)

    def _trim(self) -> None:
        while len(self._cache) > self._capacity:
            self._discard(next(iter(self._cache)))
        self.stats["peak_cached_frames"] = max(self.stats["peak_cached_frames"], len(self._cache))
        self.stats["peak_cached_bytes"] = max(self.stats["peak_cached_bytes"], self._bytes)

    def __getitem__(self, index):
        if self._closed:
            raise RuntimeError("Video reader is closed")
        selected = range(len(self))[index]
        if isinstance(selected, range):
            if selected:
                low, high = min(selected[0], selected[-1]), max(selected[0], selected[-1])
                for cached in list(self._cache):
                    if not low <= cached <= high:
                        self._discard(cached)
            return _FrameWindow(self, selected)
        index = selected
        if index in self._cache:
            self.stats["cache_hits"] += 1
            self._cache.move_to_end(index)
        else:
            if self._capture is None:
                self._capture = _open_video(self.path)
            if index < self._next_frame:
                self.stats["backward_seeks"] += 1
                if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    self._capture.release()
                    self._capture = _open_video(self.path)
                    self._next_frame = 0
                else:
                    self._next_frame = index
            while self._next_frame <= index:
                current = self._next_frame
                if not self._capture.grab():
                    raise RuntimeError(f"Video ended before declared frame {current}: {self.path}")
                self._next_frame += 1
                self.stats["decoded_frames"] += 1
                if current != index and self._required is not None and current not in self._required:
                    continue
                ok, bgr = self._capture.retrieve()
                if not ok:
                    raise RuntimeError(f"Cannot decode frame {current}: {self.path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = resize_full_frame_rgb(rgb, self.image_size_hw)
                if current in self._cache:
                    self._discard(current)
                self._cache[current] = (rgb, None)
                self._bytes += rgb.nbytes
                self.stats["retrieved_frames"] += 1
                self._trim()
        rgb, tensor = self._cache[index]
        if tensor is None:
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
            self._cache[index] = (rgb, tensor)
            self._bytes += tensor.numel() * tensor.element_size()
            self._trim()
        return tensor

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._cache.clear()
        self._bytes = 0
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def iter_video_rgb(
    path: Path,
    *,
    frame_count: int,
    image_size_hw: tuple[int, int],
    resize_mode: str = "keep_aspect",
) -> Iterator[np.ndarray]:
    capture = _open_video(path)
    try:
        for frame in range(frame_count):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended before frame {frame}: {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            source_size = (int(rgb.shape[0]), int(rgb.shape[1]))
            actual_size, _ = resolve_full_frame_input_size(source_size, image_size_hw, mode=resize_mode)
            yield resize_full_frame_rgb(rgb, actual_size)
    finally:
        capture.release()


class OverlayVideoWriter:
    def __init__(self, output_dir: Path, fps: float, *, artifact_name: str = "overlay", save_frame_images: bool = True):
        if not artifact_name or Path(artifact_name).name != artifact_name:
            raise ValueError("overlay artifact name must be a plain file stem")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("overlay FPS must be positive and finite")
        self.output_dir, self.fps = Path(output_dir), float(fps)
        self.path = self.output_dir / f"{artifact_name}.mp4"
        self.frame_dir = self.output_dir / f"{artifact_name}_frames"
        self.save_frame_images = save_frame_images
        self.count = 0
        self._process = None
        self._temporary = None
        self._stderr = None
        self._shape = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if save_frame_images:
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
            raise ValueError("overlay frames must be uint8 BGR [H, W, 3]")
        if self._shape is not None and frame.shape != self._shape:
            raise ValueError("overlay frame dimensions changed")
        if self._process is None:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise RuntimeError("Browser-compatible MP4 export requires FFmpeg with libx264; install ffmpeg.")
            self._shape = frame.shape
            with tempfile.NamedTemporaryFile(dir=self.output_dir, prefix=f".{self.path.stem}.", suffix=".mp4", delete=False) as handle:
                self._temporary = Path(handle.name)
            self._stderr = tempfile.TemporaryFile()
            self._process = subprocess.Popen(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                 "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{frame.shape[1]}x{frame.shape[0]}",
                 "-framerate", str(self.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                 "-preset", "fast", "-crf", "18", "-threads", "2", "-pix_fmt", "yuv420p",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(self._temporary)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr,
            )
        if self.save_frame_images:
            target = self.frame_dir / f"{self.count:06d}.png"
            with tempfile.NamedTemporaryFile(dir=self.frame_dir, prefix=f".{target.stem}.", suffix=".png", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                if not cv2.imwrite(str(temporary), frame):
                    raise RuntimeError(f"cannot write overlay frame: {target}")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        self._process.stdin.write(frame.tobytes())
        self.count += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _value, _traceback):
        try:
            if self._process is not None:
                if exc_type is not None:
                    self._process.terminate()
                try:
                    self._process.stdin.close()
                except BrokenPipeError:
                    pass
                if exc_type is not None:
                    try:
                        self._process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait()
                else:
                    status = self._process.wait()
                    if status:
                        self._stderr.seek(0)
                        error = self._stderr.read().decode("utf-8", errors="replace")[-2000:]
                        raise RuntimeError(f"H.264 export failed for {self.path}: {error}")
                    os.replace(self._temporary, self.path)
        finally:
            if self._temporary is not None:
                self._temporary.unlink(missing_ok=True)
            if self._stderr is not None:
                self._stderr.close()


def write_overlay_streams(output_dir: Path, rows: Iterable[Mapping[str, np.ndarray]], fps: float, *, save_frame_images: bool = True) -> None:
    with ExitStack() as stack:
        writers = None
        count = 0
        for row in rows:
            if writers is None:
                writers = {name: stack.enter_context(OverlayVideoWriter(output_dir, fps, artifact_name=name, save_frame_images=save_frame_images)) for name in row}
            if set(row) != set(writers):
                raise ValueError("overlay streams changed during export")
            for name, frame in row.items():
                writers[name].write(frame)
            count += 1
            if count % 300 == 0:
                print(f"Overlay export: {count} frames", flush=True)
