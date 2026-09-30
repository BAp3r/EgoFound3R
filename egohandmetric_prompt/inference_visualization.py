from __future__ import annotations


import colorsys


import cv2
import numpy as np


INFERENCE_COLOR_MODES = (
    "Geometry",
    "Visibility",
    "Contact Class",
    "Contact Distance",
)


GEOMETRY_MARKER_RGB = ((47, 109, 176), (217, 120, 44))


GEOMETRY_JOINT_RGB = ((255, 245, 0), (255, 0, 255))


VISIBILITY_VISIBLE_RGB = (0, 255, 0)


VISIBILITY_HIDDEN_RGB = (255, 0, 0)


CONTACT_POSITIVE_RGB = (255, 31, 31)


CONTACT_NEGATIVE_RGB = (89, 182, 217)


SEMANTIC_UNKNOWN_RGB = (190, 190, 190)


CONTACT_DISTANCE_DISPLAY_MAX_M = 0.05


JOINT_OUTLINE_RGB = (34, 30, 26)


FACE_OCCLUDED_FACTOR = 0.35


OCCLUDED_ALPHA = 0.45


FACE_ALPHA = 0.30


STROKE_LIGHTEN = 0.65


GEOMETRY_BRIGHTNESS = 1.60


SIGNAL_BRIGHTNESS = 1.15


MARKER_DOT_RADIUS = 1


MARKER_EDGE_WIDTH = 1


JOINT_DOT_RADIUS = 2


JOINT_EDGE_WIDTH = 2


def _boost(rgb, saturation: float = 1.0, value: float = 1.0):
    red, green, blue = (channel / 255.0 for channel in rgb)
    hue, sat, val = colorsys.rgb_to_hsv(red, green, blue)
    sat = min(1.0, sat * saturation)
    val = min(1.0, val * value)
    red, green, blue = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(round(red * 255)), int(round(green * 255)), int(round(blue * 255)))


def _boost_array(colors: np.ndarray, value: float) -> np.ndarray:
    if value == 1.0:
        return colors
    unique, inverse = np.unique(colors.reshape(-1, 3), axis=0, return_inverse=True)
    boosted = np.array(
        [_boost(tuple(int(channel) for channel in row), 1.0, value) for row in unique],
        dtype=colors.dtype,
    )
    return boosted[inverse].reshape(colors.shape)


HAND_RGB = tuple(
    np.array(
        _boost(
            base,
            saturation=1.0 + 0.40 * (GEOMETRY_BRIGHTNESS - 1.0),
            value=GEOMETRY_BRIGHTNESS,
        ),
        dtype=np.uint8,
    )
    for base in GEOMETRY_MARKER_RGB
)


def _stroke_color(rgb) -> np.ndarray:
    base = np.asarray(rgb, dtype=np.float32)
    return np.rint(base * (1.0 - STROKE_LIGHTEN) + 255.0 * STROKE_LIGHTEN).astype(np.uint8)


def hand_face_color(side_index: int) -> np.ndarray:
    return HAND_RGB[side_index]


def _side_broadcast(rgb: np.ndarray, point_count: int) -> np.ndarray:
    return np.repeat(np.asarray(rgb, dtype=np.uint8)[None], int(point_count), axis=0)


def signal_face_colors(
    point_colors: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    face_indices = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if face_indices.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    endpoints = np.asarray(point_colors, dtype=np.float32)[face_indices]
    return np.rint(endpoints.mean(axis=1)).astype(np.uint8)


def face_alpha(visibility: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_indices = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if face_indices.size == 0:
        return np.zeros((0,), dtype=np.float32)
    face_visibility = np.asarray(visibility, dtype=np.float32)[face_indices].mean(axis=1)
    occluded = face_visibility < 0.5
    return np.where(occluded, FACE_ALPHA * FACE_OCCLUDED_FACTOR, FACE_ALPHA).astype(np.float32)


def _binary_colors(
    values: np.ndarray,
    *,
    positive_rgb: tuple[int, int, int],
    negative_rgb: tuple[int, int, int],
) -> np.ndarray:
    scalar = np.asarray(values, dtype=np.float32).reshape(-1)
    colors = np.repeat(
        np.asarray(SEMANTIC_UNKNOWN_RGB, dtype=np.uint8)[None],
        scalar.size,
        axis=0,
    )
    finite = np.isfinite(scalar)
    colors[finite & (scalar >= 0.5)] = positive_rgb
    colors[finite & (scalar < 0.5)] = negative_rgb
    return _boost_array(colors, SIGNAL_BRIGHTNESS)


def _contact_distance_colors(values_m: np.ndarray) -> np.ndarray:
    scalar = np.asarray(values_m, dtype=np.float32).reshape(-1)
    colors = np.repeat(
        np.asarray(SEMANTIC_UNKNOWN_RGB, dtype=np.uint8)[None],
        scalar.size,
        axis=0,
    )
    finite = np.isfinite(scalar)
    if not np.any(finite):
        return colors
    normalized = np.clip(
        scalar[finite] / CONTACT_DISTANCE_DISPLAY_MAX_M,
        0.0,
        1.0,
    )
    color_indices = np.rint(normalized * 255.0).astype(np.uint8).reshape(-1, 1)
    colors[finite] = cv2.applyColorMap(
        color_indices,
        cv2.COLORMAP_TURBO,
    ).reshape(-1, 3)[:, ::-1]
    return _boost_array(colors, SIGNAL_BRIGHTNESS)


def semantic_point_colors(
    mode: str,
    *,
    side_index: int,
    point_kind: str,
    visibility: np.ndarray,
    contact_probability: np.ndarray | None,
    contact_distance_m: np.ndarray | None,
) -> np.ndarray:
    if point_kind not in {"marker", "joint"}:
        raise ValueError(f"unsupported point kind: {point_kind}")
    if side_index not in {0, 1}:
        raise ValueError(f"unsupported hand side index: {side_index}")
    point_count = int(np.asarray(visibility).size)
    if mode == "Geometry":
        if point_kind == "joint":
            joint = _boost(
                GEOMETRY_JOINT_RGB[side_index],
                saturation=1.0 + 0.40 * (GEOMETRY_BRIGHTNESS - 1.0),
                value=GEOMETRY_BRIGHTNESS,
            )
            return _side_broadcast(joint, point_count)
        return _side_broadcast(_stroke_color(HAND_RGB[side_index]), point_count)
    if mode == "Visibility":
        return _binary_colors(
            visibility,
            positive_rgb=VISIBILITY_VISIBLE_RGB,
            negative_rgb=VISIBILITY_HIDDEN_RGB,
        )
    if mode == "Contact Class":
        if contact_probability is None:
            raise ValueError("Contact Class mode requires contact probabilities")
        return _binary_colors(
            contact_probability,
            positive_rgb=CONTACT_POSITIVE_RGB,
            negative_rgb=CONTACT_NEGATIVE_RGB,
        )
    if mode == "Contact Distance":
        if contact_distance_m is None:
            raise ValueError("Contact Distance mode requires metric contact distances")
        return _contact_distance_colors(contact_distance_m)
    raise ValueError(f"unsupported inference color mode: {mode}")


def edge_endpoint_colors(
    point_colors: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edge_indices = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if edge_indices.size == 0:
        empty = np.zeros((0, 3), dtype=np.uint8)
        return empty, empty
    colors = np.asarray(point_colors, dtype=np.uint8)
    return colors[edge_indices[:, 0]], colors[edge_indices[:, 1]]


class StrokeLayer:
    def __init__(self, height: int, width: int) -> None:
        self.height = int(height)
        self.width = int(width)
        self._color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._alpha = np.zeros((self.height, self.width), dtype=np.float32)
        self._mask = np.zeros((self.height, self.width), dtype=np.uint8)

    def _bounds(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        pad: int,
    ) -> tuple[slice, slice] | None:
        left = max(min(x0, x1) - pad, 0)
        right = min(max(x0, x1) + pad + 1, self.width)
        top = max(min(y0, y1) - pad, 0)
        bottom = min(max(y0, y1) + pad + 1, self.height)
        if left >= right or top >= bottom:
            return None
        return slice(top, bottom), slice(left, right)

    def _stamp(
        self,
        window: tuple[slice, slice],
        shape: str,
        points: tuple[int, int] | tuple[int, int, int, int],
        color_rgb: np.ndarray,
        alpha: float,
        thickness: int,
    ) -> None:
        rows, columns = window
        self._mask[rows, columns] = 0
        local = (
            (points[0] - columns.start, points[1] - rows.start)
            if shape == "dot"
            else (
                points[0] - columns.start,
                points[1] - rows.start,
                points[2] - columns.start,
                points[3] - rows.start,
            )
        )
        if shape == "dot":
            cv2.circle(self._mask[rows, columns], local, thickness, 1, thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.line(self._mask[rows, columns], local[:2], local[2:], 1, thickness=thickness, lineType=cv2.LINE_AA)
        hit = self._mask[rows, columns].astype(bool)
        if not hit.any():
            return
        self._color[rows, columns][hit] = np.asarray(color_rgb, dtype=np.uint8)
        self._alpha[rows, columns][hit] = float(alpha)

    def dot(self, point_uv: np.ndarray, color_rgb: np.ndarray, *, radius: int, alpha: float) -> None:
        x = int(round(float(point_uv[0])))
        y = int(round(float(point_uv[1])))
        if x < -radius or y < -radius or x > self.width + radius or y > self.height + radius:
            return
        window = self._bounds(x, y, x, y, radius + 1)
        if window is None:
            return
        self._stamp(window, "dot", (x, y), color_rgb, alpha, int(radius))

    def segment(
        self,
        start_uv: np.ndarray,
        end_uv: np.ndarray,
        color_rgb: np.ndarray,
        *,
        width: int,
        alpha: float,
    ) -> None:
        x0 = int(round(float(start_uv[0])))
        y0 = int(round(float(start_uv[1])))
        x1 = int(round(float(end_uv[0])))
        y1 = int(round(float(end_uv[1])))
        if (x0 < 0 and x1 < 0) or (y0 < 0 and y1 < 0):
            return
        if (x0 >= self.width and x1 >= self.width) or (y0 >= self.height and y1 >= self.height):
            return
        window = self._bounds(x0, y0, x1, y1, int(width) + 1)
        if window is None:
            return
        self._stamp(window, "segment", (x0, y0, x1, y1), color_rgb, alpha, int(width))

    def composite(self, canvas: np.ndarray) -> None:
        if not self._alpha.any():
            return
        weight = self._alpha[:, :, None]
        blended = canvas.astype(np.float32) * (1.0 - weight) + self._color.astype(np.float32) * weight
        np.copyto(canvas, np.rint(blended).astype(np.uint8))
