from __future__ import annotations


import torch
from egohandmetric_prompt.models.camera_temporal_refiner import w2c_to_c2w


def masked_average_distance(
    points: torch.Tensor,
    mask: torch.Tensor,
    *,
    fallback: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if points.shape[:-1] != mask.shape:
        raise ValueError("points and mask shapes must match except for the xyz dimension")
    if points.shape[-1] != 3:
        raise ValueError("points must end with xyz dimension")
    batch_size = points.shape[0]
    flat_points = points.reshape(batch_size, -1, 3)
    flat_mask = mask.reshape(batch_size, -1).to(dtype=torch.bool)
    finite = torch.isfinite(flat_points).all(dim=-1)
    valid = flat_mask & finite
    distances = torch.linalg.norm(torch.nan_to_num(flat_points), dim=-1)
    counts = valid.sum(dim=1)
    summed = torch.where(valid, distances, torch.zeros_like(distances)).sum(dim=1)
    scale = summed / counts.clamp_min(1).to(dtype=points.dtype)
    fallback_tensor = torch.full_like(scale, float(fallback))
    return torch.where(counts > 0, scale.clamp_min(eps), fallback_tensor)


def unproject_depth_to_camera_points(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if depth.ndim == 5:
        if depth.shape[-1] != 1:
            raise ValueError("depth with 5 dims must end with a singleton channel")
        depth = depth.squeeze(-1)
    if depth.ndim != 4:
        raise ValueError("depth must have shape [B, T, H, W]")
    if intrinsics.shape[:2] != depth.shape[:2] or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [B, T, 3, 3]")
    geometry_dtype = torch.float64 if depth.dtype == torch.float64 else torch.float32
    depth = depth.to(dtype=geometry_dtype)
    intrinsics = intrinsics.to(dtype=geometry_dtype)
    height, width = depth.shape[-2:]
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    fx = intrinsics[..., 0, 0].clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    fy = intrinsics[..., 1, 1].clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1).unsqueeze(-1)
    z = depth
    x = (xs.view(1, 1, height, width) - cx) / fx * z
    y = (ys.view(1, 1, height, width) - cy) / fy * z
    return torch.stack([x, y, z], dim=-1)


def transform_camera_points_to_first_frame(points: torch.Tensor, camera_pose: torch.Tensor) -> torch.Tensor:
    if points.ndim != 5 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B, T, N, 3] or [B, T, H, W, 3]")
    if camera_pose.shape[:2] != points.shape[:2] or camera_pose.shape[-2:] != (4, 4):
        raise ValueError("camera_pose must have shape [B, T, 4, 4]")
    point_shape = points.shape
    geometry_dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
    flat_points = points.to(dtype=geometry_dtype).reshape(points.shape[0], points.shape[1], -1, 3)
    with torch.autocast(device_type=points.device.type, enabled=False):
        camera_pose_f = camera_pose.to(dtype=geometry_dtype)
        frame_to_first = torch.matmul(camera_pose_f[:, :1], w2c_to_c2w(camera_pose_f))
        rotation = frame_to_first[..., :3, :3]
        translation = frame_to_first[..., :3, 3]
        transformed = torch.einsum("btij,btnj->btni", rotation, flat_points) + translation.unsqueeze(-2)
    return transformed.reshape(point_shape)


def scene_average_distance_from_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pose: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    fallback: float = 1.0,
) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth_for_mask = depth.squeeze(-1)
    else:
        depth_for_mask = depth
    points = unproject_depth_to_camera_points(depth, intrinsics)
    points_first = transform_camera_points_to_first_frame(points, camera_pose)
    valid = torch.isfinite(depth_for_mask) & (depth_for_mask > 0)
    valid &= torch.isfinite(intrinsics).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    valid &= torch.isfinite(camera_pose).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    pose_det = torch.linalg.det(camera_pose.float()).abs().to(device=depth.device)
    valid &= (pose_det > 1e-6).unsqueeze(-1).unsqueeze(-1)
    if valid_mask is not None:
        valid = valid & valid_mask.to(device=depth.device, dtype=torch.bool)
    return masked_average_distance(
        points_first.reshape(points_first.shape[0], -1, 3),
        valid.reshape(valid.shape[0], -1),
        fallback=fallback,
    )
