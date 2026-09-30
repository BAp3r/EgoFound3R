import heapq
import math
import numpy as np
import torch
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
from .inference_root_z import _weighted_median
from .inference_hand_depth_scale import spatial_log_statistics
from .models.camera_temporal_refiner import (
    DeterministicTemporalInterpolator,
    build_temporal_interpolation_query_map,
    clip_local_metric_w2c,
)
from .multirate_contracts import build_multirate_frame_map


def compact_depth_sample(depth, confidence, frame_indices):
    depth = depth.detach().float().cpu().numpy()
    confidence = confidence.detach().float().cpu().numpy()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    frames = frame_indices.detach().cpu().numpy()
    if depth.ndim != 3 or confidence.shape != depth.shape or frames.shape != depth.shape[:1]:
        raise ValueError("raw Depth/confidence/source G frames must align")
    if np.any(np.diff(frames) <= 0):
        raise ValueError("source G frames must be unique and increasing")
    step = max(1, math.ceil(max(depth.shape[1:])/64))
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence > 0)
    safe = np.where(valid, depth, 0.)
    edge = maximum_filter(safe, size=(1,3,3))-minimum_filter(safe, size=(1,3,3))
    valid &= minimum_filter(valid.astype(np.uint8), size=(1,3,3)) > 0
    valid &= edge < .08 * safe
    return {"frames": frames.copy(), "depth": safe[:,::step,::step].copy(),
            "confidence": np.where(valid,confidence,0.)[:,::step,::step].copy(),
            "valid": valid[:,::step,::step].copy(), "image_size": depth.shape[1:], "step": step}


def estimate_overlap_depth_ratio(left, right):
    report = {"valid": False, "reason": "no_common_G", "frame_count": 0,
              "log_ratio": 0., "quality": 0., "frames": []}
    if left["image_size"] != right["image_size"] or left["step"] != right["step"]:
        report["reason"] = "different_pixel_lattice"
        return report
    common, li, ri = np.intersect1d(left["frames"],right["frames"],return_indices=True)
    logs, qualities, uncertainties, spatial_qualities = [], [], [], []
    for frame,a,b in zip(common,li,ri):
        d0,d1 = left["depth"][a],right["depth"][b]
        c0,c1 = left["confidence"][a],right["confidence"][b]
        mask = left["valid"][a] & right["valid"][b]
        if mask.sum() < 64:
            continue
        mask &= (c0 >= np.median(c0[mask])) & (c1 >= np.median(c1[mask]))
        y,x = np.where(mask)
        cells = len(np.unique(np.stack((y*4//d0.shape[0],x*4//d0.shape[1]),axis=-1),axis=0))
        if len(y) < 64 or cells < 4:
            continue
        ratios = np.log(d0[mask].astype(np.float64))-np.log(d1[mask].astype(np.float64))
        weights = np.minimum(c0[mask]/np.median(c0[mask]),c1[mask]/np.median(c1[mask])).clip(.25,1.)
        regions = np.stack((y*4//d0.shape[0],x*4//d0.shape[1]),axis=-1)
        cell_median, cell_mad, spread = spatial_log_statistics(ratios, weights, regions)
        median = _weighted_median(ratios,weights)
        mad = _weighted_median(abs(ratios-median),weights)
        row = {"source_frame": int(frame), "pixels": len(y), "cells": cells,
               "log_ratio": median, "log_mad": mad, "cell_log_mad": cell_mad,
               "cell_log_spread": spread, "cell_log_disagreement": abs(median-cell_median)}
        report["frames"].append(row)
        quality = min(1.,len(y)/256)*math.exp(-(mad/.1)**2)
        if mad <= .1 and abs(median-cell_median)<=math.log(1.1):
            logs.append(median)
            qualities.append(quality)
            uncertainties.append(max(1.4826*mad, spread))
            spatial_qualities.append(math.exp(-(spread/.2)**2))
    report["frame_count"] = len(logs)
    if not logs:
        report["reason"] = "insufficient_consistent_pixels" if len(common) else "no_common_G"
        return report
    logs,weights = np.asarray(logs),np.asarray(qualities)
    membership=abs(logs[:,None]-logs[None,:]) <= .1
    keep=membership[int(np.argmax(membership@weights))]
    if weights[keep].sum() < .6*weights.sum():
        report["reason"]="inconsistent_overlap_frames"
        return report
    logs,weights=logs[keep],weights[keep]
    uncertainties=np.asarray(uncertainties)[keep]
    report["frame_count"]=len(logs)
    median = _weighted_median(logs,weights)
    mad = _weighted_median(abs(logs-median),weights)
    quality = min(1.,len(logs)/2)*math.exp(-(mad/.1)**2)*float(weights.mean())
    quality *= float(np.mean(np.asarray(spatial_qualities)[keep]))
    report.update(log_ratio=median, ratio=math.exp(median), log_mad=mad, quality=quality,
                  log_uncertainty=max(.03,1.4826*mad,float(np.median(uncertainties)),.1*(1-quality)))
    if mad > .1 or quality < .25:
        report["reason"] = "inconsistent_overlap_frames"
        return report
    report.update(valid=True,reason="accepted")
    return report


def _components(count, edges):
    adjacency = [[] for _ in range(count)]
    for edge in edges:
        if edge["valid"]:
            adjacency[edge["left"]].append(edge["right"])
            adjacency[edge["right"]].append(edge["left"])
    remaining = set(range(count))
    result = []
    while remaining:
        stack=[min(remaining)]; remaining.remove(stack[0]); group=[]
        while stack:
            node=stack.pop();group.append(node)
            for neighbor in adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor);stack.append(neighbor)
        result.append(sorted(group))
    return result


def _solve_relative(group, edges, fixed):
    free=[node for node in group if node not in fixed]
    result=dict(fixed)
    if not free:
        return result
    columns={node:i for i,node in enumerate(free)}
    group_set=set(group)
    rows=[];cols=[];data=[];target=[];weight=[]
    for edge in edges:
        a,b=edge["left"],edge["right"]
        if not edge["valid"] or a not in group_set or b not in group_set or (a in fixed and b in fixed):
            continue
        rhs=edge["log_ratio"]
        row=len(target)
        for node,sign in ((a,-1.),(b,1.)):
            if node in fixed: rhs -= sign*fixed[node]
            else: rows.append(row);cols.append(columns[node]);data.append(sign)
        target.append(rhs);weight.append(edge["quality"])
    if not target:
        raise ValueError("connected scale component has no constraints for free nodes")
    matrix=coo_matrix((data,(rows,cols)),shape=(len(target),len(free))).tocsr()
    target=np.asarray(target);base=np.asarray(weight);robust=np.ones_like(base)
    for _ in range(3):
        root_weight=np.sqrt(base*robust)
        solved=lsmr(matrix.multiply(root_weight[:,None]),target*root_weight,atol=1e-10,btol=1e-10,
                    maxiter=max(100,5*len(free)))
        solution=solved[0]
        if solved[1] not in (0,1,2,4,5) or not np.isfinite(solution).all():
            raise RuntimeError(f"metric scale solver failed: stop_code={solved[1]}")
        residual=abs(matrix@solution-target)
        robust=np.minimum(1.,.1/np.maximum(residual,1e-12))
    result.update({node:float(solution[i]) for node,i in columns.items()})
    return result


def _attach_propagation_reliability(nodes, edges, local_scales):
    adjacency=[[] for _ in nodes]
    for e in edges:
        if not e["valid"]: continue
        risk=max(.03,e.get("log_uncertainty",0.),1.4826*e.get("log_mad",0.),
                 abs(e.get("resolved_log_residual") or 0.),.1*(1-e["quality"]))
        e["log_uncertainty"]=risk
        adjacency[e["left"]].append((e["right"],risk))
        adjacency[e["right"]].append((e["left"],risk))
    queue=[]; best={}
    for i,(node,local) in enumerate(zip(nodes,local_scales)):
        node.update(reference_chunk=None,propagation_hops=None,log_uncertainty=None,confidence=0.,reliable=False)
        if not node["valid"]: continue
        strong=local.get("strength")=="strong"
        hand_component=node["source"]=="weak_hand_component" and local.get("hand_factor") is not None
        head_component=node["source"]=="metric_head_component" and local.get("prior_valid",False)
        if not (strong or hand_component or head_component): continue
        reference=local["hand_factor"] if strong else local["factor"] if hand_component else local["prior_factor"]
        disagreement=abs(math.log(node["factor"])-math.log(reference))
        if disagreement>math.log(1.25): continue
        quality=float(local.get("quality",0.)) if not head_component else .1
        risk=max(.03 if strong else .1,local.get("log_uncertainty") or .15*(1-quality))
        if head_component: risk=max(risk,math.log(2.))
        risk+=disagreement
        best[i]=(risk,0,i)
        heapq.heappush(queue,(risk,0,i,i,quality,strong))
    while queue:
        risk,hops,reference,i,quality,strong=heapq.heappop(queue)
        if best[i]!=(risk,hops,reference): continue
        nodes[i].update(reference_chunk=reference,propagation_hops=hops,log_uncertainty=risk,
                        confidence=quality*math.exp(-(risk/math.log(1.25))**2),
                        reliable=bool(strong and risk<=math.log(1.25)))
        for neighbor,edge_risk in adjacency[i]:
            candidate=(risk+edge_risk,hops+1,reference)
            if candidate<best.get(neighbor,(math.inf,0,0)):
                best[neighbor]=candidate
                heapq.heappush(queue,(*candidate,neighbor,quality,strong))


def resolve_scale_graph(local_scales, edges):
    edges=[dict(edge) for edge in edges]
    count=len(local_scales)

    def solve():
        nodes=[{"factor":1.,"valid":False,"source":"invalid","component":-1} for _ in range(count)]
        for component,group in enumerate(_components(count,edges)):
            strong={i:math.log(local_scales[i]["hand_factor"]) for i in group if local_scales[i].get("strength")=="strong"}
            if strong:
                logs=_solve_relative(group,edges,strong)
                for i in group:
                    nodes[i].update(factor=math.exp(logs[i]),valid=True,component=component,
                                    source="local_strong" if i in strong else "overlap_strong")
                continue
            weak=[i for i in group if local_scales[i].get("hand_factor") is not None]
            candidates=weak or [i for i in group if local_scales[i].get("prior_valid",False)]
            if not candidates:
                continue
            relative=_solve_relative(group,edges,{group[0]:0.})
            absolute=np.array([math.log(local_scales[i]["factor"] if weak else local_scales[i]["prior_factor"])-relative[i] for i in candidates])
            weights=np.array([max(local_scales[i].get("quality",0.),.01) if weak else 1. for i in candidates])
            gauge=_weighted_median(absolute,weights)
            for i in group:
                nodes[i].update(factor=math.exp(relative[i]+gauge),valid=True,component=component,
                                source="weak_hand_component" if weak else "metric_head_component")
        return nodes

    for _ in range(sum(bool(e["valid"]) for e in edges)+1):
        nodes=solve(); bad=[]
        for index,edge in enumerate(edges):
            if not edge["valid"]: continue
            a,b=nodes[edge["left"]],nodes[edge["right"]]
            if not (a["valid"] and b["valid"]): continue
            residual=math.log(b["factor"])-math.log(a["factor"])-edge["log_ratio"]
            edge["resolved_log_residual"]=residual
            if abs(residual)>math.log(1.25):bad.append((abs(residual),index))
        if not bad: break
        index=max(bad)[1]
        edges[index].update(valid=False,reason="conflicting_metric_anchors_or_cycle",
                            rejection_log_residual=edges[index]["resolved_log_residual"])
    for edge in edges:
        a,b=nodes[edge["left"]],nodes[edge["right"]]
        edge["resolved_log_residual"]=(math.log(b["factor"])-math.log(a["factor"])-edge["log_ratio"]
                                        if a["valid"] and b["valid"] else None)
    _attach_propagation_reliability(nodes,edges,local_scales)
    return {"schema_version":"egofound3r.overlap_depth_scale.v1","nodes":nodes,"edges":edges}


def rebuild_chunk_scene(outputs, raw_scene, raw_camera, metadata, resolved):
    depth=raw_scene["depth_raw_global"].float()
    if depth.ndim==4 and depth.shape[-1]==1: depth=depth[...,0]
    K=raw_scene["intrinsics_global"].float()
    support=torch.isfinite(raw_camera).all(dim=(-1,-2))
    support &= torch.linalg.det(raw_camera.float()).abs()>1e-8
    support &= torch.isfinite(K).all(dim=(-1,-2)) & (K[:,0,0]>0) & (K[:,1,1]>0)
    support &= (torch.isfinite(depth)&(depth>0)).flatten(1).any(1)
    support &= bool(resolved["valid"])
    valid=bool(support.any())
    factor=float(resolved["factor"]) if valid else 1.
    if not math.isfinite(factor) or factor<=0:
        raise ValueError("resolved metric factor must be positive finite")
    high=torch.tensor(metadata["high_indices_local"],dtype=torch.long)[None]
    present=torch.ones_like(high,dtype=torch.bool)
    mapping=build_multirate_frame_map(high_frame_present=present,high_temporal_pair_valid=present[:,1:],
        global_stride=int(metadata["global_stride"]),global_anchor_phase=int(metadata["global_anchor_phase_local"]))
    query=build_temporal_interpolation_query_map(mapping,interpolation_factor=1,interval_steps=torch.diff(high,dim=1))
    indices=query.high_anchor_indices.gather(1,mapping.global_anchor_indices)[0]
    if indices.tolist()!=metadata["global_indices_local"]:
        raise ValueError("scene replay G indices differ from the original forward")
    metric_camera=clip_local_metric_w2c(raw_camera.float()[None],support[None],translation_scale=torch.tensor([factor]))
    camera,camera_valid,_,exact=DeterministicTemporalInterpolator()._camera_base(metric_camera,mapping,query,support[None])
    result=dict(outputs)
    for key in ("camera_pose_base_full","camera_pose_interpolated_full","camera_pose_refined_full"):
        result[key]=camera[0]
    for key in ("camera_refined_valid_full","camera_interpolation_valid_full"):
        result[key]=camera_valid[0]
    result["camera_anchor_exact_mask_full"]=exact[0]
    metric_depth=depth*factor
    scene={"depth_global":metric_depth,"depth_metric_global":metric_depth,
           "depth_conf_global":raw_scene["depth_conf_global"],"intrinsics_global":K,
           "camera_pose_global":metric_camera[0],"camera_pose_refined_full":camera[0],
           "frame_valid":support}
    metadata.update(raw_to_meter_fused=factor,metric_scale_valid=valid,scale_source=resolved["source"])
    for key in ("confidence","reliable","log_uncertainty","reference_chunk","propagation_hops"):
        if key in resolved:
            metadata[f"metric_scale_{key}"]=resolved[key]
    if not valid:
        metadata.update(metric_scale_confidence=0.,metric_scale_reliable=False)
    return result,scene
