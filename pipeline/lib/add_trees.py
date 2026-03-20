import numpy as np
from typing import Any, Dict, List, Tuple

from pipeline.lib.geom_utils import dbscan_obb, compute_obb


def _compute_tree_params(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    if points.shape[0] == 0:
        return np.array([0.0, 0.0, 0.0]), 0.0, 0.0
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    z_ground = float(np.percentile(zs, 5.0))
    height = max(0.5, float(np.percentile(zs, 95.0)) - z_ground)
    dx, dy = xs - cx, ys - cy
    crown_radius = max(0.5, float(np.percentile(np.sqrt(dx * dx + dy * dy), 90.0)))
    return np.array([cx, cy, z_ground]), height, crown_radius


def process_graph(
    graph_data: Dict[str, Any],
    geom_data: Dict[str, Any],
    tree_group_tol: float,
    laz_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> int:
    xyz, sem, ins = laz_data
    trees: List[Dict[str, Any]] = []

    if len(xyz) > 0:
        tree_mask = sem == 5
        xyz_t, sem_t, ins_t = xyz[tree_mask], sem[tree_mask], ins[tree_mask]
        keep = ins_t != -1
        xyz_t, sem_t, ins_t = xyz_t[keep], sem_t[keep], ins_t[keep]

        if len(xyz_t) > 0:
            for ins_id in np.unique(ins_t):
                mask = ins_t == ins_id
                pts = xyz_t[mask]
                if len(pts) == 0:
                    continue
                center, height, crown_radius = _compute_tree_params(pts)
                tid = len(trees)
                x, y, z0 = center.tolist()
                trees.append({
                    "id": tid, "sem_class": int(sem_t[mask][0]),
                    "X": x, "Y": y, "Z": z0,
                    "height": height, "crown_radius": crown_radius,
                    "min": [x - crown_radius, y - crown_radius, z0],
                    "max": [x + crown_radius, y + crown_radius, z0 + height],
                    "obb": compute_obb(pts),
                })

    clusters: List[List[int]] = []
    if trees and tree_group_tol > 0:
        obbs = [t.get("obb") for t in trees]
        mins = [np.asarray(t["min"]) for t in trees]
        maxs = [np.asarray(t["max"]) for t in trees]
        ids = [t["id"] for t in trees]
        clusters = dbscan_obb(obbs, mins, maxs, ids, eps=tree_group_tol, min_samples=3)

    id_to_group: Dict[int, int] = {}
    tree_groups: List[Dict[str, Any]] = []
    for gid, cluster in enumerate(clusters, start=1):
        tree_groups.append({"id": gid, "members": sorted(cluster)})
        for tid in cluster:
            id_to_group[tid] = gid

    graph_trees, geom_trees = [], {}
    for t in trees:
        tid_str = str(t["id"])
        entry = {"id": t["id"], "sem_class": t["sem_class"]}
        if t["id"] in id_to_group:
            entry["group_id"] = id_to_group[t["id"]]
        graph_trees.append(entry)
        geom_entry = {k: t[k] for k in ["X", "Y", "Z", "height", "crown_radius", "min", "max"]}
        geom_entry["obb"] = t.get("obb")
        geom_trees[tid_str] = geom_entry

    graph_data["trees"] = graph_trees
    graph_data["tree_groups"] = tree_groups
    geom_data["trees"] = geom_trees

    return len(trees)
