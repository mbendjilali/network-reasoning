import json
import os
import glob
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

GROUP_TYPES = ("trees", "buildings", "vehicles", "poles", "conductors")

RELATION_MEMBER_TYPES = ("tree", "building", "vehicle")

ADJACENT_DIST_MAX = {
    "building": 1.0,
    "vehicle": 0.25,
    "tree": 0.25,
}

NEAR_DIST_MAX = {
    "building": 15.0,
    "vehicle": 8.0,
    "tree": 20.0,
}


def _groups_key(object_type: str) -> str:
    return object_type.rstrip("s") + "_groups" if object_type.endswith("s") else object_type + "_groups"


def _migrate_legacy_grouped_with(graph_data: Dict[str, Any]) -> None:
    """One-time migration of legacy 'grouped_with' lists into *_groups + group_id."""
    for object_type in GROUP_TYPES:
        objects = graph_data.get(object_type, [])
        if not objects:
            continue
        if any(obj.get("group_id") is not None for obj in objects):
            for obj in objects:
                obj.pop("grouped_with", None)
            continue
        if not any(isinstance(obj.get("grouped_with"), list) and obj["grouped_with"] for obj in objects):
            for obj in objects:
                obj.pop("grouped_with", None)
            continue

        objects_map = {obj["id"]: obj for obj in objects}
        visited: Set[int] = set()
        components: List[Set[int]] = []

        def bfs(start_id: int) -> Set[int]:
            comp: Set[int] = set()
            stack = [start_id]
            while stack:
                oid = stack.pop()
                if oid in comp:
                    continue
                comp.add(oid)
                obj = objects_map.get(oid)
                if not obj:
                    continue
                for mid in obj.get("grouped_with") or []:
                    if mid not in comp:
                        stack.append(mid)
            return comp

        for oid, obj in objects_map.items():
            if oid in visited:
                continue
            gw = obj.get("grouped_with") or []
            if not gw:
                visited.add(oid)
                continue
            comp = bfs(oid)
            visited.update(comp)
            components.append(comp)

        groups_key = _groups_key(object_type)
        graph_data[groups_key] = []
        groups_list = graph_data[groups_key]
        next_gid = 1
        for comp in components:
            gid = next_gid
            next_gid += 1
            members = sorted(comp)
            groups_list.append({"id": gid, "members": members})
            for oid in members:
                objects_map[oid]["group_id"] = gid

        for obj in objects:
            obj.pop("grouped_with", None)


class GraphManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.current_tile_id: Optional[str] = None
        self.graph_data: Dict[str, Any] = {}
        self.geom_data: Dict[str, Any] = {}
        # Intra-group relation cache is kept in graph_data["group_relations"].

    def _get_graph_path(self, tile_id: str) -> str:
        return os.path.join(self.data_dir, "graph", f"graph_{tile_id}.json")

    def _get_geom_path(self, tile_id: str) -> str:
        return os.path.join(self.data_dir, "geometry", f"geom_{tile_id}.json")

    def load_tile(self, tile_id: str) -> Dict[str, Any]:
        """Loads graph and geometry data for a tile."""
        self.current_tile_id = tile_id

        graph_path = self._get_graph_path(tile_id)
        # Geometry is shared between original and edited versions:
        # try geom_{tile_id}.json first, then fall back to the base id (before any _edit_ suffix).
        geom_path = self._get_geom_path(tile_id)
        if not os.path.exists(geom_path):
            base_id = tile_id.split("_edit_")[0]
            geom_path_base = self._get_geom_path(base_id)
            if os.path.exists(geom_path_base):
                geom_path = geom_path_base

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

        with open(graph_path, "r") as f:
            self.graph_data = json.load(f)

        _migrate_legacy_grouped_with(self.graph_data)

        if os.path.exists(geom_path):
            with open(geom_path, "r") as f:
                self.geom_data = json.load(f)
        else:
            self.geom_data = {}

        return {"graph": self.graph_data, "geometry": self.geom_data}

    def save_tile(self, tile_id: str) -> str:
        """
        Save the current in-memory graph to a new edited file.

        Original graph_{tile_id}.json is left untouched.
        New files are named: graph_{tile_id}_edit_{count}.json.
        """
        if self.current_tile_id != tile_id:
            raise ValueError(
                f"Current loaded tile ({self.current_tile_id}) does not match save target ({tile_id})"
            )

        original_path = self._get_graph_path(tile_id)
        base_dir, base_name = os.path.split(original_path)
        name, ext = os.path.splitext(base_name)  # e.g. "graph_5145_54470", ".json"

        prefix = f"{name}_edit_"
        pattern = os.path.join(base_dir, f"{prefix}*.json")
        existing = glob.glob(pattern)
        next_index = len(existing) + 1

        new_name = f"{prefix}{next_index}{ext}"
        new_path = os.path.join(base_dir, new_name)

        os.makedirs(base_dir, exist_ok=True)
        save_data = {k: v for k, v in self.graph_data.items() if k != "macro_instances"}
        with open(new_path, "w") as f:
            json.dump(save_data, f, indent=2)

        return new_path

    # --- Intra-group relation helpers (adjacent / near) ---

    def _ensure_group_relations(self) -> List[Dict[str, Any]]:
        """
        Ensure there is a generic container for per-pair group relations.
        Each relation has: id, member_type, group_id, a_id, b_id, class.
        """
        rels = self.graph_data.get("group_relations")
        if rels is None:
            rels = []
            self.graph_data["group_relations"] = rels
        return rels

    def _next_group_relation_id(self) -> int:
        rels = self.graph_data.get("group_relations", [])
        if not rels:
            return 0
        return max(int(r.get("id", 0)) for r in rels) + 1

    def list_group_relations(
        self,
        member_type: Optional[str] = None,
        group_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List stored per-pair group relations, optionally filtered by member_type and/or group_id.
        """
        rels = self.graph_data.get("group_relations", [])
        out: List[Dict[str, Any]] = []
        for r in rels:
            if member_type is not None and r.get("member_type") != member_type:
                continue
            if group_id is not None and int(r.get("group_id", -1)) != int(group_id):
                continue
            out.append(r)
        return out

    def upsert_group_relation(
        self,
        member_type: str,
        group_id: int,
        a_id: int,
        b_id: int,
        rel_class: str,
    ) -> Dict[str, Any]:
        """
        Create or update a relation between two members inside a group.
        Endpoints are stored in sorted order so (a,b) and (b,a) map to the same entry.
        """
        if member_type not in RELATION_MEMBER_TYPES:
            raise ValueError(f"Relations not supported for member_type={member_type}")

        rels = self._ensure_group_relations()
        a_ord, b_ord = sorted((int(a_id), int(b_id)))
        gid = int(group_id)

        for r in rels:
            if (
                r.get("member_type") == member_type
                and int(r.get("group_id", -1)) == gid
                and int(r.get("a_id", -1)) == a_ord
                and int(r.get("b_id", -1)) == b_ord
            ):
                r["class"] = rel_class
                return r

        rid = self._next_group_relation_id()
        rel = {
            "id": rid,
            "member_type": member_type,
            "group_id": gid,
            "a_id": a_ord,
            "b_id": b_ord,
            "class": rel_class,
        }
        rels.append(rel)
        return rel

    def update_group_relation(self, rel_id: int, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a single relation by id (used by API editing).
        """
        rels = self._ensure_group_relations()
        for r in rels:
            if int(r.get("id", -1)) == int(rel_id):
                r.update(fields)
                return r
        raise ValueError(f"Group relation {rel_id} not found")

    def delete_group_relations_for_group(
        self,
        member_type: str,
        group_id: int,
        member_ids: Optional[Set[int]] = None,
    ) -> None:
        """
        Remove relations that belong to a group and (optionally) involve any of member_ids.
        Used by ungroup/split operations.
        """
        if "group_relations" not in self.graph_data:
            return
        gid = int(group_id)
        members = {int(m) for m in (member_ids or set())}
        new_rels: List[Dict[str, Any]] = []
        for r in self.graph_data.get("group_relations", []):
            if r.get("member_type") != member_type:
                new_rels.append(r)
                continue
            if int(r.get("group_id", -1)) != gid:
                new_rels.append(r)
                continue
            if not members:
                # delete all relations for this group
                continue
            a_id = int(r.get("a_id", -1))
            b_id = int(r.get("b_id", -1))
            if a_id in members or b_id in members:
                # drop this relation
                continue
            new_rels.append(r)
        self.graph_data["group_relations"] = new_rels

    # --- Geometry helpers for relation classification ---

    def _member_type_for_object_type(self, object_type: str) -> Optional[str]:
        """
        Map plural object_type used in groups (e.g. 'buildings') to member_type
        used in relations ('building', 'vehicle', 'tree').
        """
        mapping = {
            "buildings": "building",
            "vehicles": "vehicle",
            "trees": "tree",
        }
        return mapping.get(object_type)

    def _centroid_for_member(self, member_type: str, member_id: int, obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """
        Approximate XY centroid for a member object using geom_data.
        """
        try:
            if member_type == "building":
                geom = self.geom_data.get("buildings", {}).get(str(member_id), {})
            elif member_type == "vehicle":
                geom = self.geom_data.get("vehicles", {}).get(str(member_id), {})
            elif member_type == "tree":
                geom = self.geom_data.get("trees", {}).get(str(member_id), {})
            else:
                geom = {}

            if member_type in ("building", "vehicle"):
                obb = geom.get("obb") or obj.get("obb")
                if obb and "center" in obb:
                    cx, cy, _ = obb["center"]
                    return float(cx), float(cy)
                # Fallback to AABB if present
                min_v = geom.get("min")
                max_v = geom.get("max")
                if isinstance(min_v, list) and isinstance(max_v, list) and len(min_v) >= 2 and len(max_v) >= 2:
                    cx = (float(min_v[0]) + float(max_v[0])) * 0.5
                    cy = (float(min_v[1]) + float(max_v[1])) * 0.5
                    return cx, cy
                return None
            elif member_type == "tree":
                x = geom.get("X")
                y = geom.get("Y")
                if x is None or y is None:
                    return None
                return float(x), float(y)
        except Exception:
            return None
        return None

    def _object_shape_gap(self, member_type: str, a_id: int, b_id: int) -> Optional[float]:
        """
        Approximate 2D gap between two object footprints.
        Uses OBB footprints (Polygon distance) if available, otherwise falls back to AABBs.
        """
        coll_name = {
            "building": "buildings",
            "vehicle": "vehicles",
            "tree": "trees",
        }.get(member_type)
        if not coll_name: return None

        geom_a = self.geom_data.get(coll_name, {}).get(str(a_id), {})
        geom_b = self.geom_data.get(coll_name, {}).get(str(b_id), {})

        # Try OBB first
        obb_a = geom_a.get("obb")
        obb_b = geom_b.get("obb")
        if obb_a and obb_b and "footprint" in obb_a and "footprint" in obb_b:
            try:
                poly_a = Polygon(obb_a["footprint"])
                poly_b = Polygon(obb_b["footprint"])
                return float(poly_a.distance(poly_b))
            except Exception:
                pass

        # Fallback to AABB
        min_a = geom_a.get("min")
        max_a = geom_a.get("max")
        min_b = geom_b.get("min")
        max_b = geom_b.get("max")
        if not (
            isinstance(min_a, list)
            and isinstance(max_a, list)
            and isinstance(min_b, list)
            and isinstance(max_b, list)
            and len(min_a) >= 2
            and len(max_a) >= 2
            and len(min_b) >= 2
            and len(max_b) >= 2
        ):
            return None

        ax_min, ay_min = float(min_a[0]), float(min_a[1])
        ax_max, ay_max = float(max_a[0]), float(max_a[1])
        bx_min, by_min = float(min_b[0]), float(min_b[1])
        bx_max, by_max = float(max_b[0]), float(max_b[1])

        # Gap on X
        if ax_max < bx_min:
            gap_x = bx_min - ax_max
        elif bx_max < ax_min:
            gap_x = ax_min - bx_max
        else:
            gap_x = 0.0

        # Gap on Y
        if ay_max < by_min:
            gap_y = by_min - ay_max
        elif by_max < ay_min:
            gap_y = ay_min - by_max
        else:
            gap_y = 0.0

        return max(gap_x, gap_y)

    def _delaunay_edges(self, centroids: Dict[int, Tuple[float, float]]) -> Set[Tuple[int, int]]:
        """Return unique undirected edges from a 2D Delaunay triangulation of centroids."""
        ids = list(centroids.keys())
        if len(ids) < 3:
            if len(ids) == 2:
                return {(min(ids), max(ids))}
            return set()
        pts = np.array([centroids[i] for i in ids])
        tri = Delaunay(pts)
        edges: Set[Tuple[int, int]] = set()
        for simplex in tri.simplices:
            for k in range(3):
                a_idx, b_idx = int(simplex[k]), int(simplex[(k + 1) % 3])
                a, b = ids[a_idx], ids[b_idx]
                edges.add((min(a, b), max(a, b)))
        return edges

    def _recompute_relations_for_group(
        self,
        member_type: str,
        group_id: int,
        member_ids: List[int],
    ) -> None:
        """Compute adjacent/near relations for all pairs in a cluster using Delaunay."""
        if member_type not in RELATION_MEMBER_TYPES:
            return
        coll_name = {"tree": "trees", "building": "buildings", "vehicle": "vehicles"}[member_type]
        objects = {obj["id"]: obj for obj in self.graph_data.get(coll_name, [])}
        if len(member_ids) < 2:
            return

        gid = int(group_id)
        adj_max = float(ADJACENT_DIST_MAX.get(member_type, 0.0))
        near_max = float(NEAR_DIST_MAX.get(member_type, 0.0))

        centroids: Dict[int, Tuple[float, float]] = {}
        for mid in member_ids:
            obj = objects.get(int(mid))
            if not obj:
                continue
            c = self._centroid_for_member(member_type, int(mid), obj)
            if c is not None:
                centroids[int(mid)] = c

        delaunay_edges = self._delaunay_edges(centroids)

        for a, b in delaunay_edges:
            gap = self._object_shape_gap(member_type, a, b)
            if gap is None:
                continue
            if gap <= adj_max:
                self.upsert_group_relation(member_type, gid, a, b, "adjacent")
            elif gap <= near_max:
                self.upsert_group_relation(member_type, gid, a, b, "near")

        for i, aid in enumerate(member_ids):
            for bid in member_ids[i + 1:]:
                key = (min(aid, bid), max(aid, bid))
                if key in delaunay_edges:
                    continue
                gap = self._object_shape_gap(member_type, aid, bid)
                if gap is not None and gap <= adj_max:
                    self.upsert_group_relation(member_type, gid, aid, bid, "adjacent")

    def _recompute_relations_for_pairs(
        self,
        member_type: str,
        group_id: int,
        all_member_ids: List[int],
        pairs: Set[Tuple[int, int]],
    ) -> None:
        """Compute relations only for a specific set of (a, b) pairs (used by scoped wedge)."""
        if member_type not in RELATION_MEMBER_TYPES or not pairs:
            return

        gid = int(group_id)
        adj_max = float(ADJACENT_DIST_MAX.get(member_type, 0.0))
        near_max = float(NEAR_DIST_MAX.get(member_type, 0.0))

        coll_name = {"tree": "trees", "building": "buildings", "vehicle": "vehicles"}[member_type]
        objects = {obj["id"]: obj for obj in self.graph_data.get(coll_name, [])}

        centroids: Dict[int, Tuple[float, float]] = {}
        for mid in all_member_ids:
            obj = objects.get(int(mid))
            if not obj:
                continue
            c = self._centroid_for_member(member_type, int(mid), obj)
            if c is not None:
                centroids[int(mid)] = c

        delaunay_edges = self._delaunay_edges(centroids)

        for a, b in pairs:
            gap = self._object_shape_gap(member_type, a, b)
            if gap is None:
                continue
            key = (min(a, b), max(a, b))
            if gap <= adj_max:
                self.upsert_group_relation(member_type, gid, a, b, "adjacent")
            elif key in delaunay_edges and gap <= near_max:
                self.upsert_group_relation(member_type, gid, a, b, "near")

    # --- Connector spans & electrical grids ---

    def recompute_auto_macros(self) -> None:
        """Rebuild connector_spans and electrical_grids from current conductor data."""
        self._build_connector_spans_and_grids()
        self._sync_macro_instances_from_toplevel()

    def _sync_macro_instances_from_toplevel(self) -> None:
        """Populate macro_instances from connector_spans + electrical_grids for API compat."""
        macros: List[Dict[str, Any]] = []
        mid = 0
        for span in self.graph_data.get("connector_spans", []):
            macros.append({
                "id": mid, "type": "connector_span", "member_type": "conductor",
                "member_ids": span.get("conductor_ids", []),
                "user_class": "connector_span", "label": span.get("label", ""),
                "auto": True, "metadata": {"support_poles": span.get("poles", [])},
            })
            mid += 1
        for grid in self.graph_data.get("electrical_grids", []):
            macros.append({
                "id": mid, "type": "electrical_grid", "member_type": "connector_span",
                "member_ids": grid.get("span_ids", []),
                "user_class": "electrical_grid", "label": grid.get("label", ""),
                "auto": True, "metadata": {},
            })
            mid += 1
        self.graph_data["macro_instances"] = macros

    def list_macro_instances(self, macro_type: Optional[str] = None) -> List[Dict[str, Any]]:
        macro_list = self.graph_data.get("macro_instances", [])
        if macro_type is None:
            return macro_list
        return [m for m in macro_list if m.get("type") == macro_type]

    def _build_connector_spans_and_grids(self) -> None:
        """Build connector_spans and electrical_grids as top-level graph_data entries."""
        conductors = self.graph_data.get("conductors", [])
        self.graph_data["connector_spans"] = []
        self.graph_data["electrical_grids"] = []
        if not conductors:
            return

        span_key_to_conductors: Dict[Tuple[int, int], List[str]] = {}

        for c in conductors:
            poles = list({int(pid) for pid in c.get("poles", []) if pid is not None})
            if len(poles) < 2:
                continue
            poles.sort()
            a, b = poles[0], poles[-1]
            key = (a, b)
            uid_str = f"{c.get('link_idx')}_{c.get('conductor_id')}"
            span_key_to_conductors.setdefault(key, []).append(uid_str)

        spans: List[Dict[str, Any]] = []
        span_idx_by_key: Dict[Tuple[int, int], int] = {}
        for key, member_uids in span_key_to_conductors.items():
            if not member_uids:
                continue
            sid = len(spans)
            span_idx_by_key[key] = sid
            spans.append({
                "id": sid,
                "poles": list(key),
                "conductor_ids": member_uids,
                "label": f"Span {key[0]}-{key[1]}",
            })
        self.graph_data["connector_spans"] = spans

        if not spans:
            return

        pole_to_spans: Dict[int, List[int]] = {}
        for s in spans:
            for pid in s["poles"]:
                pole_to_spans.setdefault(pid, []).append(s["id"])

        neighbors: Dict[int, Set[int]] = {s["id"]: set() for s in spans}
        for span_list in pole_to_spans.values():
            for i in range(len(span_list)):
                for j in range(i + 1, len(span_list)):
                    neighbors[span_list[i]].add(span_list[j])
                    neighbors[span_list[j]].add(span_list[i])

        visited: Set[int] = set()
        grids: List[Dict[str, Any]] = []
        for s in spans:
            sid = s["id"]
            if sid in visited:
                continue
            comp: List[int] = []
            stack = [sid]
            visited.add(sid)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in neighbors.get(cur, []):
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            grids.append({
                "id": len(grids),
                "span_ids": comp,
                "label": f"Grid ({len(comp)} spans)",
            })
        self.graph_data["electrical_grids"] = grids

    def _ensure_groups(self, object_type: str) -> List[Dict[str, Any]]:
        key = _groups_key(object_type)
        if key not in self.graph_data:
            self.graph_data[key] = []
        return self.graph_data[key]

    def _next_group_id(self, object_type: str) -> int:
        groups_list = self.graph_data.get(_groups_key(object_type), [])
        if not groups_list:
            return 0
        return max(g.get("id", 0) for g in groups_list) + 1

    def _group_ids_and_members_for(
        self, object_type: str, object_ids: List[int]
    ) -> Tuple[Set[int], Set[int]]:
        """Returns (set of group_ids that contain any of object_ids, set of all member ids in those groups)."""
        objects_map = {obj["id"]: obj for obj in self.graph_data.get(object_type, [])}
        groups_list = self.graph_data.get(_groups_key(object_type), [])
        id_to_group: Dict[int, Dict[str, Any]] = {g["id"]: g for g in groups_list}

        touched_gids: Set[int] = set()
        all_members: Set[int] = set()
        for oid in object_ids:
            obj = objects_map.get(oid)
            if not obj:
                continue
            gid = obj.get("group_id")
            if gid is None:
                continue
            g = id_to_group.get(gid)
            if not g:
                continue
            touched_gids.add(gid)
            all_members.update(g.get("members", []))
        return touched_gids, all_members

    def modify_group(
        self, object_type: str, object_ids: List[int], operation: str
    ) -> Dict[str, Any]:
        """
        Modifies grouping using group_id + type_groups.
        operation: wedge (merge), split, delete (dissolve)
        """
        if object_type not in GROUP_TYPES:
            raise ValueError(f"Grouping not supported for object type: {object_type}")

        if object_type not in self.graph_data:
            self.graph_data[object_type] = []

        objects_map = {obj["id"]: obj for obj in self.graph_data[object_type]}
        for param_id in object_ids:
            if param_id not in objects_map:
                raise ValueError(f"Object ID {param_id} not found in {object_type}")

        groups_list = self._ensure_groups(object_type)

        if operation == "wedge":
            self._wedge_groups(object_type, objects_map, groups_list, object_ids)
        elif operation == "split":
            self._split_groups(object_type, objects_map, groups_list, object_ids)
        elif operation == "delete":
            self._delete_from_groups(object_type, objects_map, groups_list, object_ids)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return self.graph_data

    def _wedge_groups(
        self,
        object_type: str,
        objects_map: Dict[int, Any],
        groups_list: List[Dict[str, Any]],
        target_ids: List[int],
    ) -> None:
        """Merge all groups that contain any of target_ids into one group.
        Preserves existing inner relations; only computes cross-boundary pairs."""
        touched_gids, all_members = self._group_ids_and_members_for(object_type, target_ids)
        old_member_sets = {}
        for g in groups_list:
            if g["id"] in touched_gids:
                old_member_sets[g["id"]] = set(g.get("members", []))
        all_members.update(target_ids)

        if len(all_members) <= 1:
            return

        new_gid = self._next_group_id(object_type)
        members = sorted(all_members)
        groups_list.append({"id": new_gid, "members": members})

        for oid in members:
            objects_map[oid]["group_id"] = new_gid

        groups_list[:] = [g for g in groups_list if g["id"] not in touched_gids]

        member_type = self._member_type_for_object_type(object_type)
        if not member_type:
            return

        for r in self.graph_data.get("group_relations", []):
            if r.get("member_type") == member_type and int(r.get("group_id", -1)) in touched_gids:
                r["group_id"] = new_gid

        cross_pairs: Set[Tuple[int, int]] = set()
        sets_list = list(old_member_sets.values())
        ungrouped = set(target_ids) - set().union(*sets_list) if sets_list else set(target_ids)
        if ungrouped:
            sets_list.append(ungrouped)

        for i, s1 in enumerate(sets_list):
            for s2 in sets_list[i + 1:]:
                for a in s1:
                    for b in s2:
                        cross_pairs.add((min(a, b), max(a, b)))

        if cross_pairs:
            self._recompute_relations_for_pairs(member_type, new_gid, members, cross_pairs)

    def _split_groups(
        self,
        object_type: str,
        objects_map: Dict[int, Any],
        groups_list: List[Dict[str, Any]],
        target_ids: List[int],
    ) -> None:
        """Remove target_ids from their groups and form one new group with them.
        Preserves inner relations on both sides; drops cross-boundary relations."""
        target_set = set(target_ids)
        member_type = self._member_type_for_object_type(object_type)
        removed_ids_by_gid: Dict[int, Set[int]] = {}
        if member_type:
            for g in groups_list:
                gid = g.get("id")
                if gid is None:
                    continue
                members = g.get("members", [])
                removed = [m for m in members if m in target_set]
                kept = [m for m in members if m not in target_set]
                if removed and kept:
                    removed_ids_by_gid[int(gid)] = set(removed)

        for g in groups_list:
            members = g.get("members", [])
            new_members = [m for m in members if m not in target_set]
            if len(new_members) != len(members):
                g["members"] = new_members

        groups_list[:] = [g for g in groups_list if g.get("members")]

        for oid in target_ids:
            objects_map[oid].pop("group_id", None)

        new_gid = None
        if len(target_ids) >= 2:
            new_gid = self._next_group_id(object_type)
            members = sorted(target_ids)
            groups_list.append({"id": new_gid, "members": members})
            for oid in target_ids:
                objects_map[oid]["group_id"] = new_gid

        if member_type and removed_ids_by_gid:
            existing_rels = list(self.graph_data.get("group_relations", []))
            new_rels: List[Dict[str, Any]] = []
            for r in existing_rels:
                if r.get("member_type") != member_type:
                    new_rels.append(r)
                    continue
                rgid = int(r.get("group_id", -1))
                a_id = int(r.get("a_id", -1))
                b_id = int(r.get("b_id", -1))
                removed_ids = removed_ids_by_gid.get(rgid)
                if not removed_ids:
                    new_rels.append(r)
                    continue
                in_a = a_id in removed_ids
                in_b = b_id in removed_ids
                if in_a and in_b:
                    if new_gid is not None:
                        r["group_id"] = new_gid
                        new_rels.append(r)
                elif in_a or in_b:
                    continue
                else:
                    new_rels.append(r)
            self.graph_data["group_relations"] = new_rels

    def _delete_from_groups(
        self,
        object_type: str,
        objects_map: Dict[int, Any],
        groups_list: List[Dict[str, Any]],
        target_ids: List[int],
    ) -> None:
        """Remove target_ids from their groups. Relations involving them are deleted."""
        target_set = set(target_ids)
        member_type = self._member_type_for_object_type(object_type)

        touched_gids: Set[int] = set()
        for g in groups_list:
            members = g.get("members", [])
            new_members = [m for m in members if m not in target_set]
            if len(new_members) != len(members):
                touched_gids.add(g["id"])
                g["members"] = new_members

        groups_list[:] = [g for g in groups_list if g.get("members")]

        for oid in target_ids:
            objects_map[oid].pop("group_id", None)

        if member_type:
            self.graph_data["group_relations"] = [
                r for r in self.graph_data.get("group_relations", [])
                if not (
                    r.get("member_type") == member_type
                    and (int(r.get("a_id", -1)) in target_set or int(r.get("b_id", -1)) in target_set)
                )
            ]

