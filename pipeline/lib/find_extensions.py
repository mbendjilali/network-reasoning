from collections import defaultdict
import math
import networkx as nx
import shapely as shp
from typing import List, Dict, Any, Tuple, Optional, Set

def dist3d(p1: List[float], p2: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def get_all_conductors(
    links: List[Dict],
    node_map: Optional[Dict[int, Dict]] = None,
) -> List[Dict]:
    """
    Flatten conductors from links. Each entry has link_idx, id, startpoint, endpoint, uid_tuple.
    If node_map is provided, also set source_id, target_id, point_at_source, point_at_target.
    """
    all_conductors = []
    for link_idx, link in enumerate(links):
        source_id = link.get('source')
        target_id = link.get('target')
        for c in link.get('conductors', []):
            start = c.get('startpoint')
            end = c.get('endpoint')
            if not start or not end:
                continue
            entry = dict(c)
            entry['link_idx'] = link_idx
            entry['id'] = c.get('id')
            entry['uid_tuple'] = (link_idx, c.get('id'))
            if node_map and source_id is not None and target_id is not None:
                s_node = node_map.get(source_id)
                t_node = node_map.get(target_id)
                if s_node and t_node:
                    s_pos = [s_node.get('X', 0), s_node.get('Y', 0), s_node.get('Z', 0)]
                    t_pos = [t_node.get('X', 0), t_node.get('Y', 0), t_node.get('Z', 0)]
                    if dist3d(start, s_pos) < dist3d(end, s_pos):
                        entry['point_at_source'] = start
                        entry['point_at_target'] = end
                    else:
                        entry['point_at_source'] = end
                        entry['point_at_target'] = start
                    entry['source_id'] = source_id
                    entry['target_id'] = target_id
            all_conductors.append(entry)
    return all_conductors

def _common_support_pole(c1: Dict, c2: Dict) -> Optional[int]:
    if 'source_id' not in c1 or 'source_id' not in c2:
        return None
    p1 = {c1['source_id'], c1['target_id']}
    p2 = {c2['source_id'], c2['target_id']}
    common = p1 & p2
    if len(common) == 0:
        return None
    elif len(common) == 1:
        return common.pop()
    else:
        return common


def _point_at_pole(c: Dict, pole_id: int) -> List[float]:
    """Return conductor c's attachment point xyz at the given support pole."""
    if pole_id == c.get('source_id'):
        return c['point_at_source']
    if pole_id == c.get('target_id'):
        return c['point_at_target']
    return []

def _direction_away_from_pole(c: Dict, pole_id: int) -> Optional[List[float]]:
    """Unit direction vector of conductor c at pole_id, pointing away from the pole along the conductor."""
    at_pole = _point_at_pole(c, pole_id)
    if pole_id == c.get('source_id'):
        other = c['point_at_target']
    else:
        other = c['point_at_source']
    if not at_pole or not other:
        return None
    dx = other[0] - at_pole[0]
    dy = other[1] - at_pole[1]
    dz = other[2] - at_pole[2]
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n < 1e-12:
        return None
    return [dx / n, dy / n, dz / n]

def _angle_between_directions(d1: List[float], d2: List[float]) -> float:
    """Angle in radians between two unit direction vectors (0 to pi)."""
    dot = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)

def find_extensions(
    conductors: List[Dict],
    tolerance: float = float("inf"),
) -> Tuple[Any, Dict, Dict, Dict]:
    """
    Core extension computation.

    Given a list of flattened conductors (as returned by `get_all_conductors`),
    build the undirected extension graph and the connected components, and also
    compute bifurcation relations.

    Returns (G, comp_map, bifurcations) where:
      - G is a networkx.Graph over conductor uid_tuples
      - comp_map maps uid_tuple -> component index
      - bifurcations is uid_tuple -> list[uid_tuple] for 'bifurcates' relations
    """
    uid_to_c = {c["uid_tuple"]: c for c in conductors}

    # Build pole_to_uids map: which conductors are supported by which pole
    pole_to_uids = {}
    for c in conductors:
        uid = c["uid_tuple"]
        # A conductor might be supported by source and target poles
        for pole_id in [c.get("source_id"), c.get("target_id")]:
            if pole_id is not None:
                pole_to_uids.setdefault(pole_id, []).append(uid)

    G = nx.Graph()
    G.add_nodes_from([c["uid_tuple"] for c in conductors])
    
    # We track which 'port' (end) of a conductor is used at a specific pole.
    # used_ports: set of (uid_tuple, pole_id)
    used_ports = set()
    
    # Track conductors that formed an elbow extension at a specific pole
    # Set of (uid_tuple, pole_id)
    elbow_extensions_at_pole = set()

    for pole_id, uids in pole_to_uids.items():
        # Gather all pairs at this pole with their distances
        pairs = []
        n = len(uids)
        for i in range(n):
            u1 = uids[i]
            c1 = uid_to_c[u1]
            p1 = _point_at_pole(c1, pole_id)
            if not p1: continue
            
            for j in range(i + 1, n):
                u2 = uids[j]

                # If u2 is part of an elbow extension at this pole, skip.
                if (u2, pole_id) in elbow_extensions_at_pole:
                    continue

                c2 = uid_to_c[u2]
                p2 = _point_at_pole(c2, pole_id)
                if not p2: continue
                
                # _common_support_pole returns the shared pole ID or None.
                # If they share > 1 pole (parallel), they shouldn't extend each other.
                if _common_support_pole(c1, c2) != pole_id:
                    continue

                d = dist3d(p1, p2)
                if d <= tolerance:
                    pairs.append((d, u1, u2))
        
        # Sort pairs by distance (greedy approach)
        pairs.sort(key=lambda x: x[0])
        
        elbow_threshold = math.pi / 6  # 60 degrees

        # Pass 1: Straight extensions
        for d, u1, u2 in pairs:
            # If either port is already used, we skip (greedy)
            if (u1, pole_id) in used_ports or (u2, pole_id) in used_ports:
                continue
            
            # Check angle for extension
            c1 = uid_to_c[u1]
            c2 = uid_to_c[u2]
            d1 = _direction_away_from_pole(c1, pole_id)
            d2 = _direction_away_from_pole(c2, pole_id)
            
            if d1 and d2:
                angle = _angle_between_directions(d1, d2)
                # Extension condition: angle approx 180 (pi).
                if math.pi - angle <= elbow_threshold:
                    # It is straight enough -> Extension
                    G.add_edge(u1, u2)
                    used_ports.add((u1, pole_id))
                    used_ports.add((u2, pole_id))

        # Pass 2: Connect remaining "terminals" at the pole (Elbow/Corner extensions)
        # Rule: if two terminal conductors (unused ports) form an elbow, they belong 
        # to a unique component (corner case).
        v_shape_threshold = math.pi / 3  # 60 degrees
        for d, u1, u2 in pairs:
            if (u1, pole_id) in used_ports or (u2, pole_id) in used_ports:
                continue
            
            # Check angle to avoid narrow V-shapes
            c1 = uid_to_c[u1]
            c2 = uid_to_c[u2]
            d1 = _direction_away_from_pole(c1, pole_id)
            d2 = _direction_away_from_pole(c2, pole_id)
            
            if d1 and d2:
                angle = _angle_between_directions(d1, d2)
                # If angle is too narrow (< 60 deg), it's a V-shape, not an elbow extension.
                if angle < v_shape_threshold:
                    continue

            # They are close (sorted pairs list), share only this pole (filtered in pairs list),
            # and are both unused (terminal at this pole). 
            # We connect them as an "elbow extension".
            G.add_edge(u1, u2)
            used_ports.add((u1, pole_id))
            used_ports.add((u2, pole_id))
            
            # Mark these as part of an elbow extension at this pole
            elbow_extensions_at_pole.add((u1, pole_id))
            elbow_extensions_at_pole.add((u2, pole_id))

    components = list(nx.connected_components(G))
    comp_map: Dict[Tuple, int] = {}
    for idx, comp in enumerate(components):
        for uid in comp:
            comp_map[uid] = idx

    # 4) Compute XY crosses (non-shared pole)
    crosses = build_cross_relations(conductors, comp_map)

    # 5) Compute pole-based relationships (Bifurcations and Pole Crosses)
    # This updates 'crosses' in-place with pole-based crossings.
    bifurcations = build_pole_relations(
        conductors, G, pole_to_uids, used_ports, crosses, 
        tolerance=tolerance,
        elbow_extensions_at_pole=elbow_extensions_at_pole
    )

    return G, comp_map, bifurcations, crosses

def build_cross_relations(
    conductors: List[Dict],
    comp_map: Dict[Tuple, int],
) -> Dict[Tuple, List[Tuple]]:
    """
    Compute 'crosses' relationship:
    - Two conductors intersecting in the XY plan
    - AND belonging to different extension components.
    - Excludes pairs that share a support pole (since they touch at endpoint).
    """
    crosses: Dict[Tuple, List[Tuple]] = defaultdict(list)
    
    # 1. Build 2D geometries for all conductors
    geoms = []
    uids = []
    for c in conductors:
        start = c.get("startpoint")
        end = c.get("endpoint")
        if start and end:
            # XY projection
            line = shp.LineString([start[:2], end[:2]])
            geoms.append(line)
            uids.append(c["uid_tuple"])
    
    if not geoms:
        return dict(crosses)

    # 2. Use STRtree for efficient intersection query
    tree = shp.STRtree(geoms)
    # query returns [indices_of_geoms, indices_of_tree_items] for intersecting pairs
    # In shapely 2.0+, query(geoms) returns indices of geometries in `geoms` that intersect tree geometries.
    # Result is a (2, N) array of indices.
    
    # Shape of result: [indices_of_input_geoms, indices_of_tree_geoms]
    # Here input_geoms is same as tree geoms.
    # We use 'intersects' predicate.
    pairs = tree.query(geoms, predicate="intersects")
    
    uid_to_c = {c["uid_tuple"]: c for c in conductors}

    for k in range(pairs.shape[1]):
        i = pairs[0][k]
        j = pairs[1][k]
        
        if i >= j: continue  # Avoid duplicates and self-intersection
        
        uid_i = uids[i]
        uid_j = uids[j]
        
        # Check component
        if comp_map.get(uid_i) == comp_map.get(uid_j):
            continue
            
        # Check shared pole constraint (exclude if they share a pole)
        # "Crosses happen between two conductors... if ... don't belong to same component"
        # Implied: if they share a pole, it's usually a Bifurcation (connection) or just touching.
        # Strict "Cross" usually implies crossing paths.
        # If we include shared-pole pairs, almost all bifurcations become crosses too.
        # We exclude shared poles to keep concepts distinct unless user specified otherwise.
        # Given "Crosses happen ... in XY plan", touching at pole is an intersection.
        # But if they share a pole, they are supported by it.
        # Let's exclude strictly shared poles to avoid clutter/redundancy with Bifurcation.
        c1 = uid_to_c[uid_i]
        c2 = uid_to_c[uid_j]
        
        common = _common_support_pole(c1, c2)
        if common is not None:
            continue
            
        crosses[uid_i].append(uid_j)
        crosses[uid_j].append(uid_i)
        
    return dict(crosses)

def build_pole_relations(
    conductors: List[Dict],
    extension_graph: Any,
    pole_to_uids: Dict[int, List[Tuple]],
    used_ports: set,
    crosses: Dict[Tuple, List[Tuple]],
    tolerance: float = 0.5,
    elbow_extensions_at_pole: Optional[Set[Tuple]] = None,
) -> Dict[Tuple, List[Tuple]]:
    """
    Compute relationships at poles: Bifurcations and Crosses.
    
    Logic:
    - Iterate pairs at a pole that are close and NOT already extensions.
    - Check if they form an elbow (fail direction check).
    - If so, check topology:
        - If (u is terminal at pole) XOR (v is terminal at pole) -> Bifurcation.
        - If (u not terminal) AND (v not terminal) -> Crossing.
        - If (u terminal) AND (v terminal) -> V-shape (ignored for now).
    
    If elbow_extensions_at_pole is provided, any conductor involved in an elbow extension
    at a specific pole will be excluded from forming bifurcations or crosses at that pole.

    Updates 'crosses' dict in-place for detected pole crossings.
    Returns 'bifurcations' dict.
    """
    uid_to_c = {c["uid_tuple"]: c for c in conductors}
    bifurcations: Dict[Tuple, List[Tuple]] = defaultdict(list)
    
    if elbow_extensions_at_pole is None:
        elbow_extensions_at_pole = set()
    
    elbow_threshold = math.pi / 6

    for pole_id, uids in pole_to_uids.items():
        n = len(uids)
        for i in range(n):
            u1 = uids[i]
            
            # If u1 is part of an elbow extension at this pole, skip ALL relationships for it at this pole.
            if (u1, pole_id) in elbow_extensions_at_pole:
                continue

            c1 = uid_to_c[u1]
            p1 = _point_at_pole(c1, pole_id)
            if not p1: continue
            
            for j in range(i + 1, n):
                u2 = uids[j]
                
                # If u2 is part of an elbow extension at this pole, skip.
                if (u2, pole_id) in elbow_extensions_at_pole:
                    continue

                c2 = uid_to_c[u2]
                p2 = _point_at_pole(c2, pole_id)
                if not p2: continue
                
                # Check for existing extension
                if extension_graph.has_edge(u1, u2):
                    continue
                
                # Must share exactly this pole
                shared = _common_support_pole(c1, c2)
                if shared != pole_id:
                    continue
                
                # Distance check
                d = dist3d(p1, p2)
                
                # Direction/Elbow Check
                d1 = _direction_away_from_pole(c1, pole_id)
                d2 = _direction_away_from_pole(c2, pole_id)
                
                is_elbow = False
                if d1 and d2:
                    angle = _angle_between_directions(d1, d2)
                    # "Fail the direction check for extensions" -> Elbow
                    if math.pi - angle > elbow_threshold:
                        is_elbow = True
                
                if not is_elbow:
                    continue

                # Topology Check
                # Terminal at pole = (uid, pole_id) NOT in used_ports
                term1 = (u1, pole_id) not in used_ports
                term2 = (u2, pole_id) not in used_ports
                
                if term1 != term2:
                    # XOR -> Bifurcation (Distance matters)
                    if d > tolerance:
                        continue
                    bifurcations[u1].append(u2)
                    bifurcations[u2].append(u1)
                elif (not term1) and (not term2):
                    # Neither terminal -> Crossing (Distance does NOT matter)
                    # Check XY angle to avoid parallel lines (e.g. dual circuit)
                    d1_xy = [d1[0], d1[1]]
                    d2_xy = [d2[0], d2[1]]
                    n1 = math.hypot(d1_xy[0], d1_xy[1])
                    n2 = math.hypot(d2_xy[0], d2_xy[1])
                    
                    if n1 > 1e-6 and n2 > 1e-6:
                        # Dot product of normalized XY vectors
                        dot = (d1_xy[0] * d2_xy[0] + d1_xy[1] * d2_xy[1]) / (n1 * n2)
                        dot = max(-1.0, min(1.0, dot))
                        angle_xy = math.acos(dot)
                        
                        # Filter out parallel (approx 0) and anti-parallel (approx 180) - say between 20 and 160 degrees
                        if math.radians(20) < angle_xy < math.radians(160):
                            if u2 not in crosses.setdefault(u1, []): crosses[u1].append(u2)
                            if u1 not in crosses.setdefault(u2, []): crosses[u2].append(u1)
                # Else: Both terminal -> V-shape (ignore)

    return dict(bifurcations)
