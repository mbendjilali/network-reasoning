import os
import json
import glob

def migrate_tile(graph_path):
    geom_path = graph_path.replace("graph", "geometry").replace("graph_", "geom_")
    
    with open(graph_path, 'r') as f:
        graph = json.load(f)
        
    if os.path.exists(geom_path):
        with open(geom_path, 'r') as f:
            geom = json.load(f)
    else:
        geom = {"scale": 1.0}

    # Setup geom containers
    geom.setdefault('poles', {})
    geom.setdefault('conductors', {})
    geom.setdefault('buildings', {})
    geom.setdefault('vehicles', {})
    geom.setdefault('trees', {})

    # Migrate poles
    for p in graph.get('poles', []):
        pid = str(p['id'])
        geom_p = {}
        for key in ['X', 'Y', 'Z', 'footprint']:
            if key in p:
                geom_p[key] = p.pop(key)
        if geom_p:
            geom['poles'][pid] = geom_p

    # Migrate conductors
    for c in graph.get('conductors', []):
        cid = f"{c.get('link_idx')}_{c.get('conductor_id')}"
        geom_c = {}
        for key in ['model', 'startpoint', 'endpoint']:
            if key in c:
                geom_c[key] = c.pop(key)
        if geom_c:
            geom['conductors'][cid] = geom_c

    # Migrate buildings
    for b in graph.get('buildings', []):
        bid = str(b['id'])
        geom_b = {}
        for key in ['min', 'max']:
            if key in b:
                geom_b[key] = b.pop(key)
        if geom_b:
            geom['buildings'][bid] = geom_b

    # Migrate vehicles
    for v in graph.get('vehicles', []):
        vid = str(v['id'])
        geom_v = {}
        for key in ['min', 'max']:
            if key in v:
                geom_v[key] = v.pop(key)
        if geom_v:
            geom['vehicles'][vid] = geom_v

    # Migrate trees
    for t in graph.get('trees', []):
        tid = str(t['id'])
        geom_t = {}
        for key in ['X', 'Y', 'Z', 'height', 'crown_radius', 'min', 'max']:
            if key in t:
                geom_t[key] = t.pop(key)
        if geom_t:
            geom['trees'][tid] = geom_t

    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=2, allow_nan=False)
        
    os.makedirs(os.path.dirname(geom_path), exist_ok=True)
    with open(geom_path, 'w') as f:
        json.dump(geom, f, indent=2, allow_nan=False)

    print(f"Migrated {graph_path}")

def main():
    for graph_file in glob.glob("data/graph/graph_*.json"):
        migrate_tile(graph_file)

if __name__ == "__main__":
    main()
