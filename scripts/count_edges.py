import os
import json
import glob
import csv
from collections import defaultdict

def count_edges():
    graph_files = glob.glob("data/graph/graph_*.json")
    
    results = []
    all_rel_classes = set()
    
    for gf in sorted(graph_files):
        with open(gf, "r") as f:
            data = json.load(f)
            
        tile_id = os.path.basename(gf).replace("graph_", "").replace(".json", "")
        
        row = {"tile_id": tile_id}
        
        # 1. Conductor Extensions
        row["extensions"] = len(data.get("edges", []))
        
        # 2. Conductor Bifurcations (adjacency list is symmetric, so divide by 2)
        bifs = data.get("bifurcations", [])
        row["bifurcations"] = sum(len(b) for b in bifs) // 2
        
        # 3. Conductor Crosses (symmetric, divide by 2)
        crosses = data.get("crosses", [])
        row["crosses"] = sum(len(c) for c in crosses) // 2
        
        # 4. Conductor-Pole Attachments (bipartite edges)
        conductors = data.get("conductors", [])
        row["conductor_pole_attachments"] = sum(len(c.get("poles", [])) for c in conductors)
        
        # 5. Group Relations (intra-group semantic edges)
        group_rels = data.get("group_relations", [])
        rel_counts = defaultdict(int)
        for r in group_rels:
            rel_class = r.get("class", "unknown")
            # We prefix to make column names clear
            col_name = f"relation_{rel_class}"
            rel_counts[col_name] += 1
            all_rel_classes.add(col_name)
            
        # 6. Group Memberships (implicit bipartite edges: object -> group)
        group_member_edges = 0
        # 7. Group Cliques (implicit pairwise edges within a group)
        group_clique_edges = 0
        
        group_types = ["trees_groups", "buildings_groups", "vehicles_groups", "poles_groups", "conductors_groups"]
        for g_type in group_types:
            groups = data.get(g_type, [])
            for g in groups:
                n = len(g.get("members", []))
                group_member_edges += n
                if n > 1:
                    group_clique_edges += n * (n - 1) // 2
                    
        row["group_member_edges"] = group_member_edges
        row["group_clique_edges"] = group_clique_edges

        for k, v in rel_counts.items():
            row[k] = v
            
        results.append(row)
        
    all_rel_classes = sorted(list(all_rel_classes))
    fieldnames = [
        "tile_id", 
        "extensions", 
        "bifurcations", 
        "crosses", 
        "conductor_pole_attachments",
        "group_member_edges",
        "group_clique_edges"
    ] + all_rel_classes
    
    with open("edge_counts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Fill missing relations with 0
            out_row = {fn: row.get(fn, 0) for fn in fieldnames}
            out_row["tile_id"] = row["tile_id"]
            writer.writerow(out_row)
            
    print(f"Processed {len(results)} tiles.")
    print("Saved edge counts to edge_counts.csv")

if __name__ == "__main__":
    count_edges()
