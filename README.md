# Network Reasoning

A scene graph construction and editing tool for classified 3D point clouds. It processes LiDAR data (LAZ/LAS) with semantic and instance labels to produce structured JSON graphs capturing spatial topology and inter-object relations. An integrated web-based 3D viewer enables interactive inspection, grouping, and relation editing.

## Project Structure

```
backend/                  FastAPI server and core domain logic
  main.py                 API entry point (endpoints, LAZ upload)
  core/
    graph_manager.py      Tile loading, grouping, relation computation
    scene_builder.py      Transforms graph + geometry into frontend-ready data

pipeline/                 Batch processing pipeline
  build_full_graph.py     Orchestrator: network → graph + geometry for all tiles
  lib/
    generate_json_graph.py  Pole/conductor/extension graph from network JSON
    add_buildings.py        Building extraction and OBB-based clustering
    add_vehicules.py        Vehicle extraction and OBB-based clustering
    add_trees.py            Tree extraction and OBB-based clustering
    find_extensions.py      Conductor extension detection
    find_supports.py        Pole–building support detection
    geom_utils.py           LAZ loading, DBSCAN-OBB, geometry helpers

frontend/                 Three.js web viewer
  index.html              Application shell and UI layout
  js/
    main.js               Tile loading and initialization
    viewer.js             Three.js scene, camera, rendering
    interaction.js        Selection, grouping, relation editing, shortcuts
    api.js                Backend API client
  css/style.css           Styling

scripts/                  Standalone utilities
  count_edges.py          CSV report of edge types across all graph files
  visualize_network.py    Matplotlib-based network visualization
  migrate_geom.py         Legacy geometry format migration

data/                     Runtime data (not tracked in git)
  network/                Raw network JSON files (poles, conductors)
  graph/                  Generated graph JSON files
  geometry/               Generated geometry JSON files
```

## Semantic Taxonomy

The point cloud classification follows a 15-class taxonomy:

| ID | Class        | Graph Object Type |
|----|--------------|-------------------|
| 0  | Ground       | —                 |
| 1  | Vegetation   | —                 |
| 2  | Car          | Vehicle           |
| 3  | Powerline    | Conductor         |
| 4  | Fence        | —                 |
| 5  | Tree         | Tree              |
| 6  | Pick-up      | Vehicle           |
| 7  | Van & Truck  | Vehicle           |
| 8  | Heavy-duty   | Vehicle           |
| 9  | Utility pole | Pole              |
| 10 | Light pole   | Pole              |
| 11 | Traffic pole | Pole              |
| 12 | Habitat      | Building          |
| 13 | Complex      | Building          |
| 14 | Annex        | Building          |

## Installation

**Requirements:** Python 3.8+, a modern browser with WebGL support.

```bash
pip install -r requirements.txt
```

Dependencies: `fastapi`, `uvicorn`, `numpy`, `scipy`, `laspy`, `scikit-image`, `shapely`, `networkx`, `pydantic`, `tqdm`, `python-multipart`.

## Data Preparation

### Input: LAZ/LAS Point Clouds

Each tile is a LAZ (or LAS) file with per-point fields:

- **classification** — semantic class ID (see taxonomy above)
- **instance** (or equivalent) — per-object instance ID within each class

### Input: Network JSON (Optional)

If an electrical network description is available, provide a JSON file per tile with node/link definitions (poles, conductors, connectivity). This enables the pipeline to compute extensions, bifurcations, crosses, connector spans, and electrical grids.

Without a network file, the pipeline still processes buildings, vehicles, and trees.

## Usage

### Option A: Batch Pipeline (CLI)

Process all tiles from a network directory and LAZ directory:

```bash
export PYTHONPATH=$PYTHONPATH:.
python pipeline/build_full_graph.py \
  --input data/network \
  --laz_dir path/to/laz_tiles \
  --output data/graph \
  --geom_output data/geometry
```

This produces `graph_<tile_id>.json` and `geom_<tile_id>.json` for each tile.

### Option B: Direct LAZ Upload (UI)

Start the server and use the built-in upload feature:

```bash
python -m backend.main
```

Open `http://localhost:8000`, click **Upload**, select a LAZ/LAS file (and optionally a network JSON), and click **Process**. The tile is processed on-the-fly and immediately available in the viewer.

### Running the Viewer

```bash
python -m backend.main
```

Navigate to `http://localhost:8000`. Select a tile from the dropdown to load it.

## Graph JSON Schema

Each `graph_<tile_id>.json` contains only topological and semantic data:

```json
{
  "poles": [
    { "id": 0, "sem_class": 9, "is_ground": false, "is_building_support": false }
  ],
  "conductors": [
    { "link_idx": 0, "conductor_id": 0, "poles": [0, 1], "component": 0, "sem_class": 3 }
  ],
  "edges": [ [0, 1] ],
  "bifurcations": [ [2, 3] ],
  "crosses": [ [4, 5] ],

  "buildings": [
    { "id": 0, "sem_class": 12, "group_id": 0 }
  ],
  "vehicles": [
    { "id": 0, "sem_class": 2, "group_id": 0 }
  ],
  "trees": [
    { "id": 0, "sem_class": 5, "group_id": 0 }
  ],

  "building_groups": [ { "id": 0, "members": [0, 1, 2] } ],
  "vehicle_groups":  [ { "id": 0, "members": [0, 1] } ],
  "tree_groups":     [ { "id": 0, "members": [0, 1, 2, 3] } ],

  "group_relations": [
    { "id": 0, "member_type": "building", "group_id": 0, "a_id": 0, "b_id": 1, "class": "adjacent" }
  ],

  "connector_spans": [
    { "id": 0, "poles": [0, 1], "conductor_ids": ["0_0", "0_1"], "label": "Span 0-1" }
  ],
  "electrical_grids": [
    { "id": 0, "span_ids": [0, 1, 2], "label": "Grid (3 spans)" }
  ]
}
```

### Relation Types

- **`adjacent`** — OBB gap ≤ class-specific threshold (building: 1.0 m, vehicle: 0.25 m, tree: 0.25 m)
- **`near`** — Connected by a Delaunay triangulation edge with OBB gap ≤ class-specific threshold (building: 15.0 m, vehicle: 8.0 m, tree: 20.0 m). `adjacent` takes priority over `near`.

### Clustering

Objects are clustered per-class using DBSCAN on OBB (Oriented Bounding Box) shortest distances. Each cluster becomes a group (`*_groups`). Relations are computed within each group using Delaunay triangulation on 2D centroids.

## Geometry JSON Schema

Each `geom_<tile_id>.json` contains visualization data only:

```json
{
  "scale": 1.0,
  "poles": { "<id>": { "X": 0.0, "Y": 0.0, "Z": 0.0, "footprint": { "min": [], "max": [], "rotation": [] } } },
  "conductors": { "<uid>": { "model": {}, "startpoint": [], "endpoint": [] } },
  "buildings": { "<id>": { "min": [], "max": [], "obb": { "center": [], "extents": [], "axes": [], "footprint": [] } } },
  "buildingHulls": { "<id>": { "vertices": [], "faces": [] } },
  "vehicles": { "<id>": { "min": [], "max": [], "obb": { "center": [], "extents": [], "axes": [], "footprint": [] } } },
  "vehicleHulls": { "<id>": { "vertices": [], "faces": [] } },
  "trees": { "<id>": { "X": 0.0, "Y": 0.0, "Z": 0.0, "height": 0.0, "crown_radius": 0.0, "obb": {} } }
}
```

## Editing Tool Controls

### Mouse

| Action              | Description                         |
|---------------------|-------------------------------------|
| Click               | Select object                       |
| Shift + Click       | Multi-select (add to selection)     |
| Shift + Drag        | Lasso selection                     |
| Double Click         | Clear selection / Reset view        |
| Left Drag           | Rotate camera                       |
| Right Drag          | Pan camera                          |
| Scroll              | Zoom                                |

### Keyboard

| Key            | Action                                |
|----------------|---------------------------------------|
| W              | Wedge (merge selected macro-instances)|
| X              | Split macro-instance                  |
| C              | Delete macro-instance (dissolve)      |
| G              | Cycle relation type (adjacent / near) |
| Space          | Cycle color scheme                    |
| Shift + Space  | Randomize colors                      |
| A              | Toggle conductors                     |
| Z              | Toggle poles                          |
| E              | Toggle buildings                      |
| R              | Toggle vehicles                       |
| T              | Toggle trees                          |
| S              | Save current edits                    |

### Grouping Operations

- **Wedge (W):** Merges two or more clusters into one. Existing inner relations are preserved; only cross-boundary pairs are computed.
- **Split (X):** Extracts the selected subset from its cluster into a new one. Cross-boundary relations are removed; inner relations of both parts are preserved.
- **Delete (C):** Removes selected members from their cluster. Relations involving removed members are deleted.

## API Endpoints

| Method | Path                              | Description                          |
|--------|-----------------------------------|--------------------------------------|
| GET    | `/api/tiles`                      | List available tile IDs              |
| GET    | `/api/tile_versions/{base_id}`    | List versions (original + edits)     |
| GET    | `/api/graph/{tile_id}`            | Load tile data for the viewer        |
| POST   | `/api/save/{tile_id}`             | Save edits as a new version          |
| POST   | `/api/edit/group`                 | Wedge / split / delete groups        |
| PATCH  | `/api/group_relation/{tile_id}`   | Update group relations               |
| POST   | `/api/load_laz`                   | Upload and process a LAZ/LAS file    |

## License

*To be determined.*
