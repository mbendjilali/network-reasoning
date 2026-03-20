const API_BASE = '/api';

export async function fetchTileList() {
    const response = await fetch(`${API_BASE}/tiles`);
    if (!response.ok) throw new Error('Failed to fetch tiles');
    return await response.json();
}

export async function fetchTileData(tileId) {
    const response = await fetch(`${API_BASE}/graph/${tileId}`);
    if (!response.ok) throw new Error(`Failed to fetch graph data for tile ${tileId}`);
    return await response.json();
}

export async function fetchTileVersions(baseId) {
    const response = await fetch(`${API_BASE}/tile_versions/${baseId}`);
    if (!response.ok) throw new Error(`Failed to fetch versions for tile ${baseId}`);
    return await response.json();
}

export async function saveTile(tileId) {
    const response = await fetch(`${API_BASE}/save/${tileId}`, {
        method: 'POST'
    });
    if (!response.ok) throw new Error('Failed to save tile');
    return await response.json();
}

export async function editGroup(tileId, objectType, objectIds, operation) {
    const response = await fetch(`${API_BASE}/edit/group`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            tile_id: tileId,
            object_type: objectType,
            ids: objectIds,
            operation: operation
        })
    });
    if (!response.ok) throw new Error('Failed to edit group');
    return await response.json();
}
