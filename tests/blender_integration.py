"""Legacy file compatibility and persistent graph tests in an isolated process."""
import json
from pathlib import Path
import sys
import tempfile

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import CaffeNodes
import CaffeGenerate
from neural_operator_blender import graph, composer

CaffeNodes.register()
assert CaffeNodes.bl_info['version'] == (2, 0, 0)
CaffeGenerate.register()
assert hasattr(bpy.ops.nodes, 'make_solver')
CaffeGenerate.unregister()

expected = {
    'PatchBeltrami-aaron-base.blend': (31, 42),
    'PatchBeltrami-aaron.blend': (96, 137),
    'PatchBeltrami-aaron_BW.blend': (303, 442),
    'PatchBeltrami-aaron_BW_single.blend': (30, 43),
    'PatchBeltrami-aaron_RGB.blend': (96, 137),
}
files = sorted((ROOT / 'patch_beltrami').glob('*.blend'))
assert {file.name for file in files} == set(expected)
results = {'blender': bpy.app.version_string, 'legacy_files': [], 'checks': []}
for file in files:
    bpy.ops.wm.open_mainfile(filepath=str(file), use_scripts=False)
    trees = [tree for tree in bpy.data.node_groups if tree.bl_idname == 'CaffeNodeTree']
    assert len(trees) == 1
    tree = trees[0]
    assert (len(tree.nodes), len(tree.links)) == expected[file.name]
    assert all(node.bl_idname != 'NodeUndefined' for node in tree.nodes)
    document = graph.to_dict(tree)
    # Tree names change on import because the legacy tree still exists.
    restored = graph.from_dict(document)
    roundtrip = graph.to_dict(restored)
    roundtrip['name'] = document['name']
    assert roundtrip == document, file.name
    bpy.data.node_groups.remove(restored)
    results['legacy_files'].append({'name': file.name, 'nodes': len(tree.nodes), 'links': len(tree.links)})

with tempfile.TemporaryDirectory() as temporary:
    tree = graph.load(composer.EXAMPLES / 'operator_demo.json')
    name = tree.name
    before = graph.to_dict(tree)
    path = Path(temporary) / 'persistent.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    bpy.ops.wm.open_mainfile(filepath=str(path), use_scripts=False)
    assert graph.to_dict(bpy.data.node_groups[name]) == before
results['checks'].extend(['legacy module registration and exporter shortcut',
                          'all five original blend files load with known node types and exact JSON roundtrip',
                          'imported graph survives blend save and reopen'])
# Remove all custom data before class unregistration to avoid querying expired RNA.
for tree in list(bpy.data.node_groups):
    if tree.bl_idname == 'CaffeNodeTree':
        bpy.data.node_groups.remove(tree)
CaffeNodes.unregister()
print('VALIDATION_JSON=' + json.dumps(results, sort_keys=True))
