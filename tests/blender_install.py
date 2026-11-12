"""Install the actual deliverable ZIP into an isolated Blender user directory."""
import json
from pathlib import Path
import sys
import addon_utils
import bpy

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / 'dist' / 'caffe-gui-tool-2.0.0.zip'
assert archive.is_file(), 'Build the add-on ZIP first'
bpy.ops.preferences.addon_install(filepath=str(archive))
addon_utils.enable('neural_operator_blender', default_set=False, persistent=False)
import neural_operator_blender as addon
from neural_operator_blender import composer, graph
assert addon.bl_info['version'] == (2, 0, 0)
assert not Path(addon.__file__).resolve().is_relative_to(ROOT / 'neural_operator_blender')
tree = graph.load(composer.EXAMPLES / 'operator_demo.json')
summary = graph.validate(tree)
assert summary['nodes'] > 0
bpy.data.node_groups.remove(tree)
addon_utils.disable('neural_operator_blender', default_set=False)
print('VALIDATION_JSON=' + json.dumps({'blender': bpy.app.version_string,
      'checks': ['actual ZIP installs, enables, loads example and disables'], 'graph': summary}))
