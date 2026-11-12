# Resolve this repository when the historical scripts directory is used.
from pathlib import Path
import sys
repository = Path(__file__).resolve().parents[3]
if (repository / 'neural_operator_blender').is_dir():
    sys.path.insert(0, str(repository))
"""Compatibility entry point. Install the ZIP built by tools/build_addon.py."""
from neural_operator_blender import bl_info, register, unregister
from neural_operator_blender.caffe_nodes import *  # historical class names
# Register the complete maintained editor, including its Composer sidebar.
from neural_operator_blender import bl_info, register, unregister

if __name__ == '__main__':
    register()
