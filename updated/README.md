# Caffe-GUI-Tool updates

## Updated and ported for modern hardware

This default branch retains the historical Caffe-GUI-Tool implementation;
modern Blender work is published separately. [Blender Neural Composer](https://github.com/twerdster/blender-neural-composer/tree/main/updated#updated-and-ported-for-modern-hardware)
adapts the node editor and exporter to current property APIs, with JSON
interchange, graph checks and recorded editor/export validation. The separate
[modernization branch](https://github.com/twerdster/caffe-gui-tool/tree/codex/modernize-and-validate)
incorporates that continuation and adds compatibility, installation and scoped
CPU-inference checks; those results do not validate this historical branch.

### Modern implementations and instructions

- [Blender Neural Composer update guide](https://github.com/twerdster/blender-neural-composer/tree/main/updated#updated-and-ported-for-modern-hardware):
  retained components, new code, setup and evidence.
- [Composer installation and use](https://github.com/twerdster/blender-neural-composer#install-and-use)
  and [validation scope](https://github.com/twerdster/blender-neural-composer/blob/main/docs/VALIDATION.md).
- [Caffe-GUI-Tool modernization branch README](https://github.com/twerdster/caffe-gui-tool/blob/codex/modernize-and-validate/README.md)
  and [its validation record](https://github.com/twerdster/caffe-gui-tool/blob/codex/modernize-and-validate/docs/VALIDATION.md).
  This is a separate branch with different implementation and evidence.

### Historical reference

The [reference README](../README.md#implementation-and-outputs),
[node definitions](../CaffeNodes.py) and [exporter](../CaffeGenerate.py) describe
the preserved scripts. Their current Blender compatibility and numerical
execution remain unverified. Original author credits and the
[Unlicense](../LICENSE) are retained.
