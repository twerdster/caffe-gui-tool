# caffe-gui-tool context

## Purpose

Historical Blender node editor and Python exporter for Caffe train/test, deployment and solver definitions plus a training shell command.

## Implementation evidence

Both scripts register Blender components. The exporter writes three prototxt files and train_<name>.sh in the configured output directory; it does not implement numerical execution.

Key entry points: `CaffeNodes.py`, `CaffeGenerate.py`.

## Documentation state

The README explains operations, requirements, validation scope and limitations. Its pre-existing explanatory SVG is retained and labeled as a schematic rather than a runtime result. Source files and attribution are unchanged.

## Validation scope

Node/registration/exporter source inspection only; no Blender session, export, Caffe parse, inference or training. No automated suite or saved runtime validation.

## Limitations

No exact verified Blender version; legacy APIs and incomplete branches; subset of caffe.proto templates; output overwrite behavior.
