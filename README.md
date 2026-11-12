# Caffe-GUI-Tool

[![Python](https://img.shields.io/badge/code-Python-3776AB?logo=python&logoColor=white)](CaffeGenerate.py)
[![Historical Blender add-on](https://img.shields.io/badge/Blender-historical%20add--on-E87D0D?logo=blender&logoColor=white)](CaffeNodes.py)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-green)](LICENSE)

![Explanatory schematic: define a Blender node graph, traverse it with the Python exporter, and write Caffe network and solver prototxt files](docs/images/node-export-workflow.svg)

*Explanatory schematic based on `CaffeNodes.py` and `CaffeGenerate.py`. It is not a screenshot or a trained-model result.*

Uses Blender Nodes as a visual builder to generate caffe prototxt files.
This is a fork of the original project and includes functionality specific to our needs.

## Current continuation

For current Blender releases, use [Blender Neural Composer](https://github.com/twerdster/blender-neural-composer).
This repository preserves the historical add-on and research material; it has not been retested
with current Blender or Caffe as part of this documentation update.

## Original implementation

The node definitions live in [CaffeNodes.py](CaffeNodes.py), with network, solver and shell-script
export in [CaffeGenerate.py](CaffeGenerate.py). Exporting configuration files does not establish
that a Caffe environment can build or train the resulting graph.

The original project is by [H. Tomkins](https://github.com/hctomkins/caffe-gui-tool), with research
extensions by Aaron Wetzler and Rom Hershkovitz. Existing source attribution and the
[Unlicense](LICENSE) are retained.

### Todo

Implement all current caffe.proto functionality.
