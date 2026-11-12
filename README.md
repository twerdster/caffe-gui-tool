# Caffe GUI Tool

A visual neural network editor for current Blender, preserving the computation
graphs used in Aaron Wetzler and Rom Hershkovitz's Patch-Beltrami research.
Build networks with nodes, change connections, stages and shared weights, save
portable JSON, and export Caffe train/test, deployment and solver definitions.

This update brings the modern [Blender Neural Composer](https://github.com/twerdster/blender-neural-composer)
implementation back into the original repository and adds stricter validation,
reliable exports, legacy compatibility and numerical execution tests.
See [the research interpretation](docs/report-analysis.md),
[validation and limitations](docs/VALIDATION.md), and [source attribution](NOTICE).

## Install and use

Build the dependency-free Blender add-on:

```sh
python3 tools/build_addon.py
```

In Blender, choose **Edit → Preferences → Add-ons → Install from Disk…**,
select `dist/caffe-gui-tool-2.0.0.zip`, and enable **Caffe GUI Tool / Neural
Operator Composer**. Disable any older Caffe Nodes or Neural Operator Composer
installation first: they use the same historical node identifiers.

Open a Node Editor, select **Neural Operator Composer**, and press **N** to open
the **Composer** sidebar. Load an example, expand nodes to edit parameters, or
use **Add → Neural Operators**. Imported and new graphs persist in `.blend`
files. Use **Validate Graph**, **Export Neural Graph JSON**, or **Export Caffe
Solution**. Set the Data deployment shape and source paths, and the Solver name
and executable directory before exporting. Each Data node describes one input
shape; use separate Data nodes for differently shaped inputs. Data nodes
always emit both train/test definitions; processing nodes use their phase setting.

Six recovered research graphs and one illustrative compact graph are included.
The compact example has **untrained weights**. These intermediate historical
architectures are distinct from the final thesis models in
[Deep Patch-Beltrami](https://github.com/twerdster/deep-patch-beltrami).

To open an original `.blend`, enable this add-on **before** opening the file,
then save a new copy after checking external paths. Root `CaffeNodes.py` and
`CaffeGenerate.py`, and the historical `patch_beltrami/scripts/addons/` entries,
delegate to the maintained package; install the complete ZIP, not individual
Python files. The legacy Create Solution shortcut exports the entire active
graph to its Solver directory.

## Test

The Blender tests run in separate factory-startup processes and do not modify
the original `.blend` files or your open scene.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
python3 tools/validate.py --blender /path/to/blender --python .venv/bin/python
```

Install `protoc` (the Protocol Buffers compiler) for schema verification. The
test harness requires it unless `--skip-protobuf` is explicitly supplied.
Results and logs go to ignored `artifacts/validation/`; an installation test
uses private temporary Blender settings. The suite covers all 29 node types,
all bundled JSON examples, original `.blend` files, persistence, malformed
imports, phase handling, failed exports, repeatability and the actual ZIP.
Numerical tests execute Blender-exported standard-layer networks in OpenCV
against an independent NumPy reference, using procedural data and the public
images already present in this repository.

For an individual Blender test:

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python tests/blender_test.py -- --caffe-proto tests/schema/caffe.proto
```

## Run exported networks on CPU

`tools/run_caffe.py` provides optional OpenCV inference for standard Caffe
layers. Run `python tools/run_caffe.py --help` for its input and output options.
Its dependencies live in the external Python environment above; Blender does
not need NumPy, OpenCV, Caffe or CUDA installed into its Python.

Use **OpenCV 4.x** (`>=4.10,<5`); OpenCV 5 removed its Caffe importer. A deployment
definition alone does not contain initialized learned weights: supply the
matching `.caffemodel` when the graph needs one. The runner checks finite input
and output and reports unsupported networks. It does not train networks or
implement the historical custom `Padding` layer. Those research graphs still
require the [research Caffe fork](https://github.com/twerdster/caffe) or a
separately validated compatible implementation. Export/schema checks do not
establish training convergence or reproduce thesis denoising scores.

The CPU runner rejects grouped convolutions when a layer's spatial input is
smaller than its kernel: testing found incorrect OpenCV results on those
geometries. It disables graph fusion to keep named intermediate outputs
consistent across repeated calls. Inputs are explicit `.npy` N,C,H,W arrays;
no scaling, channel conversion or clipping is applied automatically.

## Origins and licensing

Original node editor by [H. Tomkins](https://github.com/hctomkins/caffe-gui-tool),
extended by Aaron Wetzler and Rom Hershkovitz. The thesis describes this visual
designer in §4.5.2 and Figure 4.17, printed pages 63–64:

Aaron Wetzler, *Topics in Computational Geometric Vision*, PhD thesis,
Technion — Israel Institute of Technology, August 2017.

Historical code, graphs and operator coefficients retain the [Unlicense](LICENSE).
Modern interface, tests, packaging and runtime tooling use
[MIT](LICENSES/MIT.txt). The vendored test schema retains the
[Caffe BSD license](LICENSES/Caffe-BSD.txt). Existing dataset images retain
their original terms; they are not included in the installable ZIP.
