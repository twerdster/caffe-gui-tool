# Caffe GUI Tool

A Blender add-on for editing Caffe neural networks as node graphs. Set layer
parameters, connect inputs and outputs, configure shared weights, and export
Caffe network and solver definitions.

The tool was extended for Patch-Beltrami denoising research. Each network stage
contains derivative filters, metric calculations, flux, divergence and an image
update. The node editor made it easier to change this structure, repeat stages,
and control which parameters were shared.

![Compact operator graph with convolution settings expanded in Blender 5.2.4 LTS](docs/images/composer-overview.png)

*The compact example in Blender 5.2.4 LTS. Its weights are untrained.*

## Current version

Version 2.0.0 incorporates [Blender Neural Composer](https://github.com/twerdster/blender-neural-composer).
It updates Blender property registration and the node editor interface while
preserving the node identifiers used by the original `.blend` files.

- 29 node types, including convolution, pooling, elementwise operations, losses
  and solver settings.
- Graph storage in `.blend` files and portable JSON, including parameters,
  connections and layout.
- Structural validation for missing inputs, duplicate output names and cycles;
  train/test phase dependencies are checked during export.
- Caffe train/test, deployment and solver prototxt export, plus a training shell
  script. Export writes are staged, with rollback on failure.
- Five original `.blend` files, six recovered JSON graphs and a compact example.

## Installation

Build the ZIP from this checkout. The add-on needs no extra Python packages:

```sh
python3 tools/build_addon.py
```

1. In Blender, choose **Edit → Preferences → Add-ons → Install from Disk…**
   and select `dist/caffe-gui-tool-2.0.0.zip`.
2. Disable any older Caffe Nodes or Neural Operator Composer installation, then
   enable **Caffe GUI Tool / Neural Operator Composer**.
3. Open a Node Editor, select **Neural Operator Composer**, and press **N**.
   The **Composer** sidebar loads examples, validates graphs and exports files.
   Add layers through **Add → Neural Operators**.

See [docs/USAGE.md](docs/USAGE.md) for data paths, export settings, CPU inference
and test commands.

![Graph editing, structural validation and Caffe export, with a schematic Beltrami update](docs/images/workflow.svg)

## Validation

The complete test suite passes on Blender **5.1.2** and **5.2.4 LTS**, on macOS
Apple silicon and [Linux CI](https://github.com/twerdster/caffe-gui-tool/actions/workflows/validate.yml).
Editor and export tests cover all 29 node types, graph round trips, original
files, malformed imports, failed exports and installation from the ZIP.

The CPU suite passes **527 tests**: 510 numerical cases using 487 repository
images and 23 synthetic inputs, plus 17 failure-mode, CLI and guard tests.
The numerical cases run an exported chain in OpenCV 4 on the CPU:
grouped/depthwise convolution with bias, leaky ReLU, fractional Power, and
Eltwise SUM and PROD. A separate NumPy implementation checks 602 results: five layer
outputs for each synthetic input and one final output for each image. Repeated
execution gives exactly equal outputs for all 510 inputs under the tested runtime.

![CPU suite per run: 527 tests, 487 images and 23 synthetic inputs, 602 NumPy comparisons and 510 exact repeat checks](docs/images/validation.svg)

## Scope and limitations

The editor exports definitions; training runs separately. The CPU tests use
deterministic test weights and do not evaluate training, gradients, convergence
or denoising quality. Structural graph validation does not infer tensor shapes
or establish numerical stability.

The historical `Padding` layer requires the [research Caffe fork](https://github.com/twerdster/caffe).
The optional OpenCV 4 CPU runner does not support it. An intermediate 26-step
checkpoint is retained here; final trained denoising models are maintained in
[Deep Patch-Beltrami](https://github.com/twerdster/deep-patch-beltrami).

[Validation details](docs/VALIDATION.md) · [Research background](docs/report-analysis.md)

## Attribution and licenses

Original node editor by [H. Tomkins](https://github.com/hctomkins/caffe-gui-tool),
extended by Aaron Wetzler and Rom Hershkovitz. The tool is described in §4.5.2
and Figure 4.17 of Aaron Wetzler, *Topics in Computational Geometric Vision*,
PhD thesis, Technion — Israel Institute of Technology, August 2017.

Historical code, graphs and operator coefficients retain the [Unlicense](LICENSE).
Modern interface, tests, packaging and runtime tooling use [MIT](LICENSES/MIT.txt).
The test schema retains the [Caffe BSD license](LICENSES/Caffe-BSD.txt).
Dataset images retain their original terms and are excluded from the add-on
ZIP. See [NOTICE](NOTICE) for attribution.
