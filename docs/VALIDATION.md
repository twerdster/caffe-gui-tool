# Validation and limits

Use `tools/validate.py` for the complete editor/export/runtime suite. It writes
logs and a summary under `artifacts/validation/` and fails if a required suite
fails. The same command runs in Linux CI for Blender 5.1.2 and 5.2.4. Final
sanitized local execution records are stored under `docs/validation/`.

## Executed results — 2026-11-12

| Environment | Complete harness | CPU tests | Public images | Synthetic cases |
| --- | --- | ---: | ---: | ---: |
| macOS Apple silicon, Blender 5.1.2 | Passed | 527 passed | 487 | 23 |
| macOS Apple silicon, Blender 5.2.4 LTS | Passed | 527 passed | 487 | 23 |

Both runs include schema parsing, 175 rejected malformed graph documents,
nine export regression groups, all original-file/persistence checks and
actual ZIP installation. Runtime versions were Python 3.13.11, OpenCV 4.14.0
and NumPy 2.5.4. Each run made 602 numerical comparisons and 510 bit-identical
repeat checks. Maximum absolute error against NumPy was
`7.152557373046875e-6`, within `rtol=4e-6, atol=3e-5`.
The public corpus comprises 68 test images, 400 additional images and 19
BM3D-style images at native resolution, including grayscale and color data.
Procedural cases cover constants, asymmetric ramps, impulses, checkerboards,
seeded noise, noncontiguous arrays, non-square shapes and batches.

The tested standard-layer chain is grouped/depthwise convolution with bias,
leaky ReLU, fractional Power, weighted Eltwise SUM and Eltwise PROD. All five
intermediates are compared on synthetic inputs; final outputs and repeatability
are compared on every photograph. This is an operator correctness test with
chosen coefficients, not a denoising quality measurement.

Sanitized full records: [Blender 5.1](validation/macos-blender-5.1.json),
[Blender 5.2](validation/macos-blender-5.2.json), and
[source hashes](validation/source-manifest.json). Both builds produced the
same 83,437-byte ZIP, SHA-256
`04c4b50b81e294465462e244882f3df92f4840ac20af6fc832a58058e20743d0`.

## Historical derivative repair

Four bundled grayscale derivative fragments (`Dmx1chan`, `Dmy1chan`,
`Dpx1chan`, `Dpy1chan`) declare seven taps, while 44 nodes in the recovered
grayscale graphs declare three-tap kernels. Caffe rejects the mismatched blob
shapes. At export, the pinned original fragments can be cropped to their
three central coefficients only when the removed symmetric borders are all
zero. An explanatory comment appears in affected exported layers. Original
assets and graph values remain unchanged. Arbitrary user fragments and
nonzero trimming are rejected; this is not a repair of learned checkpoints.

## Coverage

- Every one of the 29 editable Blender node types and inherited shared-weight
  settings; all six recovered graphs plus the compact demo roundtrip exactly.
- Each processing/loss node exports independently; all generated Caffe network
  and solver text parses against the pinned historical schema using protoc.
- Malformed imports, invalid properties/parents/links, graph cycles, missing
  inputs, phase dependencies, repeat exports and failed writes are exercised.
- All five original `.blend` files load with their expected node/link counts;
  graph persistence is checked by saving and reopening a new temporary file.
- The actual reproducible ZIP is installed, enabled, used and disabled from
  private temporary Blender settings.
- Standard-layer networks are constructed and exported by Blender, initialized
  with deterministic test coefficients, then executed in OpenCV CPU. Independent
  NumPy calculations check each intermediate across procedural patterns and
  the public photographs already present in the repository. Repeated execution
  is checked for consistent outputs.

## What these checks establish

The Blender tests check API compatibility, graph preservation, invalid-input
handling and exported definitions. Protobuf parsing checks field types and
syntax. Numerical tests check the exercised standard-layer operator chains
and input cases. These are separate forms of evidence.

The suite does not train a network, check gradients in Caffe, establish
convergence, or reproduce thesis denoising scores. Arbitrary user graphs may
have incompatible tensor shapes, invalid mathematical domains or unstable
learned parameters even when their definitions export successfully. Such
failures must be reported, not replaced with a different filter or hidden by
clipping intermediate values.

The historical `Padding` node requires the research Caffe implementation; the
optional OpenCV runner rejects it. OpenCV 5 has no Caffe importer, so the
external runtime dependency is bounded below version 5. The final thesis
models and their full BSD68 acceptance protocol remain in the separate
[Deep Patch-Beltrami](https://github.com/twerdster/deep-patch-beltrami) project.

CPU grouped convolutions are rejected if any such layer receives spatial
dimensions smaller than its kernel. Native OpenCV produced incorrect results
for these narrow inputs during testing. Kernel dimensions and layer input
shapes are inspected before execution, including downstream convolutions.
Fusion is disabled because repeated named-intermediate inspection otherwise
returned an activation's reused buffer instead of the convolution output.
Neither safeguard changes trained coefficients or mathematical operations.
