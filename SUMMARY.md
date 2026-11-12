# Project state

Caffe GUI Tool 2.0.0 modernizes the original Blender-based Caffe graph editor.
The implementation incorporates Blender Neural Composer 1.0.0 at revision
76154fe73e305cc6831b296dd071cdcd46fc867d, then adds import/export hardening,
legacy entry points, persistence/installation tests and actual CPU inference
tests. Read README.md, docs/report-analysis.md, docs/VALIDATION.md and NOTICE.

## Code and commands

- `neural_operator_blender/`: maintained Blender package, 29 node types,
  portable JSON interchange, Composer sidebar, Caffe solution export.
- Root `CaffeNodes.py`/`CaffeGenerate.py` and their historical scripts copies
  delegate to this package. Install the entire ZIP; keep historical node IDs.
- `python3 tools/build_addon.py` builds the deterministic, dependency-free
  `dist/caffe-gui-tool-2.0.0.zip`. No research data or models are bundled.
- `python3 tools/validate.py --blender /path/to/blender --python .venv/bin/python`
  runs all suites, schema parsing, ZIP installation and numerical tests.
  External Python dependencies are in requirements-test.txt; `protoc` is
  required unless schema checking is explicitly skipped.
- `tools/run_caffe.py` is optional standard-layer OpenCV 4 CPU inference,
  outside Blender. It requires explicit initialized weights when needed,
  checks numerical failures and does not train or support custom Padding.
- Generated logs, fixtures and results stay in ignored `artifacts/` and dist/.
  All original `.blend`, MATLAB, image, prototxt and checkpoint assets are
  preserved; no large new datasets or training runs were created.
- The README is a technical overview with a real Blender screenshot,
  graph/export schematic and CPU test summary. Detailed commands live in
  docs/USAGE.md. Regenerate the SVGs with tools/build_readme_visuals.py;
  image provenance is documented in docs/images/README.md. The add-on ZIP
  includes the supporting docs and these three explicitly allowed visuals.
- The README and diagram copy use a plain maintainer voice. Claude Code
  reviewed the wording: slogans and the feature-sales table were removed,
  structural validation is distinguished from shape inference, and numerical
  coverage names the tested operator chain and separates 510 inputs from
  the 527-test total. Review artifacts stay under ignored artifacts/.

## Research boundaries

The thesis §4.5.2/Figure 4.17 motivates an editable visual computation graph,
shared weights and flexible stages. Six recovered JSON graphs and the
untrained compact demo are intermediate examples, not final thesis models.
The tiny 26-step checkpoint in this repository is also not a final model.
Final pretrained denoising and BSD68 reproduction live in the separate
`twerdster/deep-patch-beltrami` project. Do not infer denoising quality or
training convergence from export or procedural reference tests.

The exporter targets the historical research Caffe schema (vendored under
tests/schema/ with its BSD license), including custom Padding. Data nodes
always emit TRAIN and TEST definitions; inherited Data `include_in` values
are legacy scratch state. Processing-node phase settings are respected.

The exporter repairs a historical 7-tap blob / 3-tap kernel mismatch only for
four hash-pinned grayscale derivative fragments by removing symmetric zero
borders; affected exported layers carry comments. Other mismatches fail.
The CPU runner disables fusion for stable intermediate reads and rejects
grouped-convolution inputs smaller than the layer's actual kernel.

Historical adaptations/assets retain the Unlicense. Modern interface,
interchange, compatibility, runtime tools and tests use MIT. See NOTICE and
LICENSES/. No manuscript, private portraits or machine-specific paths should
be added to the public package.

Validation evidence and exact supported versions are recorded in
docs/VALIDATION.md and docs/validation/. Review that evidence after changes;
prior results alone do not validate changed code.

2026-11-12 local complete-harness runs pass Blender 5.1.2 and 5.2.4 LTS on
Apple silicon, with protoc enabled and installed-ZIP verification. Each run
passes 527 CPU tests across 23 synthetic cases and 487 public images, 602
NumPy comparisons and 510 exact repeat checks (maximum absolute error
7.15e-6). The initial code-validation ZIPs were 83,437 bytes with SHA-256
04c4b50b81e294465462e244882f3df92f4840ac20af6fc832a58058e20743d0.
The later visual README refresh adds documentation assets to the ZIP;
docs/validation/documentation-update.json records that package's checks.
Linux CI runs the same full harness on both versions; consult its live check
results for remote status. No Caffe training/gradient/convergence test was run.
The full harness explicitly preflights external runtime dependencies and
requires a newly written numerical report. Two harness regressions prevent
an old report from certifying skipped tests when reusing an output directory.

Visual documentation package verification passes actual ZIP installation in
Blender 5.1.2 and 5.2.4 LTS, including all README images. The current archive
is 273050 bytes; SHA-256 34e48b32b0b26a395996963cc71315229fd202a0d7a581b8ca5591b23fd61808.
