# Maintain Caffe GUI Tool

Read SUMMARY.md first, then README.md, docs/report-analysis.md,
docs/VALIDATION.md and NOTICE. Keep SUMMARY.md and this file synchronized
before commits when behavior, commands, tests or handoff context changes.

## Implementation

- Preserve CaffeNodeTree and historical node/socket IDs for old `.blend`
  files. New RNA properties use annotations; shared-weight mixin properties
  must be registered on concrete nodes. Register dependencies before nodes,
  unregister in reverse, and remove test data before unregistering classes.
- Keep fake users on new/imported graphs so they survive `.blend` saves.
- JSON is data only. Reject non-finite, out-of-range, wrongly typed, lossy or
  malformed documents transactionally; never accept silent RNA coercion.
- Preserve exact roundtrips of all six recovered graphs and the compact demo.
  Changes to compatibility rules need original-file and persistence checks.
- Export must preserve graph values and the process working directory.
  Validate dependencies per phase, quote protobuf and shell values, and
  protect previous complete exports when a write fails. Data is dual-phase;
  its inherited include_in is historical exporter scratch state.
- The four bundled grayscale derivative repairs are hash-pinned and remove
  only symmetric zero borders. Preserve original assets/graph values, emit
  repair comments, and reject arbitrary mismatched or learned fragments.
- Custom research Padding needs the external fork. Do not imply stock Caffe
  or OpenCV supports it. Never silently replace unsupported learned behavior.

## Verification

Create an external Python environment with requirements-test.txt. Run
`python3 tools/validate.py --blender /path/to/blender --python .venv/bin/python`.
Run both supported Blender versions after API changes. The harness uses
isolated factory-startup processes and temporary user settings, validates
against tests/schema/caffe.proto with protoc, builds twice, installs the
actual ZIP, exports numerical fixtures and tests CPU outputs against NumPy.
Use `--skip-protobuf` only when recording that schema checks were omitted.

Keep generated fixtures, logs, image outputs, models and datasets untracked.
Only sanitized aggregate validation JSON belongs in docs/validation/. Tests
use the existing public images without adding them to the add-on ZIP.
Numerical execution tests use deterministic test coefficients, not trained
denoising weights. Missing required runtime dependencies are failures in the
full harness, not evidence of successful execution.
Delete prior numerical evidence before every run and require a fresh report;
keep the stale-report rejection regressions in the full harness.

OpenCV Caffe loading requires version 4.x. Preserve finite-value checks and
any documented geometry guards; named intermediate inspection must avoid
fusion buffer reuse. Validate changed runtime behavior with asymmetric
filters, batch/gray/RGB/non-square shapes and all public-image cases.

Never launch training as an incidental test. If training is explicitly taken
on, obey the user's project monitoring and authorized-budget requirements.
Do not modify separate final-model repositories as part of GUI maintenance.

## Distribution

Build with `python3 tools/build_addon.py`; its ZIP metadata is deterministic.
Check package size/hash and installed-module origin. Preserve MIT and
Unlicense notices. The test schema has a separate BSD notice. Root legacy
entry points depend on the installed package; don't distribute them alone.
Keep existing research files immutable unless the user requests a change.
