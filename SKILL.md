# caffe-gui-tool maintenance workflow

Read `SUMMARY.md`, this file and `README.md` before editing.

## Working on the repository

Read both scripts before documenting loading or export. Treat the Text Editor registration procedure as inferred until run. Refer current Blender users to Blender Neural Composer with its separate evidence. Retain H. Tomkins, Aaron Wetzler and Rom Hershkovitz attribution and the Unlicense.

## Documentation checks

- Check claims against the named source files and recorded results. Separate source inspection,
  configuration generation, runtime execution and benchmark evidence.
- Keep README setup requirements available without following an advanced guide. Check Markdown
  headings, links and image references, including source files omitted by a sparse checkout.
- Inspect labels inside diagrams as well as captions. Never turn a schematic or generated input
  into an implied successful run.
- Keep original source, credits and license notices intact for documentation-only changes.
- Review and synchronize `SUMMARY.md` and `SKILL.md` before every commit. Record any unrun checks
  as limitations rather than implying that validation passed.

## Current evidence boundary

Node/registration/exporter source inspection only; no Blender session, export, Caffe parse, inference or training. No automated suite or saved runtime validation.

No exact verified Blender version; legacy APIs and incomplete branches; subset of caffe.proto templates; output overwrite behavior.

## Modern update navigation

Keep the local New! badge and exact “Updated and ported for modern hardware”
link immediately below the root README title, targeting `updated/README.md`.
Put modernization summaries and links under its matching heading. Describe
Blender Neural Composer and `codex/modernize-and-validate` as separate
implementations with their own evidence; never apply their results to the
preserved historical master. Retain the reference body and original source.
