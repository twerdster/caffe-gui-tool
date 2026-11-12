# README visuals

- `composer-overview.png` is an actual capture of Caffe GUI Tool 2.0.0 in
  Blender 5.2.4 LTS. It shows the bundled, **untrained** compact operator with
  the learned-x-derivative node expanded. Only the presentation layout and
  selection were adjusted. The screenshot came from a separate factory-startup
  Blender process with private settings and temporary storage.
- `workflow.svg` is an explanatory schematic, not a screenshot or measured
  denoising result. Its mathematical stages summarize one Beltrami-style update.
- `validation.svg` summarizes the recorded numerical and compatibility checks.
  Counts are per supported-version run; 487 images and 23 synthetic cases are
  distinct from the 527 CPU test count. Chosen test coefficients do not measure
  the quality of a trained denoiser.

Regenerate the two SVGs with `python3 tools/build_readme_visuals.py`. Values are
read from the checked-in validation record. Update the record only after a
real test run. Both SVGs have accessible titles/descriptions and the README
provides alternative text and captions. Keep these visuals self-contained:
no external fonts, remote images, scripts or generated UI mockups.
