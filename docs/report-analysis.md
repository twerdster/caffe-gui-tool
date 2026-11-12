# What the original work tried to achieve

The relevant report is Aaron Wetzler's 2017 thesis, *Topics in Computational
Geometric Vision*, Chapter 4, especially §4.5.1–4.5.3 and Figure 4.17. The
implementation discussion on printed pages 63–64 directly describes this
Blender tool and cites Caffe-GUI-Tool.

The scientific idea was to turn explicit geometric denoising iterations into
an unrolled trainable computation graph. Derivative filters produce image
features; metric terms combine them; the inverse determinant controls flux;
divergence produces a residual flow update. Repeating the stage gives a
learned iterative denoiser while retaining its geometric structure. Caffe
provides automatic differentiation instead of requiring a new hand-derived
gradient for every architecture change.

The GUI serves experimentation: seeing the mathematical branches, reconnecting
them, adding or removing stages, changing filter channels, and choosing which
parameters share weights. Preserving those editable settings, named output
blobs, shared names and graph layout is central to modernization.

The thesis's final configuration uses five stages, 7×7 kernels, 80 feature
channels, tied metric weights within a stage and untied parameters between
stages. Tying the metric parameters protects the intended positive-semidefinite
structure. Arbitrary edits can destroy that property: export validation cannot
prove the resulting learned operator is numerically stable.

This repository contains intermediate experiments and a tiny 26-step
checkpoint. They are not the final thesis checkpoints. The modern composer's
six recovered graphs also include intermediate variants. The compact graph is
illustrative and untrained. Their export and CPU execution tests must be
distinguished from reproducing a trained denoising benchmark.

The separate [Deep Patch-Beltrami](https://github.com/twerdster/deep-patch-beltrami)
project preserves the final models and the BSD68 reproduction protocol. Its
checkpoint behavior takes precedence over differences in the prose, including
padding and saved step counts. This GUI update does not alter those models,
fit parameters against test images, or launch new training.

## Modernization decisions

- Reuse the existing modern composer while retaining historical Blender node
  identifiers and compatibility entry points.
- Preserve the original `.blend`, MATLAB, image and checkpoint research files.
- Keep JSON data-only and reject malformed or lossy imports transactionally.
- Validate phase dependencies and export without changing node parameters.
- Protect an existing export if a later export fails.
- Exercise real exported standard-layer networks on diverse data, comparing
  numerical outputs to independent reference calculations.
- Report custom-layer and training limitations explicitly; do not silently
  replace the research filter or clamp non-finite arithmetic into plausible output.
