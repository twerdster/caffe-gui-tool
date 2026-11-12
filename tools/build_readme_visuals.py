"""Generate accessible README schematics from the recorded validation results."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / 'docs/images'
IMAGES.mkdir(exist_ok=True)


def text(x, y, value, size=18, fill='#233b43', weight=400, extra=''):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" {extra}>{escape(str(value))}</text>'


def rect(x, y, w, h, fill, radius=16, stroke='none'):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def svg(title, description, height, elements):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" '
            f'viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">\n'
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>\n'
            '<style>text{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}'
            '.mono{font-family:"SFMono-Regular",Consolas,monospace}</style>\n'
            + '\n'.join(elements) + '\n</svg>\n')


def arrow(x1, y1, x2, y2, color='#64817c'):
    return (f'<path d="M{x1} {y1} H{x2}" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<path d="M{x2-6} {y2-5} L{x2} {y2} L{x2-6} {y2+5}" fill="none" stroke="{color}" stroke-width="2"/>')


workflow = [rect(0, 0, 1200, 552, '#f3f7f5', 24),
    text(36, 40, 'CAFFE GUI TOOL  /  GRAPH EXPORT', 13, '#527568', 700, 'letter-spacing="2"'),
    text(36, 89, 'Editing and exporting a Caffe network', 36, '#142d31', 700),
    text(36, 120, 'Set layer parameters in Blender, validate the graph, and write Caffe definitions.', 18, '#526a6d')]
cards = [(36, '#e9e3f8', '01', 'Edit graph', 'Configure layers, links and shared parameters.'),
         (424, '#def1e7', '02', 'Validate', 'Structural checks; phase checks on export.'),
         (812, '#e1ecf7', '03', 'Export', 'Write portable graphs and Caffe definitions.')]
for x, color, number, title, subtitle in cards:
    workflow += [rect(x, 151, 352, 241, '#ffffff', 18, '#dbe5e1'),
                 rect(x+20, 170, 38, 30, color, 9), text(x+29, 191, number, 14, '#294445', 700),
                 text(x+72, 194, title, 24, '#173339', 700), text(x+20, 225, subtitle, 15, '#526a6d')]
# Illustrative node branch, deliberately distinct from the actual UI screenshot.
for x, y, w, title, color in [(57, 274, 83, 'Image', '#dcecf8'), (180, 255, 120, 'Derivative', '#e9e3f8'),
                              (180, 325, 120, 'Metric', '#def1e7')]:
    workflow += [rect(x, y, w, 33, color, 8), text(x+12, y+22, title, 14, '#294445', 600)]
workflow += ['<path d="M140 290 C163 290 157 271 180 271 M240 288 V325" '
             'fill="none" stroke="#85a49c" stroke-width="2"/>']
for y, value in [(271, 'Required inputs'), (306, 'Unique output names'), (341, 'Acyclic connections')]:
    workflow += [f'<circle cx="455" cy="{y-5}" r="9" fill="#e0f1e7"/>',
                 f'<path d="M450 {y-5} l4 4 6 -8" fill="none" stroke="#267252" stroke-width="2"/>',
                 text(474, y, value, 17, '#34544b')]
for y, value in [(271, 'graph.json'), (306, '*_deploy.prototxt'), (341, '*_solver.prototxt')]:
    workflow += [rect(832, y-22, 312, 31, '#f0f5fa', 7), text(845, y, value, 15, '#355976', extra='class="mono"')]
workflow += [text(36, 429, 'ONE BELTRAMI-STYLE UPDATE', 12, '#527568', 700, 'letter-spacing="1.8"')]
steps = [('Image', 100, '#dcecf8'), ('Derivatives', 166, '#e9e3f8'), ('Metric', 126, '#def1e7'),
         ('Flux', 100, '#f8e9d4'), ('Divergence', 164, '#e9e3f8'), ('Update', 128, '#dcecf8')]
x = 36
for index, (label, width, color) in enumerate(steps):
    workflow += [rect(x, 447, width, 42, color, 10), text(x+width/2, 474, label, 17, '#294445', 600, 'text-anchor="middle"')]
    x += width
    if index != len(steps)-1:
        workflow.append(arrow(x+8, 468, x+39, 468))
        x += 51
workflow += [text(36, 526, 'Schematic only. No tensor-shape inference; training runs outside Blender.', 15, '#526a6d')]
(IMAGES / 'workflow.svg').write_text(svg('Editing and exporting a Caffe network',
    'Three steps: compose an editable node graph, check its connections and export JSON and Caffe definitions. '
    'A Beltrami-style update consists of image input, derivatives, metric, flux, divergence and update. '
    'This is a schematic, not a trained result. Tensor shapes are not inferred.',
    552, workflow), encoding='utf-8')

records = [json.loads((ROOT / f'docs/validation/macos-blender-{version}.json').read_text())
           for version in ('5.1', '5.2')]
record = records[-1]
runtime = record['runtime']
for result in records:
    assert result['status'] == 'passed'
    assert result['runtime_pytest_passed'] == record['runtime_pytest_passed']
    for key in ('public_image_cases', 'synthetic_cases', 'layer_comparisons', 'repeat_identical'):
        assert result['runtime'][key] == runtime[key], 'Validation records disagree: ' + key
inputs = runtime['public_image_cases'] + runtime['synthetic_cases']
guard_tests = record['runtime_pytest_passed'] - inputs
layer_outputs = runtime['layer_comparisons'] - runtime['public_image_cases']
validation = [rect(0, 0, 1200, 357, '#101c2a', 24),
    text(36, 42, 'VALIDATION', 13, '#9bd6bc', 700, 'letter-spacing="2"'),
    text(36, 87, 'CPU inference tests', 34, '#f0f6fa', 700),
    text(36, 117, 'OpenCV 4 CPU  ·  Blender-exported standard-layer chain  ·  deterministic weights', 17, '#aab9c9')]
metrics = [(str(record['runtime_pytest_passed']), 'CPU tests per run', f'{inputs} inputs; {guard_tests} guard/CLI/failure'),
           (f"{runtime['public_image_cases']} + {runtime['synthetic_cases']}", 'images + synthetic inputs', 'Separate NumPy reference'),
           (str(runtime['layer_comparisons']), 'NumPy comparisons', f"{layer_outputs} layer outputs + {runtime['public_image_cases']} final"),
           (str(runtime['repeat_identical']), 'exact repeat checks', 'Exact equality for all input cases')]
for index, (value, label, detail) in enumerate(metrics):
    x = 36 + 286*index
    validation += [rect(x, 146, 270, 139, '#18283b', 15, '#2c3d50'),
                   text(x+18, 201, value, 43 if len(value)>5 else 47, '#c1f0d8', 700),
                   text(x+18, 233, label, 17, '#f0f6fa', 600),
                   text(x+18, 260, detail, 13, '#aab9c9')]
validation += [text(36, 324, 'Standard-layer tests use deterministic weights; no training or denoising-quality evaluation.', 16, '#aab9c9')]
(IMAGES / 'validation.svg').write_text(svg('CPU inference test coverage',
    f"Per recorded OpenCV 4 CPU run: {record['runtime_pytest_passed']} tests; "
    f"{runtime['public_image_cases']} repository images and {runtime['synthetic_cases']} synthetic inputs; "
    f"{runtime['layer_comparisons']} NumPy comparisons and {runtime['repeat_identical']} exact repeat checks. "
    'This does not measure denoising quality or training.',
    357, validation), encoding='utf-8')
print('Generated workflow.svg and validation.svg')
