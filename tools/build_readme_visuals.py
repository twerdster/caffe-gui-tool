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
    text(36, 40, 'CAFFE GUI TOOL  /  THE WORKFLOW', 13, '#527568', 700, 'letter-spacing="2"'),
    text(36, 89, 'Make the computation visible.', 36, '#142d31', 700),
    text(36, 120, 'Change a connection, a filter or a whole stage. Keep the mathematical structure in sight.', 18, '#526a6d')]
cards = [(36, '#e9e3f8', '01', 'Compose', 'Connect the pieces of your experiment.'),
         (424, '#def1e7', '02', 'Check', 'Catch broken graphs before exporting.'),
         (812, '#e1ecf7', '03', 'Export', 'Take the graph into your next step.')]
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
for y, value in [(271, 'Check connections'), (306, 'Preserve shared weights'), (341, 'Keep the original graph')]:
    workflow += [f'<circle cx="455" cy="{y-5}" r="9" fill="#e0f1e7"/>',
                 f'<path d="M450 {y-5} l4 4 6 -8" fill="none" stroke="#267252" stroke-width="2"/>',
                 text(474, y, value, 17, '#34544b')]
for y, value in [(271, 'graph.json'), (306, '*_deploy.prototxt'), (341, '*_solver.prototxt')]:
    workflow += [rect(832, y-22, 312, 31, '#f0f5fa', 7), text(845, y, value, 15, '#355976', extra='class="mono"')]
workflow += [text(36, 429, 'THE IDEA INSIDE A BELTRAMI STAGE', 12, '#527568', 700, 'letter-spacing="1.8"')]
steps = [('Image', 100, '#dcecf8'), ('Derivatives', 166, '#e9e3f8'), ('Metric', 126, '#def1e7'),
         ('Flux', 100, '#f8e9d4'), ('Divergence', 164, '#e9e3f8'), ('Update', 128, '#dcecf8')]
x = 36
for index, (label, width, color) in enumerate(steps):
    workflow += [rect(x, 447, width, 42, color, 10), text(x+width/2, 474, label, 17, '#294445', 600, 'text-anchor="middle"')]
    x += width
    if index != len(steps)-1:
        workflow.append(arrow(x+8, 468, x+39, 468))
        x += 51
workflow += [text(36, 526, 'An explanatory schematic. Training runs separately from the Blender editor.', 15, '#526a6d')]
(IMAGES / 'workflow.svg').write_text(svg('Compose, check and export a geometric neural network',
    'Three steps: compose an editable node graph, check its connections and export JSON and Caffe definitions. '
    'A Beltrami stage follows image, derivatives, metric, flux, divergence and update. This is a schematic, not a trained result.',
    552, workflow), encoding='utf-8')

record = json.loads((ROOT / 'docs/validation/macos-blender-5.2.json').read_text())
runtime = record['runtime']
node_types = record['suites']['blender_test']['results']['node_types']
validation = [rect(0, 0, 1200, 357, '#101c2a', 24),
    text(36, 42, 'TESTED BEYOND THE CANVAS', 13, '#9bd6bc', 700, 'letter-spacing="2"'),
    text(36, 87, 'Real exports. Independent checks.', 34, '#f0f6fa', 700),
    text(36, 117, 'Blender 5.1.2 + 5.2.4 LTS  ·  macOS + Linux CI', 17, '#aab9c9')]
metrics = [(str(record['runtime_pytest_passed']), 'CPU tests per run', 'Mathematics + failure cases'),
           (f"{runtime['public_image_cases']} + {runtime['synthetic_cases']}", 'images + synthetic cases', 'Photographs, patterns and edge cases'),
           (str(runtime['repeat_identical']), 'identical repeat checks', 'Same input. Same prediction.'),
           (str(node_types), 'editable node types', 'Plus all 5 original Blender files')]
for index, (value, label, detail) in enumerate(metrics):
    x = 36 + 286*index
    validation += [rect(x, 146, 270, 139, '#18283b', 15, '#2c3d50'),
                   text(x+18, 201, value, 43 if len(value)>5 else 47, '#c1f0d8', 700),
                   text(x+18, 233, label, 17, '#f0f6fa', 600),
                   text(x+18, 260, detail, 13, '#aab9c9')]
validation += [text(36, 324, 'Operator tests use chosen coefficients. These are not denoising-quality benchmarks.', 16, '#aab9c9')]
(IMAGES / 'validation.svg').write_text(svg('Measured validation coverage',
    'Per supported-version run: 527 CPU tests; 487 public images and 23 synthetic cases; '
    '510 exact repeat checks; 29 editable node types and five original Blender files. '
    'Blender 5.1.2 and 5.2.4 passed on macOS and Linux CI. This does not measure denoising quality.',
    357, validation), encoding='utf-8')
print('Generated workflow.svg and validation.svg')
