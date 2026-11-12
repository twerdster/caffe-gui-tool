"""Strict graph interchange and live-edit regressions; run in isolated Blender."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import neural_operator_blender as addon
from neural_operator_blender import composer, graph

addon.register()
results = {'blender': bpy.app.version_string, 'suite': 'graph_regressions', 'checks': [], 'rejected_documents': 0}
base = json.loads((composer.EXAMPLES / 'operator_demo.json').read_text())


def rejected(document, message=None, **kwargs):
    before = set(bpy.data.node_groups.keys())
    try:
        graph.from_dict(document, **kwargs)
    except ValueError as error:
        if message is not None:
            assert message in str(error), str(error)
    else:
        raise AssertionError('Malformed graph document accepted: ' + repr(document)[:200])
    assert set(bpy.data.node_groups.keys()) == before, 'Failed import left partial node groups'
    results['rejected_documents'] += 1


def modified(path, value):
    document = copy.deepcopy(base)
    target = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    return document


for path in sorted(composer.EXAMPLES.glob('*.json')):
    document = json.loads(path.read_text())
    tree = graph.from_dict(document)
    before = graph.to_dict(tree)
    before['name'] = document['name']
    assert before == document, path.name
    graph.validate(tree)
    after = graph.to_dict(tree)
    after['name'] = document['name']
    assert after == document, 'Validation mutated ' + path.name
    bpy.data.node_groups.remove(tree)
results['checks'].append('all six recovered research graphs and demo roundtrip exactly; validation does not mutate them')

for value in [None, [], 1, True, 'graph', {}]:
    rejected(value)
for field in ['schema', 'name', 'nodes', 'links']:
    document = copy.deepcopy(base)
    del document[field]
    rejected(document)
rejected(dict(base, typo=True))
for field, values in {
    'name': ['', None, 2, 'x\x00y', 'é' * 128],
    'nodes': [None, {}, 'nodes', True],
    'links': [None, {}, 'links', False],
}.items():
    for value in values:
        rejected(modified([field], value))
rejected(dict(base, nodes=[{}] * (graph.MAX_NODES + 1)))
rejected(dict(base, links=[{}] * (graph.MAX_LINKS + 1)))
for value in [None, [], 123, 'node']:
    rejected(modified(['nodes', 0], value))
for field in ['id', 'type', 'location', 'outputs']:
    document = copy.deepcopy(base)
    del document['nodes'][0][field]
    rejected(document)
for field, values in {
    'id': ['', None, True, ['name'], 'x' * 64, 'é' * 32, 'bad\x00name', '\ud800'],
    'type': [[], {}, None, 'ShaderNodeTexNoise'],
    'location': [None, [1], [1, 2, 3], [True, 2], [float('nan'), 0], [0, float('inf')]],
    'color': [[0, 0], [1.01, 0, 0], [-0.01, 0, 0], [0, True, 0], [0, 0, float('nan')]],
    'width': [None, '180', True, -1, 1, float('nan'), float('inf')],
    'label': [None, 123, 'é' * 32, 'a\x00b'],
    'hide': [1, 'False', None],
    'use_custom_color': [0, 'True', None],
    'parent': [[], {}, 1, '', 'missing', base['nodes'][1]['id']],
    'outputs': [None, {}, ['valid', 1], [], ['extra'] * (graph.MAX_SOCKETS + 2)],
    'properties': [None, [], 1, {'typo': 1}],
}.items():
    for value in values:
        rejected(modified(['nodes', 0, field], value))
rejected(modified(['nodes', 0, 'id'], base['nodes'][1]['id']), 'Duplicate')
rejected(modified(['nodes', 0, 'typo'], 3), 'unknown')

# Explicit typing prevents Blender from silently coercing or clamping values.
data_index = next(i for i, n in enumerate(base['nodes']) if n['type'] == 'DataNodeType')
conv_index = next(i for i, n in enumerate(base['nodes']) if n['type'] == 'ConvNodeType')
elt_index = next(i for i, n in enumerate(base['nodes']) if n['type'] == 'EltwiseNodeType')
for field, values in {
    'train_batch_size': [0, -1, 1.5, True, '2', {}, []],
    'mirror': [0, 1, 'False', None],
    'scale': [True, '1.0', float('nan'), float('inf'), -1, 1e100],
    'output_amount': [graph.MAX_SOCKETS + 1, 2 ** 32, 2.0],
    'db_type': ['TypoBackend', [], None, 1],
    'train_data': [None, [], 1, '\ud800', 'a\x00b'],
    'weight_params': [None, [], 1, {'unknown': 1}, {'lr_mult': float('nan')}, {'name': []}],
}.items():
    for value in values:
        rejected(modified(['nodes', data_index, 'properties', field], value))
for value in [None, [], 2, {'std': float('inf')}, {'type': 'not-a-filler'}]:
    rejected(modified(['nodes', conv_index, 'properties', 'weight_filler'], value))
for value in [None, {}, [2], [{'unknown': 1}], [{'coeff': float('nan')}], [{}] * (graph.MAX_SOCKETS + 1)]:
    rejected(modified(['nodes', elt_index, 'properties', 'coeffs'], value))
# Malformed data late in the document must also roll back previously created nodes.
rejected(modified(['nodes', len(base['nodes']) - 1, 'properties'], {'unknown': 1}))
for value in [None, [], 1, {}]:
    rejected(modified(['links', 0], value))
for field in ['from', 'to', 'input', 'output']:
    document = copy.deepcopy(base)
    del document['links'][0][field]
    rejected(document)
for field in ['from', 'to']:
    for value in [None, [], {}, 1, 'missing']:
        rejected(modified(['links', 0, field], value))
for field in ['input', 'output']:
    for value in [-1, 1.5, True, None, '0', 99999]:
        rejected(modified(['links', 0, field], value))
rejected(modified(['links', 0, 'typo'], 1))
for value in ['', None, 1, 'é' * 128]:
    if value is None:
        continue  # None explicitly means use the document's name.
    rejected(base, name=value)
with patch.object(graph, 'MAX_TOTAL_SOCKETS', 1):
    rejected(base, 'total socket count')
results['checks'].append('malformed documents, typed and bounded RNA properties, UTF-8/name limits and links fail transactionally')

# Frames may nest, but a parent must exist, be a frame and be acyclic.
frame = {'id': 'Frame A', 'type': 'NodeFrame', 'label': '', 'location': [1, 2], 'width': 200, 'hide': False,
         'color': [0.3, 0.3, 0.3], 'use_custom_color': False, 'parent': None, 'properties': {}, 'outputs': []}
document = copy.deepcopy(base)
document['nodes'].append(copy.deepcopy(frame))
document['nodes'][0]['parent'] = 'Frame A'
tree = graph.from_dict(document)
assert tree.nodes[document['nodes'][0]['id']].parent == tree.nodes['Frame A']
bpy.data.node_groups.remove(tree)
document['nodes'][-1]['parent'] = 'Frame A'
rejected(document, 'Cycle')
document['nodes'][-1]['parent'] = 'Frame B'
document['nodes'].append(dict(frame, id='Frame B', parent='Frame A'))
rejected(document, 'Cycle')
# Roundtrip an actual nested-frame layout, including child-relative coordinates.
framed = bpy.data.node_groups.new('Nested frame layout', 'CaffeNodeTree')
child = framed.nodes.new('DataNodeType')
inner, outer = framed.nodes.new('NodeFrame'), framed.nodes.new('NodeFrame')
inner.parent = outer
child.parent = inner
outer.location, inner.location, child.location = (500, -200), (80, 120), (30, -40)
outer.label, inner.label = 'Outer', 'Inner'
expected = graph.to_dict(framed)
restored = graph.from_dict(expected)
actual = graph.to_dict(restored)
actual['name'] = expected['name']
assert actual == expected, 'Nested frame layout changed during roundtrip'
bpy.data.node_groups.remove(restored)
bpy.data.node_groups.remove(framed)
results['checks'].append('nested frame layouts roundtrip; missing/non-frame parents and parent cycles rejected')

with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / 'gráph.json'
    for text in ['{"schema":"first","schema":"second"}', '{"a":{"b":1,"b":2}}',
                 '{"number":NaN}', '{"number":Infinity}', '{"number":-Infinity}', '[' * 2000 + '0' + ']' * 2000]:
        path.write_text(text)
        before = set(bpy.data.node_groups.keys())
        try:
            graph.load(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed JSON accepted: ' + text[:80])
        assert set(bpy.data.node_groups.keys()) == before
    path.write_bytes(b'\xff\xfe')
    try:
        graph.load(path)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid UTF-8 accepted')
    path.write_text(json.dumps(base))
    with patch.object(graph, 'MAX_FILE_BYTES', 16):
        try:
            graph.load(path)
        except ValueError as error:
            assert 'file size limit' in str(error)
        else:
            raise AssertionError('Oversized JSON accepted')
    tree = graph.from_dict(base)
    graph.save(tree, path)
    before = path.read_bytes()
    with patch.object(graph.os, 'replace', side_effect=OSError('simulated interrupted replacement')):
        try:
            graph.save(tree, path)
        except OSError:
            pass
        else:
            raise AssertionError('Simulated write failure was swallowed')
    assert path.read_bytes() == before and list(Path(temporary).iterdir()) == [path]
    restored = graph.load(path)
    expected, actual = graph.to_dict(tree), graph.to_dict(restored)
    actual['name'] = expected['name']
    assert actual == expected
    bpy.data.node_groups.remove(restored)
    power = tree.nodes.new('PowerNodeType')
    power.power = float('nan')
    before = path.read_bytes()
    try:
        graph.save(tree, path)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonstandard JSON NaN emitted')
    assert path.read_bytes() == before, 'Failed save replaced existing JSON'
    bpy.data.node_groups.remove(tree)
results['checks'].append('duplicate JSON keys, NaN/Infinity, invalid Unicode, excessive nesting and oversized files rejected; valid Unicode file paths roundtrip and interrupted saves preserve the prior graph')

# Editing legacy empty collections must not desynchronize sockets or call remove(-1).
tree = bpy.data.node_groups.new('Dynamic edit regressions', 'CaffeNodeTree')
data = tree.nodes.new('DataNodeType')
data.output_amount = 8
assert len(data.outputs) == 8
data.output_amount = 1
assert len(data.outputs) == 1
concat = tree.nodes.new('ConcatNodeType')
concat.input_amount = 8
assert len(concat.inputs) == 8
concat.input_amount = 1
assert len(concat.inputs) == 1
eltwise = tree.nodes.new('EltwiseNodeType')
eltwise.coeffs.clear()
eltwise.input_amount = 1
assert len(eltwise.inputs) == len(eltwise.coeffs) == 1
eltwise.input_amount = 5
assert len(eltwise.inputs) == len(eltwise.coeffs) == 5
eltwise.coeffs.clear()
eltwise.input_amount = 3
assert len(eltwise.inputs) == len(eltwise.coeffs) == 3
slice_node = tree.nodes.new('SliceNodeType')
slice_node.slice_points.clear()
slice_node.num_of_slices = 4
assert len(slice_node.slice_points) == 4 and len(slice_node.outputs) == 5
assert len({socket.name for socket in slice_node.outputs}) == 5
slice_node.num_of_slices = 1
assert len(slice_node.slice_points) == 1 and len(slice_node.outputs) == 2
for node, field in [(data, 'output_amount'), (concat, 'input_amount'), (eltwise, 'input_amount'), (slice_node, 'num_of_slices')]:
    assert node.bl_rna.properties[field].hard_max == graph.MAX_SOCKETS
bpy.data.node_groups.remove(tree)
for cls in addon.caffe_nodes.CLASSES:
    for prop in cls.bl_rna.properties:
        if prop.subtype in {'FILE_PATH', 'DIR_PATH'}:
            assert prop.is_path_supports_blend_relative, cls.__name__ + '/' + prop.identifier
results['checks'].append('dynamic socket edits remain synchronized and bounded; RNA path fields support Blender relative paths')

# Kahn traversal handles fan-out, repeated dependencies and a longer chain deterministically.
tree = bpy.data.node_groups.new('Topology regression', 'CaffeNodeTree')
data = tree.nodes.new('DataNodeType')
data.outputs[0].output_name, data.outputs[1].output_name = 'input', 'label'
last = data.outputs[0]
for index in range(300):
    node = tree.nodes.new('PowerNodeType')
    node.name = 'Power %03d' % index
    node.outputs[0].output_name = 'blob_%03d' % index
    tree.links.new(last, node.inputs[0])
    last = node.outputs[0]
eltwise = tree.nodes.new('EltwiseNodeType')
eltwise.outputs[0].output_name = 'sum'
for socket in eltwise.inputs:
    tree.links.new(last, socket)
order = [node.name for node in graph.topological(tree)]
assert order == [node.name for node in graph.topological(tree)]
positions = {name: index for index, name in enumerate(order)}
assert all(positions[link.from_node.name] < positions[link.to_node.name] for link in tree.links)
assert graph.validate(tree)['nodes'] == 302
node.power = float('nan')
try:
    graph.validate(tree)
except ValueError as error:
    assert 'finite' in str(error)
else:
    raise AssertionError('Live nonfinite RNA property accepted')
node.power = 1.0
tree.links.new(last, tree.nodes['Power 000'].inputs[0])
try:
    graph.validate(tree)
except ValueError as error:
    assert 'Cycle' in str(error)
else:
    raise AssertionError('Long graph cycle accepted')
bpy.data.node_groups.remove(tree)
empty = bpy.data.node_groups.new('Empty regression', 'CaffeNodeTree')
try:
    graph.validate(empty)
except ValueError as error:
    assert 'no neural' in str(error)
else:
    raise AssertionError('Empty graph validated as exportable')
bpy.data.node_groups.remove(empty)
multiple = bpy.data.node_groups.new('Multiple live links', 'CaffeNodeTree')
source = multiple.nodes.new('DataNodeType')
source.outputs[0].output_name, source.outputs[1].output_name = 'a', 'b'
sink = multiple.nodes.new('PowerNodeType')
sink.outputs[0].output_name = 'output'
for socket in source.outputs:
    multiple.links.new(socket, sink.inputs[0], verify_limits=False)
assert len(sink.inputs[0].links) == 2
try:
    graph.validate(multiple)
except ValueError as error:
    assert 'Multiple links' in str(error) or 'Invalid' in str(error)
else:
    raise AssertionError('Multiple live input links accepted')
bpy.data.node_groups.remove(multiple)
results['checks'].append('deterministic dependency ordering handles fan-out and 300-layer chain; cycles, live nonfinite values, multiple live input links and empty graphs rejected')

addon.unregister()
print('VALIDATION_JSON=' + json.dumps(results, sort_keys=True))
