"""Run with blender --background --factory-startup --python-exit-code 1 --python tests/blender_test.py."""
import ast
import copy
import re
import argparse
import subprocess
import json
from pathlib import Path
import sys
import tempfile
import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import neural_operator_blender as addon
from neural_operator_blender import graph, composer

parser = argparse.ArgumentParser()
parser.add_argument('output', nargs='?')
parser.add_argument('--caffe-proto', type=Path)
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])

def check_prototxt(directory):
    if not args.caffe_proto:
        return
    for file in directory.glob('*.prototxt'):
        message = 'caffe.SolverParameter' if '_solver.' in file.name else 'caffe.NetParameter'
        subprocess.run(['protoc', '--proto_path=' + str(args.caffe_proto.parent), '--encode=' + message, args.caffe_proto.name], input=file.read_bytes(), stdout=subprocess.DEVNULL, check=True)

addon.register()
results = {'blender': bpy.app.version_string, 'node_types': 0, 'examples': [], 'checks': []}
tree = bpy.data.node_groups.new('All Node Types', 'CaffeNodeTree')
for cls in addon.caffe_nodes.CLASSES:
    if issubclass(cls, bpy.types.Node):
        node = tree.nodes.new(cls.bl_idname)
        for key in graph.property_names(node):
            assert not type(getattr(node, key)).__name__.startswith('_PropertyDeferred'), key
        results['node_types'] += 1
conv = next(n for n in tree.nodes if n.bl_idname == 'ConvNodeType')
assert conv.num_output == 20 and conv.weight_params.lr_mult == 1
conv.num_output = 96
assert conv.num_output == 96
results['checks'].append('editable typed RNA properties and inherited shared-weight settings')
with tempfile.TemporaryDirectory() as temp:
    for file in sorted(composer.EXAMPLES.glob('*.json')):
        document = json.loads(file.read_text())
        tree = graph.from_dict(document)
        assert tree.use_fake_user, 'Imported graphs must survive .blend saves without editor users'
        result = graph.to_dict(tree)
        result['name'] = document['name']
        assert result == document, file.name
        summary = graph.validate(tree)
        before = graph.to_dict(tree)
        directory = Path(temp) / file.stem
        composer.export_caffe(tree, directory)
        assert graph.to_dict(tree) == before, 'Export changed graph'
        exported = list(directory.glob('*.prototxt'))
        assert len(exported) == 3
        check_prototxt(directory)
        deploy = next(p for p in exported if '_deploy.' in p.name).read_text()
        assert 'input_shape' in deploy and 'Convolution' in deploy
        assert deploy.count('{') == deploy.count('}')
        assert not list(directory.glob('*.caffemodel')), 'Export must not run training'
        results['examples'].append({'name': file.name, 'nodes': len(tree.nodes), 'links': len(tree.links), 'caffe_exported': True})
        bpy.data.node_groups.remove(tree)
results['checks'].append('all recovered graphs roundtrip with exact properties, layouts, sockets and links')
results['checks'].append('Caffe train/test/deploy/solver and quoted shell-script export; graph unchanged')

bad = bpy.data.node_groups.new('Invalid', 'CaffeNodeTree')
a, b = bad.nodes.new('PowerNodeType'), bad.nodes.new('PowerNodeType')
a.outputs[0].output_name, b.outputs[0].output_name = 'a', 'b'
bad.links.new(a.outputs[0], b.inputs[0])
bad.links.new(b.outputs[0], a.inputs[0])
try:
    graph.validate(bad)
except ValueError as error:
    assert 'Cycle' in str(error)
else:
    raise AssertionError('Cyclic graph accepted')
results['checks'].append('cycle rejected')

bad.links.clear()
bad.links.new(a.outputs[0], b.inputs[0])
try:
    graph.validate(bad)
except ValueError as error:
    assert 'Unconnected' in str(error)
else:
    raise AssertionError('Incomplete graph accepted')
results['checks'].append('unconnected input rejected')

count = len(bpy.data.node_groups)
try:
    graph.from_dict({'schema': graph.SCHEMA, 'name':'Bad import', 'nodes':[{'id':'bad','type':'ShaderNodeTexNoise'}], 'links':[]})
except ValueError:
    pass
else:
    raise AssertionError('Unknown node type accepted')
assert len(bpy.data.node_groups) == count
results['checks'].append('unknown node import rejected without partial data')

with tempfile.TemporaryDirectory() as temp:
    for cls in addon.caffe_nodes.CLASSES:
        if not issubclass(cls, bpy.types.Node) or cls.bl_idname in {'DataNodeType', 'SolverNodeType'}:
            continue
        tree = bpy.data.node_groups.new(cls.__name__, 'CaffeNodeTree')
        data = tree.nodes.new('DataNodeType')
        data.outputs[0].output_name, data.outputs[1].output_name = 'image', 'label'
        solver = tree.nodes.new('SolverNodeType')
        solver.solvername, solver.solver_mode = 'node_test', 'CPU'
        solver.base_lr = 1e-8
        node = tree.nodes.new(cls.bl_idname)
        if cls.bl_idname == 'SliceNodeType':
            node.slice_points[0].slice_point = 1
        for i, socket in enumerate(node.inputs):
            tree.links.new(data.outputs[i % 2], socket)
        for i, socket in enumerate(node.outputs):
            socket.output_name = 'output_' + str(i)
        before = graph.to_dict(tree)
        directory = Path(temp) / cls.__name__
        composer.export_caffe(tree, directory)
        check_prototxt(directory)
        assert graph.to_dict(tree) == before
        solver_text = (directory / 'node_test_solver.prototxt').read_text()
        assert 'base_lr: 9.99999994e-09' in solver_text
        bpy.data.node_groups.remove(tree)
results['checks'].append('all 27 processing/loss nodes export independently; tiny learning rate preserved')
if args.caffe_proto:
    results['checks'].append('every generated prototxt parsed by protoc against original Caffe fork schema')


# Edited graphs must declare every data blob used by the deployment network.
with tempfile.TemporaryDirectory() as temp:
    tree = graph.load(composer.EXAMPLES / 'operator_demo.json')
    data = next(node for node in tree.nodes if node.bl_idname == 'DataNodeType')
    secondary = tree.nodes.new('PowerNodeType')
    secondary.outputs[0].output_name = 'processed_secondary'
    tree.links.new(data.outputs[1], secondary.inputs[0])
    directory = Path(temp) / 'secondary'
    before = graph.to_dict(tree)
    composer.export_caffe(tree, directory)
    check_prototxt(directory)
    deploy = next(directory.glob('*_deploy.prototxt')).read_text()
    declared = set(re.findall(r'^input: "([^"\n]+)"', deploy, flags=re.MULTILINE))
    assert data.outputs[1].output_name in declared
    available = declared | set(re.findall(r'\btop: "([^"\n]+)"', deploy))
    assert set(re.findall(r'\bbottom: "([^"\n]+)"', deploy)) <= available
    assert graph.to_dict(tree) == before
    solver = next(node for node in tree.nodes if node.bl_idname == 'SolverNodeType')
    solver.lr_policy = 'multistep'
    before = graph.to_dict(tree)
    unsupported = Path(temp) / 'unsupported'
    try:
        composer.export_caffe(tree, unsupported)
    except ValueError as error:
        assert 'multistep' in str(error)
    else:
        raise AssertionError('Unsupported multistep policy exported')
    assert not unsupported.exists() and graph.to_dict(tree) == before
    bpy.data.node_groups.remove(tree)
results['checks'].append('secondary deployment data outputs are declared; unsupported schedules fail without writes')

# Legitimate quotes/backslashes in paths and shared names must retain their values.
with tempfile.TemporaryDirectory() as temp:
    quoted = '/tmp/user "quoted" \\ dataset/file.txt'
    for backend in ['HDF5Data', 'ImageData', 'LMDB']:
        tree = graph.load(composer.EXAMPLES / 'operator_demo.json')
        data = next(node for node in tree.nodes if node.bl_idname == 'DataNodeType')
        data.db_type = backend
        data.train_data = data.test_data = data.train_path = data.test_path = quoted
        data.use_mean_file = True
        data.mean_file = quoted
        solver = next(node for node in tree.nodes if node.bl_idname == 'SolverNodeType')
        solver.snapshot_prefix = quoted
        conv = next(node for node in tree.nodes if node.bl_idname == 'ConvNodeType')
        conv.extra_params = True
        conv.weight_params.name = 'shared "quoted" \\ kernel'
        hdf5 = tree.nodes.new('HDF5OutputNodeType')
        hdf5.filename = quoted
        for index, socket in enumerate(hdf5.inputs):
            tree.links.new(data.outputs[index % len(data.outputs)], socket)
        directory = Path(temp) / ('export "quoted" \\ ' + backend)
        before = graph.to_dict(tree)
        composer.export_caffe(tree, directory)
        check_prototxt(directory)
        if args.caffe_proto:
            # Decode through the external schema rather than reusing exporter escaping.
            def fields(file, field, message):
                command = ['protoc', '--proto_path=' + str(args.caffe_proto.parent)]
                encoded = subprocess.run(command + ['--encode=' + message, args.caffe_proto.name], input=file.read_bytes(), stdout=subprocess.PIPE, check=True).stdout
                decoded = subprocess.run(command + ['--decode=' + message, args.caffe_proto.name], input=encoded, stdout=subprocess.PIPE, check=True).stdout.decode()
                return [ast.literal_eval(value) for value in re.findall(r'\b' + field + r': ("(?:\\.|[^"\\])*")', decoded)]
            train = next(directory.glob('*_train_test.prototxt'))
            solver_file = next(directory.glob('*_solver.prototxt'))
            assert bpy.path.abspath(quoted) in fields(train, 'source', 'caffe.NetParameter')
            assert quoted in fields(train, 'file_name', 'caffe.NetParameter')
            assert conv.weight_params.name in fields(train, 'name', 'caffe.NetParameter')
            assert str(next(directory.glob('*_train_test.prototxt')).resolve()) in fields(solver_file, 'net', 'caffe.SolverParameter')
            assert bpy.path.abspath(quoted) in fields(solver_file, 'snapshot_prefix', 'caffe.SolverParameter')
        assert graph.to_dict(tree) == before
        bpy.data.node_groups.remove(tree)
results['checks'].append('quoted paths and shared names export with correct protobuf string values')

# An invalid interchange document must not silently discard or replace graph data.
base = json.loads((composer.EXAMPLES / 'operator_demo.json').read_text())
def invalid_document(document):
    count = len(bpy.data.node_groups)
    try:
        graph.from_dict(document)
    except ValueError:
        pass
    else:
        raise AssertionError('Malformed graph document accepted')
    assert len(bpy.data.node_groups) == count

mutations = []
document = copy.deepcopy(base)
document['links'].append(copy.deepcopy(document['links'][0]))
mutations.append(document)
document = copy.deepcopy(base)
other = next(link for link in document['links'] if link['from'] != document['links'][0]['from'])
document['links'].append(dict(document['links'][0], **{'from':other['from'], 'output':other['output']}))
mutations.append(document)
for key, value in [('output', -1), ('input', -1), ('output', 0.5), ('input', True), ('output', 9999), ('input', 9999), ('from', 'missing'), ('to', 'missing')]:
    document = copy.deepcopy(base)
    document['links'][0][key] = value
    mutations.append(document)
for difference in [-1, 1]:
    document = copy.deepcopy(base)
    names = document['nodes'][0]['outputs']
    document['nodes'][0]['outputs'] = names[:-1] if difference < 0 else names + ['extra']
    mutations.append(document)
for document in mutations:
    invalid_document(document)
results['checks'].append('duplicate links, invalid socket indices, missing node references and output-count mismatches rejected without partial imports')

addon.unregister()
addon.register()
addon.unregister()
results['checks'].append('clean disable and re-enable')
print('VALIDATION_JSON=' + json.dumps(results, sort_keys=True))
if args.output:
    Path(args.output).write_text(json.dumps(results, indent=2) + '\n')
