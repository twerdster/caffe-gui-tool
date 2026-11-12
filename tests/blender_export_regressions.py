"""Exercise export failures, repeatability and phase semantics in stock Blender.

Run with Blender --background --factory-startup --python-exit-code 1 --python
this_file.py -- [output.json] [--caffe-proto tests/schema/caffe.proto].
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import neural_operator_blender as addon
from neural_operator_blender import caffe_export, composer, graph

parser = argparse.ArgumentParser()
parser.add_argument('output', nargs='?', type=Path)
parser.add_argument('--caffe-proto', type=Path, default=ROOT / 'tests/schema/caffe.proto')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
results = {'blender': bpy.app.version_string, 'checks': []}


def make_graph(kind='PowerNodeType'):
    tree = bpy.data.node_groups.new('Export regression', 'CaffeNodeTree')
    data = tree.nodes.new('DataNodeType')
    data.outputs[0].output_name, data.outputs[1].output_name = 'image', 'target'
    solver = tree.nodes.new('SolverNodeType')
    solver.solvername, solver.solver_mode = 'regression', 'CPU'
    node = tree.nodes.new(kind)
    for index, socket in enumerate(node.inputs):
        tree.links.new(data.outputs[index % 2], socket)
    for index, socket in enumerate(node.outputs):
        socket.output_name = 'processed_' + str(index)
    if kind == 'SliceNodeType':
        node.slice_points[0].slice_point = 1
    return tree, data, solver, node


def contents(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def parse_exports(directory):
    if not args.caffe_proto or not shutil.which('protoc'):
        return
    schema = args.caffe_proto.resolve()
    for path in directory.glob('*.prototxt'):
        message = 'caffe.SolverParameter' if path.name.endswith('_solver.prototxt') else 'caffe.NetParameter'
        subprocess.run(['protoc', '--proto_path=' + str(schema.parent), '--encode=' + message, schema.name],
                       input=path.read_bytes(), stdout=subprocess.DEVNULL, check=True)


def reject(tree, directory, message):
    before = graph.to_dict(tree)
    previous = contents(directory) if directory.exists() else None
    try:
        composer.export_caffe(tree, directory)
    except ValueError as error:
        assert message.lower() in str(error).lower(), str(error)
    else:
        raise AssertionError('Invalid export accepted: ' + message)
    assert graph.to_dict(tree) == before, 'Failed export mutated graph'
    if previous is None:
        assert not directory.exists(), 'Invalid export created destination'
    else:
        assert contents(directory) == previous, 'Invalid export overwrote files'


addon.register()
with tempfile.TemporaryDirectory(prefix='noc-export-regressions-') as temp:
    temp = Path(temp)
    # Rendering must never use Blender properties as temporary state. This also
    # checks the formerly hidden Data.n_type mutation omitted by JSON snapshots.
    tree, data, solver, node = make_graph()
    tree.nodes.new('NodeFrame')
    directory = temp / 'quoted "export" \\ destination'
    directory.mkdir()
    (directory / 'unrelated.txt').write_text('preserve me', encoding='utf-8')
    for backend in ('HDF5Data', 'LMDB', 'LEVELDB', 'ImageData'):
        data.db_type = backend
        data.include_in = 'TEST'  # Legacy exporter scratch value, not a Data phase control.
        before = graph.to_dict(tree)
        had_type = hasattr(data, 'n_type')
        composer.export_caffe(tree, directory)
        first = contents(directory)
        composer.export_caffe(tree, directory)
        assert contents(directory) == first
        assert graph.to_dict(tree) == before and hasattr(data, 'n_type') == had_type
        train = (directory / 'regression_train_test.prototxt').read_text()
        assert train.count('phase: TRAIN') == 1 and train.count('phase: TEST') == 1
        assert (directory / 'unrelated.txt').read_text() == 'preserve me'
        assert os.access(directory / 'train_regression.sh', os.X_OK)
        subprocess.run(['sh', '-n', str(directory / 'train_regression.sh')], check=True)
        parse_exports(directory)
    results['checks'].append('four data backends render twice identically, with both phases, executable quoted scripts, and no graph/RNA mutation')

    # TRAIN layers are absent from deploy; TEST layers become unconditional.
    node.include_in = 'TRAIN'
    test_node = tree.nodes.new('PowerNodeType')
    test_node.include_in = 'TEST'
    test_node.outputs[0].output_name = 'test_only'
    tree.links.new(data.outputs[0], test_node.inputs[0])
    composer.export_caffe(tree, directory)
    deploy = (directory / 'regression_deploy.prototxt').read_text()
    assert 'name: ' + caffe_export.proto_string(node.name) not in deploy
    assert 'name: ' + caffe_export.proto_string(test_node.name) in deploy
    assert 'include {' not in deploy
    downstream = tree.nodes.new('PowerNodeType')
    downstream.outputs[0].output_name = 'downstream'
    tree.links.new(node.outputs[0], downstream.inputs[0])
    reject(tree, directory, 'unavailable in TEST')
    tree.nodes.remove(downstream)
    bpy.data.node_groups.remove(tree)
    tree, data, solver, loss = make_graph('EULossNodeType')
    consumer = tree.nodes.new('PowerNodeType')
    consumer.outputs[0].output_name = 'invalid_deployment'
    tree.links.new(loss.outputs[0], consumer.inputs[0])
    reject(tree, temp / 'excluded_dependency', 'excluded training/evaluation')
    bpy.data.node_groups.remove(tree)
    results['checks'].append('TRAIN/TEST layers and deployment exclusions respect actual dependencies; failed re-exports preserve previous files')

    # Cross-field constraints are stricter than Blender RNA's per-field bounds.
    invalid = [
        ('LRNNodeType', lambda n: setattr(n, 'size', 2), 'odd'),
        ('DropoutNodeType', lambda n: setattr(n, 'dropout_ratio', 1), 'dropout'),
        ('LogNodeType', lambda n: setattr(n, 'base', 1), 'base'),
        ('ExpNodeType', lambda n: setattr(n, 'base', 0), 'base'),
        ('ConvNodeType', lambda n: setattr(n, 'group', 3), 'divisible'),
        ('ConvNodeType', lambda n: (setattr(n.weight_filler, 'type', 'uniform'), setattr(n.weight_filler, 'min', 2)), 'minimum'),
        ('ConvNodeType', lambda n: (setattr(n.weight_filler, 'type', 'gaussian'), setattr(n.weight_filler, 'std', -1)), 'deviation'),
        ('ConvNodeType', lambda n: (setattr(n.weight_filler, 'type', 'gaussian'), setattr(n.weight_filler, 'std', 0)), 'deviation'),
        ('FCNodeType', lambda n: setattr(n.weight_filler, 'type', 'bilinear'), 'bilinear'),
        ('ConvNodeType', lambda n: (setattr(n, 'square_kernel', False), setattr(n, 'kernel_h', 3), setattr(n.weight_filler, 'type', 'bilinear')), 'bilinear'),
        ('SliceNodeType', lambda n: setattr(n.slice_points[0], 'slice_point', 0), 'slice points'),
        ('SliceNodeType', lambda n: (setattr(n, 'num_of_slices', 2), setattr(n.slice_points[0], 'slice_point', 2), setattr(n.slice_points[1], 'slice_point', 1), setattr(n.outputs[2], 'output_name', 'third')), 'slice points'),
        ('EltwiseNodeType', lambda n: n.coeffs.remove(0), 'coefficients'),
        ('PowerNodeType', lambda n: setattr(n, 'extra_params', True), 'trainable'),
        ('PowerNodeType', lambda n: setattr(n, 'use_custom_weight', True), 'custom'),
    ]
    for index, (kind, mutate, expected) in enumerate(invalid):
        tree, data, solver, node = make_graph(kind)
        mutate(node)
        reject(tree, temp / ('invalid_' + str(index)), expected)
        bpy.data.node_groups.remove(tree)
    tree, data, solver, node = make_graph('PoolNodeType')
    node.mode = 'STOCHASTIC'
    reject(tree, temp / 'cpu_stochastic_pool', 'GPU mode')
    bpy.data.node_groups.remove(tree)
    tree, data, solver, node = make_graph()
    data.db_type, data.output_amount = 'ImageData', 1
    reject(tree, temp / 'image_data_single_output', 'exactly two')
    bpy.data.node_groups.remove(tree)
    tree, data, solver, node = make_graph()
    solver.lr_policy = 'multistep'
    reject(tree, temp / 'multistep', 'multistep')
    solver.lr_policy, solver.solver_type, solver.delta = 'fixed', 'ADAGRAD', 0
    reject(tree, temp / 'adagrad', 'delta')
    solver.delta = 1e-8
    reject(tree, temp / 'adagrad_momentum', 'momentum')
    solver.solver_type, solver.solvername = 'SGD', '../escape'
    reject(tree, temp / 'unsafe_solver', 'solver name')
    bpy.data.node_groups.remove(tree)
    results['checks'].append('invalid dropout, exponential/log bases, convolution groups, fillers, slices, coefficients, parameter types, solver schedules/names fail before writing')

    tree, data, solver, node = make_graph('PReluNodeType')
    node.extra_params = True
    node.weight_params.name = 'shared_slope'
    prelu = temp / 'prelu'
    composer.export_caffe(tree, prelu)
    train = (prelu / 'regression_train_test.prototxt').read_text()
    assert len(re.findall(r'^    param \{', train, flags=re.MULTILINE)) == 1, 'PReLU needs exactly one parameter'
    assert 'shared_slope' in train
    parse_exports(prelu)
    bpy.data.node_groups.remove(tree)
    results['checks'].append('shared PReLU exports exactly one learnable parameter')

    # Typed custom weight fragments reject malformed dimensions/data and fields
    # that could override graph connectivity or parameters.
    tree, data, solver, node = make_graph('ConvNodeType')
    node.num_output, node.kernel_size, node.bias_term = 1, 1, False
    node.use_custom_weight = True
    fragment = temp / 'weights.prototxt'
    node.custom_weight = str(fragment)
    good = 'blobs { shape { dim: 1 dim: 1 dim: 1 dim: 1 } data: 0.125 }\n'
    fragment.write_text(good)
    custom = temp / 'custom'
    before = graph.to_dict(tree)
    composer.export_caffe(tree, custom)
    assert graph.to_dict(tree) == before
    assert 'data: 0.125' in (custom / 'regression_deploy.prototxt').read_text()
    parse_exports(custom)
    bad_fragments = [
        '', good[:-3], good + 'bottom: "injected"',
        good.replace('data: 0.125', 'data: nan'),
        good.replace('data: 0.125', 'data: 1e999'),
        good.replace('data: 0.125', 'data: 1e100'),
        good.replace('dim: 1 dim: 1 dim: 1 dim: 1', 'dim: 1.0 dim: 1 dim: 1 dim: 1'),
        good.replace('data: 0.125', 'data: 1 data: 2'),
        good.replace('dim: 1 dim: 1 dim: 1 dim: 1', 'dim: -1'),
        good.replace('shape {', 'shape { dim: 1.5'),
        good.replace('data: 0.125', 'unknown: 1'),
        good.replace('data: 0.125', 'data: 0.125 diff: 1 diff: 2'),
        good + good,
        good.replace('shape {', 'num: 1 channels: 1 height: 1 width: 1 shape {'),
    ]
    for bad in bad_fragments:
        fragment.write_text(bad)
        reject(tree, custom, 'custom weight')
    fragment.write_text(good)
    node.bias_term = True
    reject(tree, custom, 'blob count')
    bpy.data.node_groups.remove(tree)
    results['checks'].append('custom weights parse as finite numeric BlobProto data with matching shape/count, without layer-field injection')

    # The only shape repair is for immutable bundled seven-tap derivative files.
    expected = {'Dmx1chan.prototxt': [-1, 1, 0], 'Dmy1chan.prototxt': [-1, 1, 0],
                'Dpx1chan.prototxt': [0, -1, 1], 'Dpy1chan.prototxt': [0, -1, 1]}
    for name, values in expected.items():
        tree, data, solver, node = make_graph('ConvNodeType')
        node.num_output, node.bias_term, node.square_kernel = 1, False, False
        node.kernel_h, node.kernel_w = (1, 3) if name[2] == 'x' else (3, 1)
        node.custom_weight, node.use_custom_weight = 'operators/' + name, True
        bundled = composer.EXAMPLES / node.custom_weight
        raw = bundled.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == caffe_export.LEGACY_OPERATOR_HASHES[name]
        repaired = temp / name
        composer.export_caffe(tree, repaired)
        deploy = (repaired / 'regression_deploy.prototxt').read_text()
        assert '# Repaired bundled ' + name in deploy
        assert [float(value) for value in re.findall(r'\bdata: ([-+eE.0-9]+)', deploy)] == values
        assert bundled.read_bytes() == raw
        parse_exports(repaired)
        # Even byte-identical files elsewhere are user fragments and never repaired.
        copy = temp / ('copied_' + name)
        copy.write_bytes(raw)
        node.custom_weight = str(copy)
        reject(tree, temp / ('copied_export_' + name), 'weight shape')
        # Explicitly exercise the zero-border predicate independently of hashes.
        dimensions = [1, 1, 1, 7] if name[2] == 'x' else [1, 1, 7, 1]
        fields = {'data': [1, 0] + values + [0, 0]}
        kernel = [node.kernel_h, node.kernel_w]
        assert caffe_export._repair_legacy_operator(bundled, raw, node, dimensions, fields, kernel) is None
        bpy.data.node_groups.remove(tree)
    results['checks'].append('four hash-pinned bundled derivative repairs trim only symmetric zero borders, preserve coefficients/assets, and disclose changes in exported comments')

    # Fault injection verifies write staging and rollback, including a recovery
    # folder if the filesystem prevents restoring an old file.
    tree, data, solver, node = make_graph()
    atomic = temp / 'atomic'
    composer.export_caffe(tree, atomic)
    original = contents(atomic)
    node.scale = 2
    before = graph.to_dict(tree)
    real_write = Path.write_text
    writes = 0
    def fail_write(path, *arguments, **keywords):
        global writes
        writes += 1
        if writes == 2:
            raise OSError('injected write failure')
        return real_write(path, *arguments, **keywords)
    with mock.patch.object(Path, 'write_text', fail_write):
        try:
            composer.export_caffe(tree, atomic)
        except OSError as error:
            assert 'injected' in str(error)
        else:
            raise AssertionError('Injected write failure did not occur')
    assert contents(atomic) == original and graph.to_dict(tree) == before
    assert not list(atomic.glob('.noc-export-*'))
    real_replace = os.replace
    replacements = 0
    def fail_replace(source, target):
        global replacements
        replacements += 1
        if replacements == 2:
            raise OSError('injected replacement failure')
        return real_replace(source, target)
    with mock.patch.object(composer.os, 'replace', fail_replace):
        try:
            composer.export_caffe(tree, atomic)
        except OSError as error:
            assert 'injected' in str(error)
        else:
            raise AssertionError('Injected replacement failure did not occur')
    assert contents(atomic) == original and graph.to_dict(tree) == before
    assert not list(atomic.glob('.noc-export-*'))
    replacements = 0
    def persistent_replace_failure(source, target):
        global replacements
        replacements += 1
        if replacements >= 2:
            raise OSError('injected persistent filesystem failure')
        return real_replace(source, target)
    with mock.patch.object(composer.os, 'replace', persistent_replace_failure):
        try:
            composer.export_caffe(tree, atomic)
        except RuntimeError as error:
            assert 'previous files are preserved' in str(error)
        else:
            raise AssertionError('Failed rollback was not reported')
    recovery, = atomic.glob('.noc-export-*')
    assert (recovery / 'regression_train_test.prototxt.previous').read_bytes() == original['regression_train_test.prototxt']
    for filename, content in original.items():
        (atomic / filename).write_bytes(content)
    shutil.rmtree(recovery)
    results['checks'].append('staged write/replacement failures preserve complete prior exports; persistent rollback failures retain recovery evidence')

    outside = temp / 'outside.txt'
    outside.write_text('outside file')
    script = atomic / 'train_regression.sh'
    script.unlink()
    script.symlink_to(outside)
    reject(tree, atomic, 'regular file')
    assert outside.read_text() == 'outside file' and script.is_symlink()
    bpy.data.node_groups.remove(tree)
    results['checks'].append('export refuses symlink and non-regular targets without touching external files')

    # Blender paths beginning with // resolve against the saved .blend, even
    # when the working directory and selected export directory are elsewhere.
    tree, data, solver, output = make_graph('HDF5OutputNodeType')
    blend_directory = temp / 'saved project'
    blend_directory.mkdir()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_directory / 'relative.blend'))
    data.db_type = 'ImageData'
    data.train_data = data.test_data = '//datasets/images.txt'
    data.use_mean_file, data.mean_file = True, '//datasets/mean.binaryproto'
    solver.snapshot_prefix, solver.caffe_exec = '//snapshots/model', '//tools'
    output.filename = '//outputs/result.h5'
    composer.export_caffe(tree, '//relative_export')
    directory = blend_directory / 'relative_export'
    train = (directory / 'regression_train_test.prototxt').read_text()
    config = (directory / 'regression_solver.prototxt').read_text()
    script = (directory / 'train_regression.sh').read_text()
    assert 'file_name: ' + caffe_export.proto_string(str(blend_directory / 'outputs/result.h5')) in train
    assert 'source: ' + caffe_export.proto_string(str(blend_directory / 'datasets/images.txt')) in train
    assert 'mean_file: ' + caffe_export.proto_string(str(blend_directory / 'datasets/mean.binaryproto')) in train
    assert 'snapshot_prefix: ' + caffe_export.proto_string(str(blend_directory / 'snapshots/model')) in config
    assert str(blend_directory / 'tools/caffe') in script
    parse_exports(directory)
    bpy.data.node_groups.remove(tree)
    results['checks'].append('all Blender-relative input, mean, output, snapshot, executable and export paths resolve against the saved project')

addon.unregister()
print('VALIDATION_JSON=' + json.dumps(results, sort_keys=True))
if args.output:
    args.output.write_text(json.dumps(results, indent=2) + '\n')
