"""Export deterministic numerical fixtures in an isolated Blender process.

blender --background --factory-startup --python-exit-code 1 \
  --python tests/blender_runtime_fixtures.py -- --output /tmp/caffe-fixtures

The caffemodel below contains our tiny deterministic test coefficients,
not learned weights. No external protobuf schema or model is required.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import neural_operator_blender as addon
from neural_operator_blender import composer, graph


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def field(number, value):
    return varint((number << 3) | 2) + varint(len(value)) + value


def blob(shape, values):
    # Standard Caffe BlobProto: shape=7 (BlobShape dim=1), packed data=5.
    dimensions = b''.join(varint(8) + varint(size) for size in shape)
    return field(7, dimensions) + field(5, struct.pack('<' + 'f' * len(values), *values))


def export_fixture(directory, channels):
    directory.mkdir(parents=True, exist_ok=True)
    tree = bpy.data.node_groups.new(f'CPU runtime fixture {channels}', 'CaffeNodeTree')

    def node(kind, name):
        result = tree.nodes.new(kind)
        result.name = name
        for index, output in enumerate(result.outputs):
            output.output_name = name if index == 0 else name + str(index)
        return result

    def link(source, target, index=0):
        tree.links.new(source.outputs[0], target.inputs[index])

    data = node('DataNodeType', 'data')
    data.input_channels, data.input_height, data.input_width = channels, 11, 17
    data.db_type = 'HDF5Data'
    solver = node('SolverNodeType', 'solver')
    solver.solvername = 'runtime'
    solver.solver_mode = 'CPU'
    conv = node('ConvNodeType', 'correlation')
    conv.num_output, conv.group, conv.kernel_size, conv.pad = channels, channels, 3, 1
    conv.bias_term = True
    # Asymmetric filters catch accidental transposition, flipping and channel mixing.
    base = [-0.125, 0.0, 0.25, 0.0625, 0.5, -0.1875, 0.125, -0.25, 0.03125]
    weights = [value * (channel + 1) for channel in range(channels) for value in base]
    bias = [0.0625 * (channel - 1) for channel in range(channels)]
    # Inline weights verify the exporter's custom-weight path as well; OpenCV
    # requires the same coefficients in a separate binary caffemodel.
    custom = directory / 'coefficients.prototxt'
    custom.write_text('blobs { shape { dim: %d dim: 1 dim: 3 dim: 3 } %s }\n'
                      'blobs { shape { dim: %d } %s }\n' % (
                          channels, ' '.join('data: %.9g' % v for v in weights),
                          channels, ' '.join('data: %.9g' % v for v in bias)))
    conv.use_custom_weight, conv.custom_weight = True, str(custom)
    relu = node('ReluNodeType', 'activation')
    relu.negative_slope = 0.125
    power = node('PowerNodeType', 'positive_power')
    power.scale, power.shift, power.power = 0.125, 2.0, 1.5
    summation = node('EltwiseNodeType', 'weighted_sum')
    summation.operation = 'SUM'
    summation.coeffs[0].coeff, summation.coeffs[1].coeff = 0.75, -0.125
    product = node('EltwiseNodeType', 'product')
    product.operation = 'PROD'
    link(data, conv)
    link(conv, relu)
    link(relu, power)
    link(power, summation)
    link(data, summation, 1)
    link(summation, product)
    link(power, product, 1)
    before = graph.to_dict(tree)
    composer.export_caffe(tree, directory)
    assert graph.to_dict(tree) == before, 'Numerical fixture export mutated its source graph'
    layer = field(1, b'correlation') + field(2, b'Convolution')
    layer += field(7, blob([channels, 1, 3, 3], weights)) + field(7, blob([channels], bias))
    (directory / 'coefficients.caffemodel').write_bytes(field(100, layer))
    graph.save(tree, directory / 'graph.json')
    manifest = {'channels': channels, 'kernel': [weights[i:i + 9] for i in range(0, len(weights), 9)],
                'bias': bias, 'layers': ['correlation', 'activation', 'positive_power',
                                        'weighted_sum', 'product'], 'trained': False,
                'blender': bpy.app.version_string}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    bpy.data.node_groups.remove(tree)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    addon.register()
    try:
        results = [export_fixture(args.output / f'channels-{channels}', channels) for channels in (1, 3)]
        print(json.dumps({'runtime_fixtures': results}))
    finally:
        addon.unregister()


if __name__ == '__main__':
    main()
