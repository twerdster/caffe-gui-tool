"""Modernized Caffe-GUI-Tool implementation; see NOTICE and LICENSES/Unlicense.txt."""
__author__ = 'gip'
bl_info = {'name': 'Create Caffe solution', 'category': 'Object'}
import bpy
import json
import hashlib
import math
from pathlib import Path
import re
import shlex
tab = '    '
tab2 = tab + tab
tab3 = tab2 + tab

def proto_string(value):
    return json.dumps(str(value), ensure_ascii=False)

def getFillerString(filler, name):
    fillerString = tab3 + 'type: %s\n' % proto_string(filler.type)
    if filler.type == 'constant':
        fillerString += tab3 + 'value: %.9g\n' % filler.value
    elif filler.type == 'xavier' or filler.type == 'msra':
        fillerString += tab3 + 'variance_norm: %s\n' % filler.variance_norm
    elif filler.type == 'gaussian':
        fillerString += tab3 + 'mean: %.9g\n' % filler.mean
        fillerString += tab3 + 'std: %.9g\n' % filler.std
        if filler.is_sparse:
            fillerString += tab3 + 'sparse: %i\n' % filler.sparse
    elif filler.type == 'uniform':
        fillerString += tab3 + 'min: %.9g\n' % filler.min
        fillerString += tab3 + 'max: %.9g\n' % filler.max
    string = '        %s {\n%s\n        }\n' % (name, fillerString)
    return string

def conv_template(node):
    if node.square_padding:
        padding_string = tab2 + 'pad: %i\n' % node.pad
    else:
        padding_string = tab2 + 'pad_h: %i\n' % node.pad_h
        padding_string += tab2 + 'pad_w: %i\n' % node.pad_w
    if node.square_kernel:
        kernel_string = tab2 + 'kernel_size: %i\n' % node.kernel_size
    else:
        kernel_string = tab2 + 'kernel_h: %i\n' % node.kernel_h
        kernel_string += tab2 + 'kernel_w: %i\n' % node.kernel_w
    if node.square_stride:
        stride_string = tab2 + 'stride: %i\n' % node.stride
    else:
        stride_string = tab2 + 'stride_h: %i\n' % node.stride_h
        stride_string += tab2 + 'stride_w: %i\n' % node.stride_w
    weight_filler_string = ''
    bias_filler_string = ''
    if not node.use_custom_weight:
        weight_filler_string = getFillerString(node.weight_filler, 'weight_filler')
        bias_filler_string = getFillerString(node.bias_filler, 'bias_filler')
    string = '    convolution_param {\n        num_output: %i\n        bias_term: %i\n        group: %i\n%s\n%s\n%s\n%s\n%s\n    }\n' % (node.num_output, node.bias_term, node.group, padding_string, kernel_string, stride_string, weight_filler_string, bias_filler_string)
    return string

def data_param_template(node, source, batch_size):
    string = '    data_param {\n        source: %s\n        backend: %s\n        batch_size: %i\n        rand_skip: %i\n    }\n' % (proto_string(source), node.db_type, batch_size, node.rand_skip)
    return string

def image_data_param_template(node, source, batch_size):
    string = '    image_data_param {\n        source: %s\n        batch_size: %i\n        rand_skip: %i\n        shuffle: %i\n        new_height: %i\n        new_width: %i\n        is_color: %i\n    }\n' % (proto_string(source), batch_size, node.rand_skip, node.shuffle, node.new_height, node.new_width, node.is_color)
    return string

def transform_param_template(node):
    mean_file_string = ''
    if node.use_mean_file:
        mean_file_string = tab2 + 'mean_file: %s\n' % proto_string(bpy.path.abspath(node.mean_file))
    string = '    transform_param {\n        scale: %.9g\n        mirror: %i\n%s\n    }\n' % (node.scale, node.mirror, mean_file_string)
    return string

def hdf5_data_template(node, source, batch_size):
    string = '    hdf5_data_param {\n        source: %s\n        batch_size: %i\n        shuffle: %i\n    }\n' % (proto_string(source), batch_size, node.shuffle)
    return string

def pool_template(node):
    string = '    pooling_param {\n        pool: %s\n        kernel_size: %i\n        stride: %i\n    }\n' % (node.mode, node.kernel_size, node.stride)
    return string

def padding_template(node):
    string = '    padding_param {\n        pad_method: %s\n        pad_l: %i\n        pad_r: %i\n        pad_t: %i\n        pad_b: %i\n        pad_value: %.9g\n    }\n' % (node.mode, node.padl, node.padr, node.padt, node.padb, node.padval)
    return string

def mvntemplate(name, bottom, normalize_variance, across_channels, eps):
    string = 'layer {\n        name: %s\n        type: "MVN"\n        bottom: %s\n        top: %s\n        mvn_param  {\n        normalize_variance: %s\n        across_channels: %s\n        eps: %.9g\n        }\n        }\n' % (proto_string(name), proto_string(bottom), proto_string(name), normalize_variance, across_channels, eps)
    return string

def FC_template(node):
    weight_filler_string = getFillerString(node.weight_filler, 'weight_filler')
    bias_filler_string = getFillerString(node.bias_filler, 'bias_filler')
    string = '    inner_product_param {\n        num_output: %i\n        bias_term: %i\n%s\n%s\n        axis: %i\n    }\n' % (node.num_output, node.bias_term, weight_filler_string, bias_filler_string, node.axis)
    return string

def PReLU_template(node):
    filler_string = getFillerString(node.filler, 'filler')
    string = '    prelu_param {\n        channel_shared: %i\n%s\n    }\n' % (node.channel_shared, filler_string)
    return string

def Concattemplate(node):
    string = '    concat_param {\n        axis: %i\n    }\n    ' % node.axis
    return string

def argmaxtemplate(node):
    string = '    argmax_param {\n        out_max_val: %i\n        top_k: %i\n    }\n' % (node.OutMaxVal, node.TopK)
    return string

def hdf5outputtemplate(node):
    string = '    hdf5_output_param {\n        file_name: %s\n    }\n' % proto_string(bpy.path.abspath(node.filename))
    return string

def logtemplate(node):
    string = '    log_param {\n        scale: %.9g\n        shift: %.9g\n        base: %.9g\n    }\n' % (node.scale, node.shift, node.base)
    return string

def powertemplate(node):
    string = '    power_param {\n        power: %.9g\n        scale: %.9g\n        shift: %.9g\n    }\n' % (node.power, node.scale, node.shift)
    return string

def reductiontemplate(node):
    string = '    reduction_param {\n        operation: %s\n        axis: %i\n        coeff: %.9g\n    }\n' % (node.operation, node.axis, node.coeff)
    return string

def slicetemplate(node):
    slice_points_string = '\n'.join(map(lambda x: tab2 + 'slice_point: %i' % x.slice_point, node.slice_points))
    string = '    slice_param {\n        axis: %i\n%s\n    }\n' % (node.axis, slice_points_string)
    return string

def solver_template(node, config_path=None):
    net_path = str(Path(config_path if config_path is not None else bpy.path.abspath(node.config_path)) / (node.solvername + '_train_test.prototxt'))
    lr_string = ''
    if node.lr_policy == 'step':
        lr_string += 'gamma: %.9g\n' % node.gamma
        lr_string += 'stepsize: %i\n' % node.stepsize
    elif node.lr_policy == 'exp':
        lr_string += 'gamma: %.9g\n' % node.gamma
    elif node.lr_policy == 'inv':
        lr_string += 'gamma: %.9g\n' % node.gamma
        lr_string += 'power: %.9g\n' % node.power
    elif node.lr_policy == 'multistep':
        raise ValueError('The multistep learning-rate policy is not supported; choose another policy')
    elif node.lr_policy == 'poly':
        lr_string += 'power: %.9g\n' % node.power
    elif node.lr_policy == 'sigmoid':
        lr_string += 'gamma: %.9g\n' % node.gamma
        lr_string += 'stepsize: %i\n' % node.stepsize
    random_seed_string = ''
    if node.use_random_seed:
        random_seed_string = 'random_seed: %i' % node.random_seed
    delta_string = ''
    if node.solver_type == 'ADAGRAD':
        delta_string = 'delta: %.9g' % node.delta
    string = 'net: %s\ntest_iter: %i\ntest_interval: %i\ntest_compute_loss: %i\ntest_initialization: %i\nbase_lr: %.9g\ndisplay: %i\naverage_loss: %i\nmax_iter: %i\niter_size: %i\nlr_policy: %s\n%s\nmomentum: %.9g\nweight_decay: %.9g\nregularization_type: %s\nsnapshot: %i\nsnapshot_prefix: %s\nsnapshot_diff: %i\nsolver_mode: %s\n%s\nsolver_type: %s\n%s\ndebug_info: %i\nsnapshot_after_train: %i\n' % (proto_string(bpy.path.abspath(net_path)), node.test_iter, node.test_interval, node.test_compute_loss, node.test_initialization, node.base_lr, node.display, node.average_loss, node.max_iter, node.iter_size, proto_string(node.lr_policy), lr_string, node.momentum, node.weight_decay, proto_string(node.regularization_type), node.snapshot, proto_string(bpy.path.abspath(node.snapshot_prefix)), node.snapshot_diff, node.solver_mode, random_seed_string, node.solver_type, delta_string, node.debug_info, node.snapshot_after_train)
    return '\n'.join(filter(lambda x: x.strip(), string.splitlines())) + '\n'

def deploytemplate(batch, channels, size, datain):
    deploystring = 'name: "Autogen"\ninput: %s\ninput_dim: %i\ninput_dim: %i\ninput_dim: %i\ninput_dim: %i\n' % (proto_string(datain), batch, channels, size, size)
    return deploystring

def scripttemplate(caffepath, configpath, solvername, gpus, solver):
    command = [str(Path(caffepath) / 'caffe'), 'train',
               '--solver=' + str(Path(configpath) / (solvername + '_solver.prototxt'))]
    if solver == 'GPU':
        command.append('--gpu=' + (gpus or '0'))
    return '#!/usr/bin/env sh\nexec ' + shlex.join(command) + '\n'

def loss_weight_template(loss_weight):
    return tab + 'loss_weight: %.9g' % loss_weight

def param_template(param):
    string = tab + 'param {\n'
    if param.name.strip():
        string += tab2 + 'name: %s\n' % proto_string(param.name)
    string += tab2 + 'lr_mult: %.9g\n' % param.lr_mult
    string += tab2 + 'decay_mult: %.9g\n' % param.decay_mult
    string += tab + '}'
    return string

def get_params(node):
    params = []
    if node.extra_params:
        params.append(param_template(node.weight_params))
        if getattr(node, 'bias_term', False):
            params.append(param_template(node.bias_params))
    return params

def get_include_in(node, phase=None):
    phase = node.include_in if phase is None else phase
    if phase == 'BOTH':
        return ''
    return '    include {\n        phase: %s\n    }\n' % phase

def layer_template(node, tops, bottoms, special_params, *, phase=None, layer_type=None, custom_weights=''):
    tops_string = '\n'.join(tab + 'top: %s' % proto_string(value) for value in tops)
    bottoms_string = '\n'.join(tab + 'bottom: %s' % proto_string(value) for value in bottoms)
    params_string = '\n'.join(get_params(node))
    special_params_string = '\n'.join(special_params)
    include_in_string = get_include_in(node, phase)
    custom_weight_string = '\n'.join(tab + row for row in custom_weights.splitlines())
    if layer_type is None:
        layer_type = node.mode.strip(chr(34)) if node.bl_idname == 'AcNodeType' else node.n_type
    string = 'layer {\n    name: %s\n    type: %s\n%s\n%s\n%s\n%s\n%s\n%s\n}\n' % (proto_string(node.name), proto_string(layer_type), tops_string, bottoms_string, params_string, special_params_string, include_in_string, custom_weight_string)
    return '\n'.join(filter(lambda row: row.strip(), string.splitlines())) + '\n'

def LRNtemplate(node):
    string = '    lrn_param {\n        local_size: %i\n        alpha: %.9g\n        beta: %.9g\n        norm_region: %s\n    }\n    ' % (node.size, node.alpha, node.beta, node.mode)
    return string

def eltwise_template(node):
    stable_prod_grad_string = ''
    coeffs_string = ''
    if node.operation == 'PROD':
        stable_prod_grad_string = tab2 + 'stable_prod_grad: %i' % node.stable_prod_grad
    elif node.operation == 'SUM':
        coeffs_string = '\n'.join(map(lambda x: tab2 + 'coeff: %.9g' % x.coeff, node.coeffs))
    string = '    eltwise_param {\n        operation: %s\n%s\n%s\n    }\n' % (node.operation, coeffs_string, stable_prod_grad_string)
    return string

def Relutemplate(node):
    string = '    relu_param {\n        negative_slope: %.9g\n    }\n    ' % node.negative_slope
    return string

def dropouttemplate(node):
    string = '    dropout_param {\n        dropout_ratio: %.9g\n    }\n    ' % node.dropout_ratio
    return string

class Vertex:
    pass

def reorder(graph):
    res_string = []
    res_dstring = []
    while len(graph) > 0:
        curr = min(graph, key=lambda x: len(x.bottoms))
        if len(curr.bottoms) != 0:
            raise ValueError('Unresolved Caffe graph dependencies')
        res_string.append(curr.string)
        res_dstring.append(curr.dstring)
        for item in graph:
            for top in curr.tops:
                try:
                    item.bottoms.remove(top)
                except:
                    pass
        graph.remove(curr)
    return (res_string, res_dstring)

# These layers consume labels or write training/evaluation artifacts.
DEPLOY_EXCLUDED = {'DataNodeType', 'SMLossNodeType', 'SCELossNodeType',
                   'EULossNodeType', 'AccuracyNodeType', 'HDF5OutputNodeType',
                   'SolverNodeType', 'NodeFrame', 'NodeReroute'}
TRAINABLE = {'ConvNodeType', 'DeConvNodeType', 'FCNodeType', 'PReluNodeType'}


def _validate_filler(filler, node, key):
    if filler.type == 'uniform' and filler.min > filler.max:
        raise ValueError('Uniform filler minimum exceeds maximum: ' + node.name)
    if filler.type == 'gaussian' and filler.std <= 0:
        raise ValueError('Gaussian filler standard deviation must be positive: ' + node.name)
    if filler.type == 'bilinear':
        if key != 'weight_filler' or node.bl_idname not in {'ConvNodeType', 'DeConvNodeType'} or (not node.square_kernel and node.kernel_h != node.kernel_w):
            raise ValueError('Bilinear filler requires square convolution weights: ' + node.name)


def validate_parameters(nodes):
    """Reject definitions which parse but fail Caffe's layer setup checks."""
    # Blender's bounds do not cover cross-field constraints or edited collections.
    from .graph import properties
    solver_mode = next((node.solver_mode for node in nodes if node.bl_idname == 'SolverNodeType'), None)
    def finite(value, name):
        if isinstance(value, dict):
            for key, item in value.items():
                finite(item, name + '/' + key)
        elif isinstance(value, list):
            for item in value:
                finite(item, name)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Non-finite parameter: ' + name)
    for node in nodes:
        finite(properties(node), node.name)
        kind = node.bl_idname
        if node.extra_params and kind not in TRAINABLE:
            raise ValueError('Shared weight parameters require a trainable layer: ' + node.name)
        if node.use_custom_weight and kind not in TRAINABLE:
            raise ValueError('Custom weights require a trainable layer: ' + node.name)
        if kind in {'ConvNodeType', 'DeConvNodeType'} and node.num_output % node.group:
            raise ValueError('Convolution outputs must be divisible by group: ' + node.name)
        if kind == 'PoolNodeType' and node.mode == 'STOCHASTIC' and solver_mode == 'CPU':
            raise ValueError('Stochastic pooling requires Caffe GPU mode: ' + node.name)
        if kind == 'LRNNodeType' and node.size % 2 != 1:
            raise ValueError('LRN local size must be odd: ' + node.name)
        if kind == 'DropoutNodeType' and not 0 <= node.dropout_ratio < 1:
            raise ValueError('Dropout ratio must be at least zero and less than one: ' + node.name)
        if kind in {'ExpNodeType', 'LogNodeType'} and node.base != -1 and (node.base <= 0 or (kind == 'LogNodeType' and node.base == 1)):
            raise ValueError('Invalid Log/Exp base (use -1 for e; Log also excludes one): ' + node.name)
        if kind == 'EltwiseNodeType' and node.operation == 'SUM' and len(node.coeffs) not in {0, len(node.inputs)}:
            raise ValueError('Eltwise SUM needs zero coefficients or one per input: ' + node.name)
        if kind == 'SliceNodeType':
            points = [point.slice_point for point in node.slice_points]
            if len(points) != len(node.outputs) - 1 or any(point <= 0 for point in points) or any(a >= b for a, b in zip(points, points[1:])):
                raise ValueError('Slice points must be positive, strictly increasing, and match the outputs: ' + node.name)
        if kind == 'DataNodeType':
            if node.db_type in {'LMDB', 'LEVELDB'} and len(node.outputs) not in {1, 2}:
                raise ValueError('Data supports one or two outputs; use HDF5Data for more: ' + node.name)
            if node.db_type == 'ImageData' and len(node.outputs) != 2:
                raise ValueError('ImageData requires exactly two outputs (image and label): ' + node.name)
            if node.db_type == 'ImageData' and bool(node.new_height) != bool(node.new_width):
                raise ValueError('Image resize height and width must both be zero or both be positive: ' + node.name)
        if kind == 'SolverNodeType' and node.solver_type == 'ADAGRAD':
            if node.delta <= 0:
                raise ValueError('AdaGrad delta must be positive: ' + node.name)
            if node.momentum != 0:
                raise ValueError('AdaGrad requires momentum zero: ' + node.name)
        for key in ('weight_filler', 'bias_filler', 'filler'):
            if hasattr(node, key) and not node.use_custom_weight:
                _validate_filler(getattr(node, key), node, key)


# Immutable recovered grayscale fragments had a seven-tap storage shape while
# the saved graphs declare three taps. Only these exact bundled assets may be
# repaired, and only if all symmetrically removed coefficients are zero.
LEGACY_OPERATOR_HASHES = {
    'Dmx1chan.prototxt': '3bf18290aa7375d9b688a674b2286ec75cb7a3f79ae6286fecc9a432958cb3bf',
    'Dmy1chan.prototxt': 'b098000985aa12aba3fb225fe09f5e01d633140f84de7edb4bbb0d2828419718',
    'Dpx1chan.prototxt': 'a9c674b48980a4d8614bf4b1be3815497dfca87b5f435ae16f57a98901de21de',
    'Dpy1chan.prototxt': 'cb9e0a09bc4a710e03d0b7e0f6730834978b2dc10a9c096463268e66aa47b4be',
}


def _repair_legacy_operator(path, raw, node, dimensions, fields, kernel):
    path = Path(path)
    bundled = Path(__file__).parent / 'examples' / 'operators' / path.name
    if (path.name not in LEGACY_OPERATOR_HASHES or path.resolve() != bundled.resolve()
            or hashlib.sha256(raw).hexdigest() != LEGACY_OPERATOR_HASHES[path.name]
            or node.bl_idname != 'ConvNodeType' or node.num_output != 1 or node.group != 1
            or node.bias_term or dimensions[:2] != [1, 1]
            or (dimensions[2:], kernel) not in [([1, 7], [1, 3]), ([7, 1], [3, 1])]):
        return None
    values = fields['data']
    removed = values[:2] + values[-2:]
    if any(value != 0 for value in removed):
        return None
    cropped = values[2:-2]
    comment = '# Repaired bundled %s: removed symmetric zero borders (7 taps to 3); source asset unchanged.\n' % path.name
    shape = ' '.join('dim: %i' % value for value in [1, 1] + kernel)
    return comment + 'blobs { shape { ' + shape + ' } ' + ' '.join('data: %.9g' % value for value in cropped) + ' }\n'


def _custom_weights(path, node):
    """Read a numeric BlobProto fragment without allowing layer-field injection.

    This intentionally accepts only blobs, dimensions and numeric data/diff;
    layer configuration belongs to the node's typed properties.
    """
    raw = Path(path).read_bytes()
    text = raw.decode('utf-8')
    token = re.compile(r'\s+|#[^\n]*|//[^\n]*|/\*.*?\*/|[A-Za-z_][A-Za-z_0-9]*|[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|[{}<>:\[\],;]', re.DOTALL)
    tokens, offset = [], 0
    for match in token.finditer(text):
        if match.start() != offset:
            raise ValueError('Invalid custom weight syntax: ' + node.name)
        value = match.group()
        if not value.isspace() and not value.startswith(('#', '//', '/*')):
            tokens.append(value)
        offset = match.end()
    if offset != len(text):
        raise ValueError('Invalid custom weight syntax: ' + node.name)
    cursor = 0
    def pop(expected=None):
        nonlocal cursor
        if cursor == len(tokens):
            raise ValueError('Incomplete custom weights: ' + node.name)
        value = tokens[cursor]
        cursor += 1
        if expected is not None and value != expected:
            raise ValueError('Invalid custom weight field: ' + node.name)
        return value
    def opening():
        if cursor < len(tokens) and tokens[cursor] == ':':
            pop(':')
        brace = pop()
        if brace not in {'{', '<'}:
            raise ValueError('Expected custom weight blob: ' + node.name)
        return '}' if brace == '{' else '>'
    def numbers(integer=False):
        pop(':')
        values = []
        multiple = cursor < len(tokens) and tokens[cursor] == '['
        if multiple:
            pop('[')
        while True:
            try:
                literal = pop()
                if integer:
                    if not re.fullmatch(r'[+-]?(?:0|[1-9][0-9]*)', literal):
                        raise ValueError('Integer dimension required')
                    number = int(literal)
                else:
                    number = float(literal)
            except (ValueError, OverflowError) as error:
                raise ValueError('Invalid numeric custom weight: ' + node.name) from error
            if not integer and not math.isfinite(number):
                raise ValueError('Non-finite custom weight: ' + node.name)
            values.append(number)
            if not multiple or tokens[cursor:cursor + 1] != [',']:
                break
            pop(',')
        if multiple:
            pop(']')
        return values
    blobs = []
    while cursor < len(tokens):
        pop('blobs')
        close = opening()
        fields, dimensions = {}, None
        while tokens[cursor:cursor + 1] != [close]:
            field = pop()
            if field == 'shape':
                if dimensions is not None:
                    raise ValueError('Duplicate custom weight shape: ' + node.name)
                shape_close = opening()
                dimensions = []
                while tokens[cursor:cursor + 1] != [shape_close]:
                    pop('dim')
                    dimensions.extend(numbers(integer=True))
                    while tokens[cursor:cursor + 1] in [[','], [';']]:
                        pop()
                pop(shape_close)
            elif field in {'num', 'channels', 'height', 'width', 'data', 'diff', 'double_data', 'double_diff'}:
                integer = field in {'num', 'channels', 'height', 'width'}
                values = numbers(integer=integer)
                if field in {'data', 'diff'} and any(abs(value) > 3.4028234663852886e38 for value in values):
                    raise ValueError('Custom weight exceeds finite float32 range: ' + node.name)
                if integer and any(value > 2147483647 for value in values):
                    raise ValueError('Custom weight legacy dimension exceeds int32: ' + node.name)
                if field in {'num', 'channels', 'height', 'width'} and (field in fields or len(values) != 1):
                    raise ValueError('Duplicate custom weight dimension: ' + node.name)
                fields.setdefault(field, []).extend(values)
            else:
                raise ValueError('Only numeric blobs are allowed in custom weights: ' + node.name)
            while tokens[cursor:cursor + 1] in [[','], [';']]:
                pop()
        pop(close)
        while tokens[cursor:cursor + 1] in [[','], [';']]:
            pop()
        legacy = [field in fields for field in ('num', 'channels', 'height', 'width')]
        if dimensions is not None and any(legacy):
            raise ValueError('Use shape or legacy custom weight dimensions, not both: ' + node.name)
        if dimensions is None:
            if not all(legacy):
                raise ValueError('Custom weight blob is missing dimensions: ' + node.name)
            dimensions = [fields[field][0] for field in ('num', 'channels', 'height', 'width')]
        if any(value <= 0 or value > 9223372036854775807 for value in dimensions):
            raise ValueError('Custom weight dimensions must be positive integers: ' + node.name)
        size = math.prod(int(value) for value in dimensions)
        data_fields = [key for key in ('data', 'double_data') if key in fields]
        if len(data_fields) != 1 or len(fields[data_fields[0]]) != size:
            raise ValueError('Custom weight data count does not match shape: ' + node.name)
        if any(key in fields and len(fields[key]) != size for key in ('diff', 'double_diff')):
            raise ValueError('Custom weight diff count does not match shape: ' + node.name)
        blobs.append(dimensions)
    expected = 1 + int(getattr(node, 'bias_term', False))
    if len(blobs) != expected:
        raise ValueError('Custom weight blob count does not match the layer weight/bias settings: ' + node.name)
    if node.bl_idname in {'ConvNodeType', 'DeConvNodeType'}:
        dimensions = blobs[0]
        kernel = [node.kernel_size, node.kernel_size] if node.square_kernel else [node.kernel_h, node.kernel_w]
        output_axis = 0 if node.bl_idname == 'ConvNodeType' else 1
        expected_outputs = node.num_output if output_axis == 0 else node.num_output // node.group
        if len(dimensions) != 4 or dimensions[output_axis] != expected_outputs or dimensions[2:] != kernel:
            repaired = _repair_legacy_operator(path, raw, node, dimensions, fields, kernel) if len(blobs) == 1 else None
            if repaired is not None:
                return repaired
            raise ValueError('Custom convolution weight shape does not match output/kernel settings: ' + node.name)
    if node.bl_idname == 'FCNodeType' and (len(blobs[0]) != 2 or blobs[0][0] != node.num_output):
        raise ValueError('Custom inner-product weight shape does not match layer outputs: ' + node.name)
    if node.bl_idname == 'PReluNodeType' and (len(blobs[0]) != 1 or (node.channel_shared and blobs[0] != [1])):
        raise ValueError('Custom PReLU weight shape must match the shared slope setting: ' + node.name)
    if len(blobs) == 2 and math.prod(blobs[1]) != node.num_output:
        raise ValueError('Custom bias count does not match the layer outputs: ' + node.name)
    return text


def _validate_phases(nodes):
    for phase in ('TRAIN', 'TEST'):
        available = set()
        for node in nodes:
            if node.bl_idname in {'SolverNodeType', 'NodeFrame'}:
                continue
            # Data owns explicit train/test sources. Its inherited include_in
            # was scratch state in the legacy exporter and is not a UI setting.
            if node.bl_idname != 'DataNodeType' and node.include_in not in {'BOTH', phase}:
                continue
            for socket in node.inputs:
                if not socket.is_linked or socket.links[0].from_socket.output_name not in available:
                    raise ValueError('Input unavailable in ' + phase + ': ' + node.name + '/' + socket.name)
            available.update(socket.output_name for socket in node.outputs)


def generate(nodes, *, config_path=None, custom_weight_paths=None):
    """Render a topologically ordered graph without changing Blender properties."""
    nodes = list(nodes)
    solvers = [node for node in nodes if node.bl_idname == 'SolverNodeType']
    if len(solvers) != 1:
        raise ValueError('Add exactly one Solver node before exporting')
    solver = solvers[0]
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', solver.solvername) or solver.solvername in {'.', '..'}:
        raise ValueError('Solver name must use letters, digits, underscores, dots or hyphens')
    validate_parameters(nodes)
    _validate_phases(nodes)
    config_path = str(Path(config_path if config_path is not None else bpy.path.abspath(solver.config_path)))
    custom_weight_paths = custom_weight_paths or {}
    strings, deployment, used_data = [], [], set()
    deploy_available = set()
    data_nodes = [node for node in nodes if node.bl_idname == 'DataNodeType']
    data_names = {socket.output_name for node in data_nodes for socket in node.outputs}
    templates = {
        'PoolNodeType': pool_template, 'PadNodeType': padding_template,
        'ConvNodeType': conv_template, 'DeConvNodeType': conv_template,
        'FCNodeType': FC_template, 'LRNNodeType': LRNtemplate,
        'ReluNodeType': Relutemplate, 'PReluNodeType': PReLU_template,
        'DropoutNodeType': dropouttemplate, 'ConcatNodeType': Concattemplate,
        'ArgMaxNodeType': argmaxtemplate, 'HDF5OutputNodeType': hdf5outputtemplate,
        'LogNodeType': logtemplate, 'PowerNodeType': powertemplate,
        'ReductionNodeType': reductiontemplate, 'SliceNodeType': slicetemplate,
        'EltwiseNodeType': eltwise_template,
    }
    parameterless = {'FlattenNodeType', 'AbsValNodeType', 'SilenceNodeType', 'AcNodeType', 'AccuracyNodeType'}
    for node in nodes:
        kind = node.bl_idname
        if kind in {'SolverNodeType', 'NodeFrame'}:
            continue
        tops = [socket.output_name for socket in node.outputs]
        bottoms = [socket.links[0].from_socket.output_name for socket in node.inputs if socket.is_linked]
        weights = _custom_weights(custom_weight_paths.get(node.name, bpy.path.abspath(node.custom_weight)), node) if node.use_custom_weight else ''
        if kind == 'DataNodeType':
            for phase, batch, path, source in [('TRAIN', node.train_batch_size, node.train_path, node.train_data), ('TEST', node.test_batch_size, node.test_path, node.test_data)]:
                if node.db_type in {'LMDB', 'LEVELDB'}:
                    params, layer_type = [data_param_template(node, bpy.path.abspath(path), batch), transform_param_template(node)], 'Data'
                elif node.db_type == 'ImageData':
                    params, layer_type = [image_data_param_template(node, bpy.path.abspath(source), batch), transform_param_template(node)], 'ImageData'
                elif node.db_type == 'HDF5Data':
                    params, layer_type = [hdf5_data_template(node, bpy.path.abspath(source), batch)], 'HDF5Data'
                else:
                    raise ValueError('Unsupported data backend: ' + node.name)
                strings.append(layer_template(node, tops, bottoms, params, phase=phase, layer_type=layer_type))
            deploy_available.update(tops)
            continue
        if kind in templates:
            params = [templates[kind](node)]
        elif kind == 'ExpNodeType':
            params = ['exp_param { base: %.9g scale: %.9g shift: %.9g }' % (node.base, node.scale, node.shift)]
        elif kind == 'MVNNodeType':
            params = ['mvn_param { normalize_variance: %s across_channels: %s eps: %.9g }' % (str(node.normalize_variance).lower(), str(node.across_channels).lower(), node.eps)]
        elif kind in {'SMLossNodeType', 'SCELossNodeType', 'EULossNodeType'}:
            params = [loss_weight_template(node.w)]
        elif kind in parameterless:
            params = []
        else:
            raise ValueError('Unsupported Caffe node type: ' + kind)
        strings.append(layer_template(node, tops, bottoms, params, custom_weights=weights))
        if kind not in DEPLOY_EXCLUDED and node.include_in != 'TRAIN':
            if any(bottom not in deploy_available for bottom in bottoms):
                raise ValueError('Deployment input depends on an excluded training/evaluation node: ' + node.name)
            used_data.update(bottom for bottom in bottoms if bottom in data_names)
            deploy_available.update(tops)
            # A deployment network is phase-independent and must never retain TRAIN-only layers.
            deployment.append(layer_template(node, tops, bottoms, params, phase='BOTH', custom_weights=weights))
    headers = []
    for node in data_nodes:
        for index, socket in enumerate(node.outputs):
            if index == 0 or socket.output_name in used_data:
                headers.append('input: %s\ninput_shape { dim: 1 dim: %i dim: %i dim: %i }\n' % (proto_string(socket.output_name), node.input_channels, node.input_height, node.input_width))
    name = solver.solvername
    return {
        name + '_train_test.prototxt': ''.join(strings),
        name + '_deploy.prototxt': ''.join(headers + deployment),
        name + '_solver.prototxt': solver_template(solver, config_path),
        'train_' + name + '.sh': scripttemplate(bpy.path.abspath(solver.caffe_exec), config_path, name, solver.gpus, solver.solver_mode),
    }
