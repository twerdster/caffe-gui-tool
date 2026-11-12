"""Modernized Caffe-GUI-Tool implementation; see NOTICE and LICENSES/Unlicense.txt."""

bl_info = {'name': 'Caffe Nodes', 'category': 'Object'}
# Bound update callbacks so a mistyped count cannot freeze the Blender interface.
MAX_SOCKETS = 1024
import bpy
from bpy.types import NodeTree, Node, NodeSocket

class CaffeTree(NodeTree):
    """A custom node tree type that will show up in the node editor header"""
    bl_idname = 'CaffeNodeTree'
    bl_label = 'Neural Operator Composer'
    bl_icon = 'NODETREE'

class ImageSocket(NodeSocket):
    """Blob socket type"""
    bl_idname = 'ImageSocketType'
    bl_label = 'Image Socket'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (0.0, 1.0, 1.0, 0.5)

class OutputSocket(NodeSocket):
    """Custom node socket type"""
    bl_idname = 'OutputSocketType'
    bl_label = 'Output Socket'
    output_name: bpy.props.StringProperty(name='')

    def draw(self, context, layout, node, text):
        layout.prop(self, 'output_name')

    def draw_color(self, context, node):
        return (0.0, 1.0, 1.0, 0.5)

class LabelSocket(NodeSocket):
    """Label socket type"""
    bl_idname = 'LabelSocketType'
    bl_label = 'Label Socket'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (0.5, 1.0, 0.2, 0.5)

class LossSocket(NodeSocket):
    """Loss socket type"""
    bl_idname = 'LossSocketType'
    bl_label = 'Loss Socket'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (1.0, 0.3, 1.0, 0.5)

class NAFlatSocket(NodeSocket):
    """NAFlat socket type"""
    bl_idname = 'NAFlatSocketType'
    bl_label = 'Linear Flat Socket'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (1.0, 0.2, 0.2, 0.5)

class AFlatSocket(NodeSocket):
    """AFlat socket type"""
    bl_idname = 'AFlatSocketType'
    bl_label = 'Non linear Flat Socket'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (0.0, 0.8, 0.8, 0.5)

class params_p_g(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name='Shared name')
    lr_mult: bpy.props.FloatProperty(default=1.0)
    decay_mult: bpy.props.FloatProperty(default=1.0)

    def draw(self, context, layout):
        layout.prop(self, 'name')
        layout.prop(self, 'lr_mult')
        layout.prop(self, 'decay_mult')

class CaffeTreeNode:

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'CaffeNodeTree'
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    phases = [('TRAIN', 'TRAIN', 'Train only'), ('TEST', 'TEST', 'Test only'), ('BOTH', 'BOTH', 'Both')]
    include_in: bpy.props.EnumProperty(items=phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})

    def draw_include_in(self, layout):
        layout.prop(self, 'include_in')

    def draw_extra_params(self, context, layout):
        layout.prop(self, 'extra_params')
        if self.extra_params:
            layout.label(text='Weight Params')
            self.weight_params.draw(context, layout)
        if not hasattr(self, 'bias_term') or self.bias_term:
            layout.label(text='Bias Params')
            self.bias_params.draw(context, layout)

    def copy(self, node):
        print('Copying from node ', node)
        for output in self.outputs:
            output.output_name = output.output_name + '0'

class DataNode(Node, CaffeTreeNode):
    input_channels: bpy.props.IntProperty(name='Deploy channels', default=1, min=1)
    input_height: bpy.props.IntProperty(name='Deploy height', default=256, min=1)
    input_width: bpy.props.IntProperty(name='Deploy width', default=256, min=1)
    """A data node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'DataNodeType'
    bl_label = 'Data input'
    bl_icon = 'SOUND'
    DBs = [('LMDB', 'LMDB', 'Lmdb database'), ('LEVELDB', 'LEVELDB', 'LevelDB database'), ('ImageData', 'ImageData', 'Image files'), ('HDF5Data', 'HDF5Data', 'HDF5 Data')]
    db_type: bpy.props.EnumProperty(name='Database type', description='Type of Data', items=DBs, default='HDF5Data')
    train_batch_size: bpy.props.IntProperty(min=1, default=100)
    test_batch_size: bpy.props.IntProperty(min=1, default=100)

    def update_tops(self, context):
        while len(self.outputs) < self.output_amount:
            self.outputs.new('OutputSocketType', 'Out%i' % len(self.outputs))
        while len(self.outputs) > self.output_amount:
            self.outputs.remove(self.outputs[len(self.outputs) - 1])
    output_amount: bpy.props.IntProperty(min=1, max=MAX_SOCKETS, default=2, update=update_tops)
    train_path: bpy.props.StringProperty(name='Train Data Path', default='', description='Get the path to the data', subtype='DIR_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    test_path: bpy.props.StringProperty(name='Test Data Path', default='', description='Get the path to the data', subtype='DIR_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    train_data: bpy.props.StringProperty(name='Train Data Path', default='', description='Get the path to the data', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    test_data: bpy.props.StringProperty(name='Test Data Path', default='', description='Get the path to the data', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    scale: bpy.props.FloatProperty(default=1.0, min=0)
    mirror: bpy.props.BoolProperty(name='Random Mirror', default=False)
    use_mean_file: bpy.props.BoolProperty(name='Use mean file', default=False)
    mean_file: bpy.props.StringProperty(name='Mean File Path', default='', description='Mean file location', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    new_height: bpy.props.IntProperty(name='New image height', min=0, default=0, soft_max=1000)
    new_width: bpy.props.IntProperty(name='New image width', min=0, default=0, soft_max=1000)
    is_color: bpy.props.BoolProperty(name='Is color image', default=True)
    shuffle: bpy.props.BoolProperty(name='Shuffle', default=False)
    rand_skip: bpy.props.IntProperty(name='Random skip', min=0, default=0, soft_max=1000)

    def init(self, context):
        self.outputs.new('OutputSocketType', 'Image Stack')
        self.outputs.new('OutputSocketType', 'Label')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'input_channels')
        layout.prop(self, 'input_height')
        layout.prop(self, 'input_width')
        layout.prop(self, 'output_amount')
        layout.prop(self, 'db_type')
        if self.db_type in ('ImageData', 'HDF5Data'):
            layout.prop(self, 'train_data')
            layout.prop(self, 'test_data')
        else:
            layout.prop(self, 'train_path')
            layout.prop(self, 'test_path')
        layout.prop(self, 'train_batch_size')
        layout.prop(self, 'test_batch_size')
        if self.db_type in ('ImageData', 'LMDB', 'LEVELDB'):
            layout.label(text='Transformation Parameters')
            layout.prop(self, 'scale')
            layout.prop(self, 'mirror')
            layout.prop(self, 'use_mean_file')
            if self.use_mean_file:
                layout.prop(self, 'mean_file')
        layout.label(text='Special Parameters')
        if self.db_type == 'ImageData':
            layout.prop(self, 'shuffle')
            layout.prop(self, 'new_height')
            layout.prop(self, 'new_width')
            layout.prop(self, 'is_color')
            layout.prop(self, 'rand_skip')
        elif self.db_type == 'HDF5Data':
            layout.prop(self, 'shuffle')
        else:
            layout.prop(self, 'rand_skip')

    def draw_label(self):
        return 'Data Node'

class filler_p_g(bpy.types.PropertyGroup):
    types = [('constant', 'constant', 'Constant val'), ('uniform', 'uniform', 'Uniform dist'), ('gaussian', 'gaussian', 'Gaussian dist'), ('positive_unitball', 'positive_unitball', 'Positive unit ball dist'), ('xavier', 'xavier', 'Xavier dist'), ('msra', 'msra', 'MSRA dist'), ('bilinear', 'bilinear', 'Bi-linear upsample weights')]
    vnormtypes = [('FAN_IN', 'FAN_IN', 'Constant val'), ('FAN_OUT', 'FAN_OUT', 'Uniform dist'), ('AVERAGE', 'AVERAGE', 'Gaussian dist')]
    type: bpy.props.EnumProperty(name='Type', items=types, default='xavier')
    value: bpy.props.FloatProperty(default=0.0, soft_max=1000.0, soft_min=-1000.0)
    min: bpy.props.FloatProperty(default=0.0, soft_max=1000.0, soft_min=-1000.0)
    max: bpy.props.FloatProperty(default=1.0, soft_max=1000.0, soft_min=-1000.0)
    mean: bpy.props.FloatProperty(default=0.0, soft_max=1000.0, soft_min=-1000.0)
    std: bpy.props.FloatProperty(default=1.0, soft_max=1000.0, soft_min=-1000.0)
    variance_norm: bpy.props.EnumProperty(name='Weight variance norm', default='FAN_IN', items=vnormtypes)
    is_sparse: bpy.props.BoolProperty(name='Use Sprasity', default=False)
    sparse: bpy.props.IntProperty(default=100, min=1)

    def draw(self, context, layout):
        layout.prop(self, 'type')
        if self.type == 'constant':
            layout.prop(self, 'value')
        elif self.type in ('xavier', 'msra'):
            layout.prop(self, 'variance_norm')
        elif self.type == 'gaussian':
            layout.prop(self, 'mean')
            layout.prop(self, 'std')
            layout.prop(self, 'is_sparse')
            if self.is_sparse:
                layout.prop(self, 'sparse')
        elif self.type == 'uniform':
            layout.prop(self, 'min')
            layout.prop(self, 'max')

class PoolNode(Node, CaffeTreeNode):
    """A pooling node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'PoolNodeType'
    bl_label = 'Pooling Node'
    bl_icon = 'SOUND'
    n_type = 'Pooling'
    modes = [('MAX', 'MAX', 'Max pooling'), ('AVE', 'AVE', 'Average pooling'), ('STOCHASTIC', 'SGD', 'Stochastic pooling')]
    kernel_size: bpy.props.IntProperty(default=2, min=1, soft_max=5)
    stride: bpy.props.IntProperty(default=2, min=1, soft_max=5)
    mode: bpy.props.EnumProperty(name='Mode', default='MAX', items=modes)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input image')
        self.outputs.new('OutputSocketType', 'Output image')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'kernel_size')
        layout.prop(self, 'stride')
        layout.prop(self, 'mode')

class PaddingNode(Node, CaffeTreeNode):
    """A padding node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'PadNodeType'
    bl_label = 'Padding Node'
    bl_icon = 'SOUND'
    n_type = 'Padding'
    modes = [('CONSTANT', 'CONSTANT', 'Set padding values'), ('REPEAT', 'REPEAT', 'Repeat border values')]
    padl: bpy.props.IntProperty(default=0, min=0, soft_max=5)
    padr: bpy.props.IntProperty(default=0, min=0, soft_max=5)
    padt: bpy.props.IntProperty(default=0, min=0, soft_max=5)
    padb: bpy.props.IntProperty(default=0, min=0, soft_max=5)
    padval: bpy.props.FloatProperty(default=0, min=-255, soft_max=255)
    mode: bpy.props.EnumProperty(name='Mode', default='REPEAT', items=modes)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input image')
        self.outputs.new('OutputSocketType', 'Output image')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'mode')
        layout.prop(self, 'padl')
        layout.prop(self, 'padr')
        layout.prop(self, 'padt')
        layout.prop(self, 'padb')
        if self.mode == 'CONSTANT':
            layout.prop(self, 'padval')

class eltwise_coeff_p_g(bpy.types.PropertyGroup):
    coeff: bpy.props.FloatProperty(default=1)

    def draw(self, context, layout):
        layout.prop(self, 'coeff')

class EltwiseNode(Node, CaffeTreeNode):
    """An element-wise node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'EltwiseNodeType'
    bl_label = 'Element-wise Node'
    bl_icon = 'SOUND'
    n_type = 'Eltwise'
    eltwiseOps = [('PROD', 'PROD', 'Eltwise prod: c(i) -> a(i)*b(i)'), ('SUM', 'SUM', 'Eltwise sum: c(i) -> a(i)+b(i)'), ('MAX', 'MAX', 'Eltwise max: c(i) -> max [a(i),b(i)]')]
    coeffs: bpy.props.CollectionProperty(type=eltwise_coeff_p_g)
    stable_prod_grad: bpy.props.BoolProperty(name='Stable(slower) gradient', default=1)
    operation: bpy.props.EnumProperty(name='Operation', default='SUM', items=eltwiseOps)

    def update_bottoms(self, context):
        while len(self.inputs) < self.input_amount:
            self.inputs.new('ImageSocketType', 'Input %i' % (len(self.inputs) + 1))
        while len(self.inputs) > self.input_amount:
            self.inputs.remove(self.inputs[len(self.inputs) - 1])
        # Legacy PROD/MAX nodes can contain no coefficients. Resize independently
        # so editing their input count repairs the UI without invalid removals.
        while len(self.coeffs) < self.input_amount:
            self.coeffs.add()
        while len(self.coeffs) > self.input_amount:
            self.coeffs.remove(len(self.coeffs) - 1)
    input_amount: bpy.props.IntProperty(min=1, max=MAX_SOCKETS, default=2, update=update_bottoms)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input 1')
        self.inputs.new('ImageSocketType', 'Input 2')
        self.outputs.new('OutputSocketType', 'Output blob')
        self.coeffs.add()
        self.coeffs.add()

    def draw_buttons(self, context, layout):
        layout.prop(self, 'input_amount')
        layout.prop(self, 'operation')
        if self.operation == 'PROD':
            layout.prop(self, 'stable_prod_grad')
        elif self.operation == 'SUM':
            for coeff in self.coeffs:
                coeff.draw(context, layout)

class ExpNode(Node, CaffeTreeNode):
    """An exponential node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'ExpNodeType'
    bl_label = 'Exponential Node'
    bl_icon = 'SOUND'
    n_type = 'Exp'
    base: bpy.props.FloatProperty(default=-1.0, soft_max=10.0, min=-1.0)
    scale: bpy.props.FloatProperty(default=1.0, soft_max=10.0, min=0)
    shift: bpy.props.FloatProperty(default=0.0, soft_max=10.0, min=-10)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input blob')
        self.outputs.new('OutputSocketType', 'Output blob')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'base')
        layout.prop(self, 'scale')
        layout.prop(self, 'shift')
        self.draw_extra_params(context, layout)

class MVNNode(Node, CaffeTreeNode):
    n_type = 'MVN'
    """Mean variance normalization node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'MVNNodeType'
    bl_label = 'MVN Node'
    bl_icon = 'SOUND'
    normalize_variance: bpy.props.BoolProperty(default=True)
    across_channels: bpy.props.BoolProperty(default=False)
    eps: bpy.props.FloatProperty(default=1e-09, soft_max=1.0, min=1e-20)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input blob')
        self.outputs.new('OutputSocketType', 'Output blob')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'normalize_variance')
        layout.prop(self, 'across_channels')
        layout.prop(self, 'eps')
        self.draw_extra_params(context, layout)

class ConvNode(Node, CaffeTreeNode):
    """A Convolution node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'ConvNodeType'
    bl_label = 'Convolution Node'
    bl_icon = 'SOUND'
    n_type = 'Convolution'
    num_output: bpy.props.IntProperty(name='Number of outputs', default=20, min=1, soft_max=300)
    bias_term: bpy.props.BoolProperty(name='Include Bias term', default=True)
    square_padding: bpy.props.BoolProperty(name='Equal x,y padding', default=True)
    pad: bpy.props.IntProperty(name='Padding', default=0, min=0, soft_max=5)
    pad_h: bpy.props.IntProperty(name='Padding height', default=0, min=0, soft_max=5)
    pad_w: bpy.props.IntProperty(name='Padding width', default=0, min=0, soft_max=5)
    square_kernel: bpy.props.BoolProperty(name='Equal x,y kernel', default=True)
    kernel_size: bpy.props.IntProperty(name='Kernel size', default=5, min=1, soft_max=25)
    kernel_h: bpy.props.IntProperty(name='Kernel height', default=5, min=1, soft_max=25)
    kernel_w: bpy.props.IntProperty(name='Kernel width', default=5, min=1, soft_max=25)
    group: bpy.props.IntProperty(name='Group size', default=1, min=1, soft_max=25)
    square_stride: bpy.props.BoolProperty(name='Equal x,y stride', default=True)
    stride: bpy.props.IntProperty(name='Stride', default=1, min=1, soft_max=5)
    stride_h: bpy.props.IntProperty(name='Stride height', default=1, min=1, soft_max=5)
    stride_w: bpy.props.IntProperty(name='Stride width', default=1, min=1, soft_max=5)
    weight_filler: bpy.props.PointerProperty(type=filler_p_g)
    bias_filler: bpy.props.PointerProperty(type=filler_p_g)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input image')
        self.outputs.new('OutputSocketType', 'Output image')
        self.color = [1, 0, 1]
        self.use_custom_color = True

    def draw_buttons(self, context, layout):
        layout.prop(self, 'num_output')
        layout.prop(self, 'bias_term')
        layout.prop(self, 'group')
        layout.prop(self, 'square_padding')
        if self.square_padding:
            layout.prop(self, 'pad')
        else:
            layout.prop(self, 'pad_h')
            layout.prop(self, 'pad_w')
        layout.prop(self, 'square_kernel')
        if self.square_kernel:
            layout.prop(self, 'kernel_size')
        else:
            layout.prop(self, 'kernel_h')
            layout.prop(self, 'kernel_w')
        layout.prop(self, 'square_stride')
        if self.square_stride:
            layout.prop(self, 'stride')
        else:
            layout.prop(self, 'stride_h')
            layout.prop(self, 'stride_w')
        layout.prop(self, 'use_custom_weight')
        if self.use_custom_weight:
            layout.prop(self, 'custom_weight')
        else:
            layout.label(text='Weight Filler')
            self.weight_filler.draw(context, layout)
            if self.bias_term:
                layout.label(text='bias Filler')
                self.bias_filler.draw(context, layout)
        self.draw_extra_params(context, layout)

class DeConvNode(Node, CaffeTreeNode):
    """A DeConvolution node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'DeConvNodeType'
    bl_label = 'DeConvolution Node'
    bl_icon = 'SOUND'
    n_type = 'Deconvolution'
    num_output: bpy.props.IntProperty(name='Number of outputs', default=20, min=1, soft_max=300)
    bias_term: bpy.props.BoolProperty(name='Include Bias term', default=True)
    square_padding: bpy.props.BoolProperty(name='Equal x,y padding', default=True)
    pad: bpy.props.IntProperty(name='Padding', default=0, min=0, soft_max=5)
    pad_h: bpy.props.IntProperty(name='Padding height', default=0, min=0, soft_max=5)
    pad_w: bpy.props.IntProperty(name='Padding width', default=0, min=0, soft_max=5)
    square_kernel: bpy.props.BoolProperty(name='Equal x,y kernel', default=True)
    kernel_size: bpy.props.IntProperty(name='Kernel size', default=5, min=1, soft_max=25)
    kernel_h: bpy.props.IntProperty(name='Kernel height', default=5, min=1, soft_max=25)
    kernel_w: bpy.props.IntProperty(name='Kernel width', default=5, min=1, soft_max=25)
    group: bpy.props.IntProperty(name='Group size', default=1, min=1, soft_max=25)
    square_stride: bpy.props.BoolProperty(name='Equal x,y stride', default=True)
    stride: bpy.props.IntProperty(name='Stride', default=1, min=1, soft_max=5)
    stride_h: bpy.props.IntProperty(name='Stride height', default=1, min=1, soft_max=5)
    stride_w: bpy.props.IntProperty(name='Stride width', default=1, min=1, soft_max=5)
    weight_filler: bpy.props.PointerProperty(type=filler_p_g)
    bias_filler: bpy.props.PointerProperty(type=filler_p_g)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input image')
        self.outputs.new('OutputSocketType', 'Output image')
        self.color = [1, 0, 1]
        self.use_custom_color = True

    def draw_buttons(self, context, layout):
        layout.prop(self, 'num_output')
        layout.prop(self, 'bias_term')
        layout.prop(self, 'group')
        layout.prop(self, 'square_padding')
        if self.square_padding:
            layout.prop(self, 'pad')
        else:
            layout.prop(self, 'pad_h')
            layout.prop(self, 'pad_w')
        layout.prop(self, 'square_kernel')
        if self.square_kernel:
            layout.prop(self, 'kernel_size')
        else:
            layout.prop(self, 'kernel_h')
            layout.prop(self, 'kernel_w')
        layout.prop(self, 'square_stride')
        if self.square_stride:
            layout.prop(self, 'stride')
        else:
            layout.prop(self, 'stride_h')
            layout.prop(self, 'stride_w')
        layout.prop(self, 'use_custom_weight')
        if self.use_custom_weight:
            layout.prop(self, 'custom_weight')
        else:
            layout.label(text='Weight Filler')
            self.weight_filler.draw(context, layout)
            if self.bias_term:
                layout.label(text='bias Filler')
                self.bias_filler.draw(context, layout)
        self.draw_extra_params(context, layout)

class FCNode(Node, CaffeTreeNode):
    """An inner product node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'FCNodeType'
    bl_label = 'Fully connected Node'
    bl_icon = 'SOUND'
    n_type = 'InnerProduct'
    num_output: bpy.props.IntProperty(name='Number of outputs', default=10, min=1)
    bias_term: bpy.props.BoolProperty(name='Include Bias term', default=True)
    weight_filler: bpy.props.PointerProperty(type=filler_p_g)
    bias_filler: bpy.props.PointerProperty(type=filler_p_g)
    axis: bpy.props.IntProperty(name='Starting axis', default=1)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input image')
        self.outputs.new('OutputSocketType', 'Output Activations')
        self.color = [1, 0, 0]
        self.use_custom_color = True

    def draw_buttons(self, context, layout):
        layout.prop(self, 'num_output')
        layout.prop(self, 'bias_term')
        layout.prop(self, 'axis')
        layout.label(text='Weight Filler')
        self.weight_filler.draw(context, layout)
        layout.label(text='bias Filler')
        self.bias_filler.draw(context, layout)
        self.draw_extra_params(context, layout)

class FlattenNode(Node, CaffeTreeNode):
    """Flatten layer node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'FlattenNodeType'
    bl_label = 'Flatten Node'
    bl_icon = 'SOUND'
    n_type = 'Flatten'

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input image')
        self.outputs.new('OutputSocketType', 'Flat output')

    def draw_buttons(self, context, layout):
        layout.label(text='Flatten')
        self.draw_extra_params(context, layout)

class AbsValNode(Node, CaffeTreeNode):
    """AbsVal layer node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'AbsValNodeType'
    bl_label = 'AbsVal Node'
    bl_icon = 'SOUND'
    n_type = 'AbsVal'

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input image')
        self.outputs.new('OutputSocketType', 'AbsVal output')

    def draw_buttons(self, context, layout):
        layout.label(text='AbsVal')

class SilenceNode(Node, CaffeTreeNode):
    """A Silence node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'SilenceNodeType'
    bl_label = 'Silence Node'
    bl_icon = 'SOUND'
    n_type = 'Silence'

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input')

    def draw_buttons(self, context, layout):
        layout.label(text='Silence')

class LRNNode(Node, CaffeTreeNode):
    """A Convolution node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'LRNNodeType'
    bl_label = 'LRN Node'
    bl_icon = 'SOUND'
    n_type = 'LRN'
    modes = [('ACROSS_CHANNELS', 'ACROSS_CHANNELS', 'Go across Channels'), ('WITHIN_CHANNEL', 'WITHIN_CHANNEL', 'Go by location')]
    alpha: bpy.props.FloatProperty(default=1, min=0, soft_max=50)
    beta: bpy.props.FloatProperty(default=5, min=0, soft_max=50)
    size: bpy.props.IntProperty(default=5, min=1, soft_max=50)
    mode: bpy.props.EnumProperty(name='Mode', default='ACROSS_CHANNELS', items=modes)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input image')
        self.outputs.new('OutputSocketType', 'Normalized output')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'alpha')
        layout.prop(self, 'beta')
        layout.prop(self, 'size')
        layout.prop(self, 'mode')
        self.draw_extra_params(context, layout)

class ActivationNode(Node, CaffeTreeNode):
    """A Convolution node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'AcNodeType'
    bl_label = 'Activation Node'
    bl_icon = 'SOUND'
    modes = [('"Sigmoid"', 'Sigmoid', 'Sigmoid'), ('"TanH"', 'TanH', 'TanH')]
    mode: bpy.props.EnumProperty(name='Mode', default='"Sigmoid"', items=modes)

    def init(self, context):
        self.inputs.new('NAFlatSocketType', 'Linear input')
        self.outputs.new('OutputSocketType', 'Non Linear output')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'mode')
        self.draw_extra_params(context, layout)

class ReLuNode(Node, CaffeTreeNode):
    """A ReLU node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'ReluNodeType'
    bl_label = 'ReLu Node'
    bl_icon = 'SOUND'
    n_type = 'ReLU'
    negative_slope: bpy.props.FloatProperty(default=0)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input image')
        self.outputs.new('OutputSocketType', 'Rectified output')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'negative_slope')
        self.draw_extra_params(context, layout)

class PReLuNode(Node, CaffeTreeNode):
    """A PReLU node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'PReluNodeType'
    bl_label = 'PReLu Node'
    bl_icon = 'SOUND'
    n_type = 'PReLU'
    channel_shared: bpy.props.BoolProperty(default=False)
    filler: bpy.props.PointerProperty(type=filler_p_g)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input image')
        self.outputs.new('OutputSocketType', 'Rectified output')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'channel_shared')
        self.filler.draw(context, layout)
        self.draw_extra_params(context, layout)

class SMLossNode(Node, CaffeTreeNode):
    """A Convolution node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'SMLossNodeType'
    bl_label = 'Softmax Loss Node'
    bl_icon = 'SOUND'
    n_type = 'SoftmaxWithLoss'
    w: bpy.props.FloatProperty(default=1)

    def init(self, context):
        self.inputs.new('NAFlatSocketType', 'Input Probabilities')
        self.inputs.new('LabelSocketType', 'Input Label')
        self.outputs.new('OutputSocketType', 'Loss output')

    def draw_buttons(self, context, layout):
        layout.label(text='Softmax Loss')
        layout.prop(self, 'w')
        self.draw_extra_params(context, layout)

class SCELossNode(Node, CaffeTreeNode):
    """A Convolution node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'SCELossNodeType'
    bl_label = 'SCE Loss Node'
    bl_icon = 'SOUND'
    n_type = 'SigmoidCrossEntropyLoss'
    w: bpy.props.FloatProperty(default=0)

    def init(self, context):
        self.inputs.new('NAFlatSocketType', 'Input values')
        self.inputs.new('AFlatSocketType', 'Input values 2')
        self.outputs.new('OutputSocketType', 'Loss output')

    def draw_buttons(self, context, layout):
        layout.label(text='SCE Loss')
        layout.prop(self, 'w')
        self.draw_extra_params(context, layout)

class EULossNode(Node, CaffeTreeNode):
    """A Convolution node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'EULossNodeType'
    bl_label = 'EU Loss Node'
    bl_icon = 'SOUND'
    n_type = 'EuclideanLoss'
    w: bpy.props.FloatProperty(default=0)

    def init(self, context):
        self.inputs.new('AFlatSocketType', 'Input values')
        self.inputs.new('AFlatSocketType', 'Input values 2')
        self.outputs.new('OutputSocketType', 'Loss output')

    def draw_buttons(self, context, layout):
        layout.label(text='EU Loss')
        layout.prop(self, 'w')
        self.draw_extra_params(context, layout)

class DropoutNode(Node, CaffeTreeNode):
    """A Convolution node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'DropoutNodeType'
    bl_label = 'Dropout Node'
    bl_icon = 'SOUND'
    n_type = 'Dropout'
    dropout_ratio: bpy.props.FloatProperty(default=0.5, min=0, max=1)

    def init(self, context):
        self.inputs.new('NAFlatSocketType', 'Input image')
        self.outputs.new('OutputSocketType', 'Output image')

    def draw_buttons(self, context, layout):
        layout.label(text='Dropout factor')
        layout.prop(self, 'dropout_ratio')
        self.draw_extra_params(context, layout)

class ConcatNode(Node, CaffeTreeNode):
    """A Concatination node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'ConcatNodeType'
    bl_label = 'Concatanation Node'
    bl_icon = 'SOUND'
    n_type = 'Concat'

    def update_bottoms(self, context):
        while len(self.inputs) < self.input_amount:
            self.inputs.new('ImageSocketType', 'Input%i' % (len(self.inputs) + 1))
        while len(self.inputs) > self.input_amount:
            self.inputs.remove(self.inputs[len(self.inputs) - 1])
    input_amount: bpy.props.IntProperty(min=1, max=MAX_SOCKETS, default=2, update=update_bottoms)
    axis: bpy.props.IntProperty(default=1)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input1')
        self.inputs.new('ImageSocketType', 'Input2')
        self.outputs.new('OutputSocketType', 'Output image')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'input_amount')
        layout.prop(self, 'axis')
        self.draw_extra_params(context, layout)

class AccuracyNode(Node, CaffeTreeNode):
    """A Convolution node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'AccuracyNodeType'
    bl_label = 'Accuracy Node'
    bl_icon = 'SOUND'
    n_type = 'Accuracy'
    Testonly: bpy.props.BoolProperty(default=True)

    def init(self, context):
        self.inputs.new('NAFlatSocketType', 'Input class')
        self.inputs.new('LabelSocketType', 'Input label')
        self.outputs.new('OutputSocketType', 'Output Accuracy')
        self.include_in = 'TEST'

    def draw_buttons(self, context, layout):
        # Testonly is retained in RNA for legacy graphs; include_in controls export.
        self.draw_include_in(layout)
        self.draw_extra_params(context, layout)

class ArgMaxNode(Node, CaffeTreeNode):
    """Arg Max Node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'ArgMaxNodeType'
    bl_label = 'Arg Max Node'
    bl_icon = 'SOUND'
    n_type = 'ArgMax'
    OutMaxVal: bpy.props.BoolProperty(name='Output max value', default=False)
    TopK: bpy.props.IntProperty(name='Top k', default=1, min=1, soft_max=200)

    def init(self, context):
        self.inputs.new('LossSocketType', 'Input loss')
        self.outputs.new('OutputSocketType', 'Output Arg Max')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'OutMaxVal')
        layout.prop(self, 'TopK')
        self.draw_extra_params(context, layout)

class HDF5OutputNode(Node, CaffeTreeNode):
    """HDF5 Output Node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'HDF5OutputNodeType'
    bl_label = 'HDF 5 Output Node'
    bl_icon = 'SOUND'
    n_type = 'HDF5Output'
    filename: bpy.props.StringProperty(name='HDF5 output File', default='', description='The path to the data file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input Data')
        self.inputs.new('ImageSocketType', 'Input Label')
        self.include_in = 'TRAIN'

    def draw_buttons(self, context, layout):
        layout.prop(self, 'filename')
        self.draw_include_in(layout)

class LogNode(Node, CaffeTreeNode):
    """Log Node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'LogNodeType'
    bl_label = 'Log Node'
    bl_icon = 'SOUND'
    n_type = 'Log'
    scale: bpy.props.FloatProperty(name='Scale', default=1, min=0, soft_max=200)
    shift: bpy.props.FloatProperty(name='Shift', default=0, soft_min=-200, soft_max=200)
    base: bpy.props.FloatProperty(name='base', default=-1, min=-1, soft_max=200)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input data')
        self.outputs.new('OutputSocketType', 'Output data')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'scale')
        layout.prop(self, 'shift')
        layout.prop(self, 'base')
        self.draw_extra_params(context, layout)

class PowerNode(Node, CaffeTreeNode):
    """Power Node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'PowerNodeType'
    bl_label = 'Power Node'
    bl_icon = 'SOUND'
    n_type = 'Power'
    power: bpy.props.FloatProperty(name='Power', default=1)
    scale: bpy.props.FloatProperty(name='Scale', default=1)
    shift: bpy.props.FloatProperty(name='Shift', default=0)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input data')
        self.outputs.new('OutputSocketType', 'Output data')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'power')
        layout.prop(self, 'scale')
        layout.prop(self, 'shift')

class ReductionNode(Node, CaffeTreeNode):
    """Reduction Node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'ReductionNodeType'
    bl_label = 'Reduction Node'
    bl_icon = 'SOUND'
    n_type = 'Reduction'
    ops = [('SUM', 'SUM', 'Sum'), ('ASUM', 'ASUM', 'Absolute Sum'), ('SUMSQ', 'SUMSQ', 'Sum of squares'), ('MEAN', 'MEAN', 'Mean')]
    operation: bpy.props.EnumProperty(name='Operation', default='SUM', items=ops)
    axis: bpy.props.IntProperty(name='Axis', default=0)
    coeff: bpy.props.FloatProperty(name='Coeff', default=1)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input data')
        self.outputs.new('OutputSocketType', 'Output data')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'operation')
        layout.prop(self, 'axis')
        layout.prop(self, 'coeff')
        self.draw_extra_params(context, layout)

class slice_point_p_g(bpy.types.PropertyGroup):
    slice_point: bpy.props.IntProperty(min=0)

    def draw(self, context, layout):
        layout.prop(self, 'slice_point')

class SliceNode(Node, CaffeTreeNode):
    """Slice Node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'SliceNodeType'
    bl_label = 'Slice Node'
    bl_icon = 'SOUND'
    n_type = 'Slice'
    axis: bpy.props.IntProperty(name='Axis', default=0)
    slice_points: bpy.props.CollectionProperty(type=slice_point_p_g)

    def update_slices(self, context):
        while len(self.slice_points) < self.num_of_slices:
            self.slice_points.add()
        while len(self.slice_points) > self.num_of_slices:
            self.slice_points.remove(len(self.slice_points) - 1)
        while len(self.outputs) < self.num_of_slices + 1:
            self.outputs.new('OutputSocketType', 'Out%i' % (len(self.outputs) + 1))
        while len(self.outputs) > self.num_of_slices + 1:
            self.outputs.remove(self.outputs[len(self.outputs) - 1])
    num_of_slices: bpy.props.IntProperty(default=1, min=1, max=MAX_SOCKETS, update=update_slices)

    def init(self, context):
        self.inputs.new('ImageSocketType', 'Input data')
        self.outputs.new('OutputSocketType', 'Out1')
        self.outputs.new('OutputSocketType', 'Out2')
        self.slice_points.add()

    def draw_buttons(self, context, layout):
        layout.prop(self, 'axis')
        layout.prop(self, 'num_of_slices')
        for slice_point in self.slice_points:
            slice_point.draw(context, layout)
        self.draw_extra_params(context, layout)

class SolverNode(Node, CaffeTreeNode):
    """A Solver node"""
    extra_params: bpy.props.BoolProperty(name='Extra Parameters', default=False)
    weight_params: bpy.props.PointerProperty(type=params_p_g)
    bias_params: bpy.props.PointerProperty(type=params_p_g)
    include_in: bpy.props.EnumProperty(items=CaffeTreeNode.phases, default='BOTH')
    use_custom_weight: bpy.props.BoolProperty(name='Use custom weights', default=False)
    custom_weight: bpy.props.StringProperty(name='Custom weights', default='', description='Custom weights and bias from file', subtype='FILE_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    bl_idname = 'SolverNodeType'
    bl_label = 'Solver Node'
    bl_icon = 'SOUND'
    lr_policies = [('fixed', 'fixed', 'Fixed'), ('step', 'step', 'Step'), ('exp', 'exp', 'Exponential'), ('inv', 'inv', 'Inverse'), ('multistep', 'multistep', 'Multi Step'), ('poly', 'poly', 'Polinomial'), ('sigmoid', 'sigmoid', 'Sigmoid')]
    regularization_types = [('L1', 'L1', 'L1'), ('L2', 'L2', 'L2')]
    gputoggles = []
    gpunum = 1
    gpufailed = 0
    if gpunum == 'Error':
        gpufailed = 1
    else:
        for gpu in range(gpunum):
            gpu_name = '%i' % gpu
            gputoggles.append((gpu_name, gpu_name, gpu_name))
    solvername: bpy.props.StringProperty()
    test_iter: bpy.props.IntProperty(name='Test Iterations', default=100, min=1, description='How many forward passes the test should carry out')
    test_interval: bpy.props.IntProperty(name='Test Interval', default=500, min=1, description='Carry out testing every test interval iterations')
    test_compute_loss: bpy.props.BoolProperty(name='Test Compute Loss', default=False, description='Compute loss in testing')
    test_initialization: bpy.props.BoolProperty(name='Test Initialization', default=True, description='run an initial test pass before the first iteration, ensuring memory availability and printing the starting value of the loss.')
    base_lr: bpy.props.FloatProperty(name='Base Learning rate', default=0.01, min=0)
    display: bpy.props.IntProperty(name='Display', default=100, min=0, description='The number of iterations between displaying info. If display = 0, no info will be displayed')
    average_loss: bpy.props.IntProperty(name='Average Loss', default=1, min=1, description='Display the loss averaged over the last average_loss iterations')
    max_iter: bpy.props.IntProperty(name='Maximum Iterations', default=50000, min=1)
    iter_size: bpy.props.IntProperty(name='Iteration Size', default=1, min=1, description='Accumulate gradients over iter_size x batch_size instances')
    lr_policy: bpy.props.EnumProperty(name='Learning rate Policy', items=lr_policies, default='step')
    gamma: bpy.props.FloatProperty(name='Gamma', default=0.1, min=0.0)
    power: bpy.props.FloatProperty(name='Power', default=0.75)
    momentum: bpy.props.FloatProperty(name='Momentum', default=0.9, min=0)
    weight_decay: bpy.props.FloatProperty(name='Weight Decay', default=0.0005, min=0)
    regularization_type: bpy.props.EnumProperty(name='Regularization type', items=regularization_types, default='L2')
    stepsize: bpy.props.IntProperty(name='Step size', default=5000, min=1)
    snapshot: bpy.props.IntProperty(name='Snapshot Interval', default=0, min=0, description='The snapshot interval. 0 for no snapshot')
    snapshot_prefix: bpy.props.StringProperty(name='Snapshot Path', default='', description='Give the path to the snapshot data', subtype='DIR_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    snapshot_diff: bpy.props.BoolProperty(name='Snapshot diff', default=False, description='Whether to snapshot diff in the results or not')
    solver_modes = [('GPU', 'GPU', 'GPU'), ('CPU', 'CPU', 'CPU')]
    solver_mode: bpy.props.EnumProperty(name='Solver mode', items=solver_modes, default='GPU')
    gpus: bpy.props.EnumProperty(name='GPU', description='GPU to use', items=gputoggles)
    use_random_seed: bpy.props.BoolProperty(name='Use Random seed', default=False)
    random_seed: bpy.props.IntProperty(name='Random seed', default=10, description='The seed with which the Solver will initialize the Caffe random number generator')
    solver_types = [('NESTEROV', 'NESTEROV', 'Nesterovs Accelerated Gradient'), ('ADAGRAD', 'ADAGRAD', 'Adaptive gradient descent'), ('SGD', 'SGD', 'Stochastic Gradient Descent')]
    solver_type: bpy.props.EnumProperty(name='Solver type', items=solver_types, default='SGD')
    delta: bpy.props.FloatProperty(name='Delta', default=1e-08, min=0, description='Numerical stability for AdaGrad')
    debug_info: bpy.props.BoolProperty(name='Debug info', default=False)
    snapshot_after_train: bpy.props.BoolProperty(name='Snapshot after train', default=True, description="If false, don't save a snapshot after training finishes")
    config_path: bpy.props.StringProperty(name='Configuration Data Path', default='', description='Give the path to the config data', subtype='DIR_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})
    caffe_exec: bpy.props.StringProperty(name='Caffe Tools Folder', default='', description='Give the path to the caffe executable', subtype='DIR_PATH', options={'ANIMATABLE', 'PATH_SUPPORTS_BLEND_RELATIVE'})

    def draw_buttons(self, context, layout):
        layout.prop(self, 'solvername')
        layout.prop(self, 'test_iter')
        layout.prop(self, 'test_interval')
        layout.prop(self, 'test_compute_loss')
        layout.prop(self, 'test_initialization')
        layout.prop(self, 'base_lr')
        layout.prop(self, 'display')
        layout.prop(self, 'average_loss')
        layout.prop(self, 'max_iter')
        layout.prop(self, 'iter_size')
        layout.prop(self, 'lr_policy')
        if self.lr_policy == 'step':
            layout.prop(self, 'gamma')
            layout.prop(self, 'stepsize')
        elif self.lr_policy == 'exp':
            layout.prop(self, 'gamma')
        elif self.lr_policy == 'inv':
            layout.prop(self, 'gamma')
            layout.prop(self, 'power')
        elif self.lr_policy == 'multistep':
            layout.label(text='NOT IMPLEMENTED', icon='ERROR')
        elif self.lr_policy == 'poly':
            layout.prop(self, 'power')
        elif self.lr_policy == 'sigmoid':
            layout.prop(self, 'gamma')
            layout.prop(self, 'stepsize')
        layout.prop(self, 'momentum')
        layout.prop(self, 'weight_decay')
        layout.prop(self, 'regularization_type')
        layout.prop(self, 'snapshot')
        layout.prop(self, 'snapshot_prefix')
        layout.prop(self, 'snapshot_diff')
        layout.prop(self, 'snapshot_after_train')
        layout.prop(self, 'solver_mode')
        if self.solver_mode == 'GPU':
            if self.gpufailed:
                layout.label(text='WARNING: GPU NOT DETECTED', icon='ERROR')
                layout.label(text="Check 'nvidia-smi' command can be run", icon='ERROR')
            else:
                layout.prop(self, 'gpus')
        layout.prop(self, 'use_random_seed')
        if self.use_random_seed:
            layout.prop(self, 'random_seed')
        layout.prop(self, 'solver_type')
        if self.solver_type == 'ADAGRAD':
            layout.prop(self, 'delta')
        layout.prop(self, 'debug_info')
        layout.prop(self, 'config_path')
        layout.prop(self, 'caffe_exec')

CLASSES = (filler_p_g, params_p_g, eltwise_coeff_p_g, slice_point_p_g, ImageSocket, OutputSocket, LabelSocket, LossSocket, AFlatSocket, NAFlatSocket, CaffeTree, DataNode, PoolNode, PaddingNode, EltwiseNode, ExpNode, MVNNode, ConvNode, DeConvNode, FCNode, FlattenNode, AbsValNode, SilenceNode, LRNNode, ActivationNode, ReLuNode, PReLuNode, SMLossNode, SCELossNode, EULossNode, DropoutNode, ConcatNode, AccuracyNode, ArgMaxNode, HDF5OutputNode, LogNode, PowerNode, ReductionNode, SliceNode, SolverNode,)

def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
