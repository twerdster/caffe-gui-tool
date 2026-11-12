"""Blender interface for composing and exporting neural operators."""
from pathlib import Path
import os
import shutil
import tempfile
import bpy
from bpy.props import StringProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper
from . import caffe_nodes, caffe_export, graph

EXAMPLES = Path(__file__).parent / 'examples'

def active_tree(context):
    tree = getattr(context.space_data, 'edit_tree', None)
    if tree is None or tree.bl_idname != 'CaffeNodeTree':
        raise ValueError('Open a Neural Operator Composer node tree first')
    return tree

def show_tree(context, tree):
    space = context.space_data
    space.tree_type = 'CaffeNodeTree'
    space.pin = True
    space.node_tree = tree

def _write_export(directory, files):
    """Stage all files first, then replace with rollback on a write failure.

    Individual replacements are atomic. A failed replacement restores the prior
    complete export; unrelated files in the selected directory are preserved.
    """
    existed = directory.exists()
    directory.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.noc-export-', dir=directory))
    preserve_backups = False
    try:
        backups = {}
        replaced = []
        for filename, content in files.items():
            target = directory / filename
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError('Export target must be a regular file: ' + str(target))
            staged = staging / filename
            staged.write_text(content, encoding='utf-8')
            staged.chmod(0o755 if filename.endswith('.sh') else 0o644)
            if target.exists():
                backup = staging / (filename + '.previous')
                shutil.copy2(target, backup)
                backups[filename] = backup
        try:
            for filename in files:
                os.replace(staging / filename, directory / filename)
                replaced.append(filename)
        except BaseException as error:
            failures = []
            for filename in reversed(replaced):
                target = directory / filename
                try:
                    if filename in backups:
                        os.replace(backups[filename], target)
                    else:
                        target.unlink()
                except OSError as failure:
                    failures.append(str(failure))
            if failures:
                preserve_backups = True
                raise RuntimeError('Export rollback failed; previous files are preserved in ' + str(staging) + ': ' + '; '.join(failures)) from error
            raise
    finally:
        if not preserve_backups:
            shutil.rmtree(staging)
            if not existed:
                try:
                    directory.rmdir()
                except OSError:
                    pass


def export_caffe(tree, directory):
    summary = graph.validate(tree)
    directory = Path(bpy.path.abspath(str(directory))).resolve()
    nodes = graph.topological(tree) + [node for node in tree.nodes if node.bl_idname == 'SolverNodeType']
    weight_paths = {}
    for node in nodes:
        if getattr(node, 'use_custom_weight', False):
            path = Path(node.custom_weight)
            if not path.is_absolute() and not node.custom_weight.startswith('//'):
                path = EXAMPLES / path
            else:
                path = Path(bpy.path.abspath(node.custom_weight))
            if not path.is_file():
                raise ValueError('Missing custom operator weights: ' + node.name)
            weight_paths[node.name] = path
    files = caffe_export.generate(nodes, config_path=str(directory), custom_weight_paths=weight_paths)
    _write_export(directory, files)
    return summary

class NOC_OT_new(bpy.types.Operator):
    bl_idname = 'noc.new'
    bl_label = 'New Neural Graph'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        tree = bpy.data.node_groups.new('Neural Operator', 'CaffeNodeTree')
        tree.use_fake_user = True
        show_tree(context, tree)
        return {'FINISHED'}

class NOC_OT_import(bpy.types.Operator, ImportHelper):
    bl_idname = 'noc.import_graph'
    bl_label = 'Import Neural Graph'
    filename_ext = '.json'
    filter_glob: StringProperty(default='*.json', options={'HIDDEN'})
    def execute(self, context):
        try:
            show_tree(context, graph.load(self.filepath))
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}

class NOC_OT_export(bpy.types.Operator, ExportHelper):
    bl_idname = 'noc.export_graph'
    bl_label = 'Export Neural Graph JSON'
    filename_ext = '.json'
    filter_glob: StringProperty(default='*.json', options={'HIDDEN'})
    def execute(self, context):
        try:
            graph.save(active_tree(context), self.filepath)
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}

class NOC_OT_caffe(bpy.types.Operator):
    bl_idname = 'noc.export_caffe'
    bl_label = 'Export Caffe Solution'
    directory: StringProperty(subtype='DIR_PATH', options={'PATH_SUPPORTS_BLEND_RELATIVE'})
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    def execute(self, context):
        try:
            summary = export_caffe(active_tree(context), self.directory)
            self.report({'INFO'}, f"Exported {summary['nodes']} nodes")
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}

class NOC_OT_validate(bpy.types.Operator):
    bl_idname = 'noc.validate'
    bl_label = 'Validate Graph'
    def execute(self, context):
        try:
            summary = graph.validate(active_tree(context))
            self.report({'INFO'}, f"Valid graph: {summary['nodes']} nodes, {summary['links']} links")
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}

class NOC_OT_example(bpy.types.Operator):
    bl_idname = 'noc.example'
    bl_label = 'Load Patch-Beltrami Example'
    bl_options = {'REGISTER', 'UNDO'}
    example: EnumProperty(items=[('patch_beltrami_bw', 'Patch-Beltrami Grayscale', ''), ('patch_beltrami_rgb', 'Patch-Beltrami Color', ''), ('patch_beltrami_multifilter', 'Patch-Beltrami Five Stages', ''), ('operator_demo', 'Compact Operator Demo', '')])
    def execute(self, context):
        try:
            show_tree(context, graph.load(EXAMPLES / (self.example + '.json')))
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}

class NOC_MT_nodes(bpy.types.Menu):
    bl_idname = 'NOC_MT_nodes'
    bl_label = 'Neural Operators'
    def draw(self, context):
        for cls in caffe_nodes.CLASSES:
            if issubclass(cls, bpy.types.Node):
                op = self.layout.operator('node.add_node', text=cls.bl_label)
                op.type = cls.bl_idname
                op.use_transform = True

class NOC_PT_composer(bpy.types.Panel):
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Composer'
    bl_label = 'Neural Operator Composer'
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == 'CaffeNodeTree'
    def draw(self, context):
        layout = self.layout
        layout.operator('noc.new', icon='ADD')
        layout.operator('noc.import_graph', icon='IMPORT')
        layout.operator('noc.export_graph', icon='EXPORT')
        layout.separator()
        for key, title in [('operator_demo', 'Compact Operator Demo'), ('patch_beltrami_bw', 'Patch-Beltrami Grayscale'), ('patch_beltrami_rgb', 'Patch-Beltrami Color'), ('patch_beltrami_multifilter', 'Five-Stage Network')]:
            layout.operator('noc.example', text=title).example = key
        layout.separator()
        layout.operator('noc.validate', icon='CHECKMARK')
        layout.operator('noc.export_caffe', icon='FILE_TICK')
        tree = getattr(context.space_data, 'edit_tree', None)
        if tree:
            layout.label(text=f'{len(tree.nodes)} nodes / {len(tree.links)} links')
        layout.label(text='Caffe execution uses an external installation.')

def draw_add(self, context):
    if getattr(context.space_data, 'tree_type', None) == 'CaffeNodeTree':
        self.layout.menu('NOC_MT_nodes')

CLASSES = (NOC_OT_new, NOC_OT_import, NOC_OT_export, NOC_OT_caffe, NOC_OT_validate, NOC_OT_example, NOC_MT_nodes, NOC_PT_composer)

def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.NODE_MT_add.append(draw_add)

def unregister():
    bpy.types.NODE_MT_add.remove(draw_add)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
