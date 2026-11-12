"""Legacy exporter entry point backed by the maintained implementation."""
import bpy
from neural_operator_blender.caffe_export import *
from neural_operator_blender import composer

bl_info = {'name': 'Caffe GUI legacy export shortcut', 'category': 'Node',
           'blender': (5, 1, 0), 'version': (2, 0, 0)}

class Solve(bpy.types.Operator):
    """Export the complete active neural graph using its Solver directory."""
    bl_idname = 'nodes.make_solver'
    bl_label = 'Create Caffe Solution'

    def execute(self, context):
        try:
            tree = composer.active_tree(context)
            solvers = [n for n in tree.nodes if n.bl_idname == 'SolverNodeType']
            if len(solvers) != 1:
                raise ValueError('Add exactly one Solver node before exporting')
            directory = solvers[0].config_path
            if not directory:
                raise ValueError('Choose an export directory on the Solver node')
            composer.export_caffe(tree, bpy.path.abspath(directory))
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}

def register():
    bpy.utils.register_class(Solve)

def unregister():
    bpy.utils.unregister_class(Solve)

if __name__ == '__main__':
    register()
