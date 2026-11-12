"""Neural Operator Composer for Blender."""
bl_info = {
    'name': 'Caffe GUI Tool / Neural Operator Composer',
    'author': 'Aaron Wetzler, Rom Hershkovitz; original Caffe GUI by H. Tomkins',
    'version': (2, 0, 0),
    'blender': (5, 1, 0),
    'location': 'Node Editor > Neural Operator Composer',
    'description': 'Compose neural differential operators and export Caffe computational graphs',
    'category': 'Node',
}
from . import caffe_nodes, caffe_export, composer

def register():
    caffe_nodes.register()
    composer.register()

def unregister():
    composer.unregister()
    caffe_nodes.unregister()
