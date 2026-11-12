"""Create the compact learned differential-operator example in background Blender."""
from pathlib import Path
import sys
import bpy
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import neural_operator_blender as addon
from neural_operator_blender import graph, composer
addon.register()
tree = bpy.data.node_groups.new('Learned Beltrami Operator', 'CaffeNodeTree')
def node(kind,name,x,y,color):
    n=tree.nodes.new(kind)
    n.name=name
    n.label=name
    n.location=(x,y)
    n.width=180
    n.hide=True
    n.use_custom_color=True
    n.color=color
    for i,socket in enumerate(n.outputs): socket.output_name=name.lower().replace(' ','_')+'_'+str(i)
    return n
def link(a,b,i=0):tree.links.new(a.outputs[0],b.inputs[i])
blue=(.1,.42,.65);purple=(.45,.2,.65);mint=(.15,.55,.4);gold=(.65,.4,.12)
data=node('DataNodeType','Noisy input',0,0,blue)
gx=node('ConvNodeType','Learned Dx',240,80,purple)
gy=node('ConvNodeType','Learned Dy',240,-140,purple)
for n in [gx,gy]:n.num_output=16;n.kernel_size=7;n.pad=3;n.bias_term=False
x2=node('PowerNodeType','Dx squared',480,80,mint);x2.power=2
y2=node('PowerNodeType','Dy squared',480,-140,mint);y2.power=2
metric=node('EltwiseNodeType','Metric energy',720,0,mint)
inv=node('PowerNodeType','Inverse metric',960,0,mint);inv.shift=1;inv.power=-.5
fx=node('EltwiseNodeType','Flux x',1200,80,gold);fx.operation='PROD'
fy=node('EltwiseNodeType','Flux y',1200,-140,gold);fy.operation='PROD'
dx=node('ConvNodeType','Learned divergence x',1440,80,purple)
dy=node('ConvNodeType','Learned divergence y',1440,-140,purple)
for n in [dx,dy]:n.num_output=1;n.kernel_size=7;n.pad=3;n.bias_term=False
update=node('EltwiseNodeType','Flow update',1680,0,blue);update.input_amount=3
for c,value in zip(update.coeffs,[1,-.15,-.15]):c.coeff=value
solver=node('SolverNodeType','Training settings',1680,-260,gold);solver.solvername='learned_beltrami';solver.solver_mode='CPU';solver.base_lr=1e-4
for a,b,i in [(data,gx,0),(data,gy,0),(gx,x2,0),(gy,y2,0),(x2,metric,0),(y2,metric,1),(metric,inv,0),(gx,fx,0),(inv,fx,1),(gy,fy,0),(inv,fy,1),(fx,dx,0),(fy,dy,0),(data,update,0),(dx,update,1),(dy,update,2)]:link(a,b,i)
graph.validate(tree)
graph.save(tree,composer.EXAMPLES/'operator_demo.json')
print('Created compact operator example:',len(tree.nodes),'nodes',len(tree.links),'links')
