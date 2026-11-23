"""Verify the selected ONNX face did not change the inherited computation."""
import copy
import sys
import onnx

a,b=(onnx.load(path,load_external_data=False) for path in sys.argv[1:])
ga,gb=a.graph,b.graph
assert len(gb.node)==len(ga.node)+1
assert len(gb.initializer)==len(ga.initializer)+1
for x,y in zip(ga.node[:-1],gb.node[:-2]):
    assert x==y
for x,y in zip(ga.initializer,gb.initializer[:-1]):
    assert x==y
assert ga.input==gb.input and ga.output[1:]==gb.output[1:]
assert a.opset_import==b.opset_import and a.ir_version==b.ir_version
t=copy.deepcopy(gb.node[-1])
assert t.input[0]=="idol_last_hidden"
t.input[0]=ga.node[-1].input[0]
assert t==ga.node[-1]
new=gb.node[-2]
assert new.op_type=="Gather" and list(new.output)==["idol_last_hidden"]
assert list(new.input)==[ga.node[-1].input[0],"idol_last_hidden_index"]
assert len(new.attribute)==1 and new.attribute[0].name=="axis" and new.attribute[0].i==2
g=gb.initializer[-1]
assert g.name=="idol_last_hidden_index" and onnx.numpy_helper.to_array(g).tolist()==[-1]
out=copy.deepcopy(gb.output[0])
d=out.type.tensor_type.shape.dim[2]
assert d.dim_value==1
d.dim_param=ga.output[0].type.tensor_type.shape.dim[2].dim_param
assert out==ga.output[0]
print(f"verified {len(ga.node)-1} identical upstream nodes and {len(ga.initializer)} unchanged initializers")
