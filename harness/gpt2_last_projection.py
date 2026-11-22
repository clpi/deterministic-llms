"""Build a last-position output face of the exact GPT-2 FP32 ONNX artifact."""
import argparse
import hashlib
from pathlib import Path

import onnx
from onnx import TensorProto, helper

PIN = "12fbb1ec2d2d70c8ebd21a290a348a4109447b98582af64c6f93b6526d5c8f35"


def identity(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def build(source,output):
    if identity(source)!=PIN:
        raise ValueError("unexpected model digest")
    if output.exists():
        raise FileExistsError(output)
    model = onnx.load(source,load_external_data=False)
    graph = model.graph
    final = graph.node[-1]
    if (final.name,final.op_type,list(final.input),list(final.output)) != (
        "MatMul_2533","MatMul",["3085","3272"],["output1"]
    ):
        raise ValueError("unsupported final head")
    values = {a for node in graph.node for a in (*node.input,*node.output)}
    last, indices = "idol_last_hidden", "idol_last_hidden_index"
    if last in values or indices in values or any("output1" in node.input for node in graph.node):
        raise ValueError("head projection is not a terminal output")
    gather = helper.make_node("Gather", ["3085",indices], [last], name="idol_last_hidden", axis=2)
    graph.initializer.append(helper.make_tensor(indices,TensorProto.INT64,[1],[-1]))
    final.input[0] = last
    graph.node.insert(len(graph.node)-1, gather)
    graph.output[0].type.tensor_type.shape.dim[2].dim_value = 1
    onnx.save(model,output)
    onnx.checker.check_model(str(output))
    print(identity(output))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    build(a.source,a.output)
