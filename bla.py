import torch
from torch import nn
import onnx
import onnxruntime as ort
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, axes):
        return torch.min(x, axes.item(), True)[0]

def make_model():
    return Model()
    # return nn.Sequential(torch.min(nn.ReLU()))

def run_model():
    x = torch.tensor([[-1, 5], [0, 13]])
    print(x)
    model = make_model()
    y = model(x)
    print(y)

def export_model():
    model = make_model()
    model.train(False)
    x = torch.tensor([[-1, 5], [0, 13]])
    torch.onnx.export(model, (x, 0), "bla.onnx", opset_version=18)

def import_model():
    ort_session = ort.InferenceSession("bla.onnx")
    out = ort_session.run(None, {"data": np.array([[-1, 5], [0, 13]])})
    print(out)

def reduce_min_13():
    keepdims = 1
    node = onnx.helper.make_node("ReduceMin", inputs=["data"], outputs=["reduced"], keepdims=keepdims, axes=[0])
    print(node)
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.INT64, [2, 2])
    output = onnx.helper.make_tensor_value_info("reduced", onnx.TensorProto.INT64, [None, None])
    graph = onnx.helper.make_graph([node],"reduce_min", [data], [output])
    print(graph)
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, "mlem.onnx")

def reduce_min_18():
    keepdims = 1
    node = onnx.helper.make_node("ReduceMin", inputs=["data", "axes"], outputs=["reduced"], keepdims=keepdims)
    print(node)
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.INT64, [2, 2])
    axes = onnx.helper.make_tensor_value_info("axes", onnx.TensorProto.INT64, [1])
    output = onnx.helper.make_tensor_value_info("reduced", onnx.TensorProto.INT64, [None, None])
    graph = onnx.helper.make_graph([node],"reduce_min", [data, axes], [output], initializer = [onnx.helper.make_tensor("axes", onnx.TensorProto.INT64, [1], [0])])
    print(graph)
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, "reducemin_18.onnx")

def test_reduce_min_18():
    ort_session = ort.InferenceSession("reducemin_18.onnx")
    out = ort_session.run(None, {"data": np.array([[-1, 5], [0, 13]])})
    print(out)

if __name__=="__main__":
    test_reduce_min_18()