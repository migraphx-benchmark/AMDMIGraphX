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
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.INT64, [None, None])
    axes = onnx.helper.make_tensor_value_info("axes", onnx.TensorProto.INT64, [1])
    output = onnx.helper.make_tensor_value_info("reduced", onnx.TensorProto.INT64, [None, None])
    graph = onnx.helper.make_graph([node],"reduce_min", [data, axes], [output])
    print(graph)
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, "reducemin_dynamic_18.onnx")

def test_reduce_min_18():
    ort_session = ort.InferenceSession("reducemin_18.onnx")
    out = ort_session.run(None, {"data": np.array([[-1, 5], [0, 13]])})
    print(out)

def bitwise_not():
    shape = [2];
    node = onnx.helper.make_node("BitwiseNot", inputs=["x"], outputs=["y"])
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.INT64, shape)
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.INT64, shape)
    graph = onnx.helper.make_graph([node], "bitwise_not", [x], [y]);
    opset = onnx.helper.make_opsetid("", 18); 
    model = onnx.helper.make_model(graph, opset_imports=[opset])
    onnx.checker.check_model(model)
    onnx.save(model, "bitwise_not.onnx")
    ort_session = ort.InferenceSession("bitwise_not.onnx")
    out = ort_session.run(None, {"x": np.array([0, 2])})
    print(out)

def relu():
    shape = [2];
    node = onnx.helper.make_node("Relu", inputs=["x"], outputs=["y"])
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.INT64, shape)
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.INT64, shape)
    graph = onnx.helper.make_graph([node], "relu", [x], [y]);
    opset = onnx.helper.make_opsetid("", 14); 
    model = onnx.helper.make_model(graph, opset_imports=[opset])
    onnx.checker.check_model(model)
    onnx.save(model, "relu.onnx")
    ort_session = ort.InferenceSession("relu.onnx")
    out = ort_session.run(None, {"x": np.array([-2, 2])})
    print(out)

def sin():
    shape = [2];
    node = onnx.helper.make_node("Sin", inputs=["x"], outputs=["y"])
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.DOUBLE, shape)
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.DOUBLE, shape)
    graph = onnx.helper.make_graph([node], "sin", [x], [y]);
    opset = onnx.helper.make_opsetid("", 14); 
    model = onnx.helper.make_model(graph, opset_imports=[opset])
    onnx.checker.check_model(model)
    onnx.save(model, "sin.onnx")
    # ort_session = ort.InferenceSession("relu.onnx")
    # out = ort_session.run(None, {"x": np.array([-2, 2])})
    # print(out)

class ReLUModel(torch.nn.Module):
    def __init__(self):
        super(ReLUModel, self).__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)

def pytorch_code():
    model = ReLUModel()
    x = torch.tensor([-2, 2])
    y = model(x)
    print(y)
    torch.onnx.export(model, x, "relu.onnx", opset_version=14, input_names=["x"], output_names=["y"])
    ort_session = ort.InferenceSession("relu.onnx")
    y = ort_session.run(None, {"x": x})
    print(y)

if __name__=="__main__":
    reduce_min_18()