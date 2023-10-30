import torch
import onnx
import urllib
from PIL import Image
from torchvision import transforms

model = torch.hub.load('pytorch/vision:v0.16.0', 'inception_v3', weights='Inception_V3_Weights.DEFAULT')
model.eval()
torch_input = torch.randn(1, 3, 299, 299)
torch.onnx.export(model, torch_input, "inception_v3.onnx", export_params=False, opset_version=17, input_names = ['input'], output_names = ['output'])

onnx_model = onnx.load("inception_v3.onnx")
onnx.checker.check_model(onnx_model)
