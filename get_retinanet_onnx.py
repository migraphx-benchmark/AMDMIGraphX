import torch
import onnx
import urllib
from PIL import Image
from torchvision import transforms

from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights

model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
model.eval()
torch_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, torch_input, "retinanet.onnx", export_params=False, opset_version=17, input_names = ['input'], output_names = ['output'])

onnx_model = onnx.load("retinanet.onnx")
onnx.checker.check_model(onnx_model)
