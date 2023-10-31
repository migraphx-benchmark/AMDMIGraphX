# from https://huggingface.co/eugenesiow/awsrn-bam#how-to-use

import torch
import onnx
from super_image import AwsrnModel
import requests

for scale in [2, 3, 4]: # scale 2, 3 and 4 models available
    model = AwsrnModel.from_pretrained('eugenesiow/awsrn-bam', scale=scale)
    torch_input = torch.randn(1, 3, 85, 85)

    onnx_name = f"awsrn_bam_scale{scale}.onnx"
    torch.onnx.export(model, torch_input, onnx_name, export_params=False, opset_version=17, input_names = ['input'], output_names = ['output'])

    onnx_model = onnx.load(onnx_name)
    onnx.checker.check_model(onnx_model)
