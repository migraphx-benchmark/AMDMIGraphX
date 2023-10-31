# from https://huggingface.co/eugenesiow/a2n#how-to-use

import torch
import onnx
from super_image import A2nModel

for scale in [2, 3, 4]: # scale 2, 3 and 4 models available
    model = A2nModel.from_pretrained('eugenesiow/a2n', scale=scale)
    torch_input = torch.randn(1, 3, 85, 85)

    onnx_name = f"a2n_scale{scale}.onnx"
    torch.onnx.export(model, torch_input, onnx_name, export_params=False, opset_version=17, input_names = ['input'], output_names = ['output'])

    onnx_model = onnx.load(onnx_name)
    onnx.checker.check_model(onnx_model)
