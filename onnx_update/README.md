# ONNX 1.14.0

The project uses `onnx==1.10.2` currently.

Updating it to `onnx==1.14.0` will require multiple changes.

## Python tests

Bumping the onnx version will pass python backend test with the [these changes](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/commit/19b8d94a9ee96dfd992104e5825e29e03abd85c6).

*Note: [onnx_backend_test.py](../test/py/onnx_backend_test.py) only uses a subset of all onnx operator tests. The full list is [here](./full_test_list.md) and the missing test list is [here](./missing_test_list.md).*

## ONNX file generation

MIGraphX generated actual onnx files. It uses [gen_onnx.py](../test/onnx/gen_onnx.py) script.

The generation will fail the following tests because of the onnx api change:
- onehot_test()
- reducel1_dyn_noaxes_test()
- reducel1_dyn_test()
- reducemax_dyn_test()
- slice_reverse_dyn_test()
- slice_step_dyn_test()
- spacetodepth_test()
- transpose_gather_test()

The generated models will be checked with [onnx_test](../test/onnx/onnx_test.cpp) and [verify_onnx](../test/onnx/verify_onnx.cpp) tests.

`onnx_test` failures:
- add_fp16_test
- gather_test
- if_param_test
- if_then_else_multi_output_shapes_inlined_test
- if_then_else_multi_output_shapes_test
- if_then_test
- if_then_test_inlined
- reducemax_test
- reducemin_test
- softmax_test

`verify_onnx` failures
- if_else_test
- if_else_test_inlined

The full log is [here](./test_result.txt).

*Note: When the remaining models are compiled, those will also fail these tests*
