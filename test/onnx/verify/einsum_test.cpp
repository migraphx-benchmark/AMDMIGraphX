/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>

// TEST_CASE(einsum_transpose_missing_output_test)
// {
//     migraphx::program p = migraphx::parse_onnx("einsum_transpose_missing_output_test.onnx");

//     migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
//     std::cout << migraphx::shape{migraphx::shape::float_type, {2, 3, 4}} << std::endl;
//     std::vector<float> x_data{0, 1, 2, 3, 4, 5};

//     migraphx::parameter_map pm;
//     pm["x"] = migraphx::argument{x_shape, x_data.data()};

//     auto result = p.eval(pm).back();
//     std::vector<float> result_vector;
//     result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

//     std::vector<float> gold{0, 3, 1, 4, 2, 5};
//     EXPECT(result_vector == gold);
// }

// TEST_CASE(einsum_hadamard_product_test)
// {
//     migraphx::program p = migraphx::parse_onnx("einsum_hadamard_product_test.onnx");

//     migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
//     std::vector<float> x1_data{0, 1, 2, 3, 4, 5};
//     std::vector<float> x2_data{6, 7, 8, 9, 10, 11};

//     migraphx::parameter_map pm;
//     pm["x1"] = migraphx::argument{x_shape, x1_data.data()};
//     pm["x2"] = migraphx::argument{x_shape, x2_data.data()};

//     auto result = p.eval(pm).back();
//     std::vector<float> result_vector;
//     result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

//     std::vector<float> gold{0, 7, 16, 27, 40, 55};
//     EXPECT(result_vector == gold);
// }

TEST_CASE(einsum_ijkl_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_ijkl_test.onnx");

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x1_data{0, 1, 2, 3, 4, 5};
    std::vector<float> x2_data{6, 7, 8, 9, 10, 11};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{0,  0,  0,  0,  0,  0,  6,  7,  8,  9,  10, 11, 12, 14, 16, 18, 20, 22,
                            18, 21, 24, 27, 30, 33, 24, 28, 32, 36, 40, 44, 30, 35, 40, 45, 50, 55};
    EXPECT(result_vector == gold);
}