/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "migraphx/module.hpp"
#include <cstdint>
#include <migraphx/float8.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(matmulnbits_test)
{
    auto p = optimize_onnx("matmulnbits_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map pm;
    auto a_shape = migraphx::shape{migraphx::shape::float_type, {2, 16}};
    std::vector<float> a(a_shape.elements());
    std::iota(a.begin(), a.end(), 0);
    pm["a"] = migraphx::argument(a_shape, a.data());

    auto b_shape = migraphx::shape{migraphx::shape::uint8_type, {4, 1, 8}};
    std::vector<uint8_t> b(b_shape.elements(), 0x21);
    pm["b"] = migraphx::argument(b_shape, b.data());

    auto scales_shape = migraphx::shape{migraphx::shape::float_type, {4}};
    std::vector<float> scales{1, 2, 3, 4};
    pm["scales"] = migraphx::argument(scales_shape, scales.data());

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::cout << result.get_shape() << std::endl;
    std::cout << migraphx::to_string_range(result_vector) << std::endl;
    // for(auto i = 0; i < 4; ++i)
    // {
    //     for(auto j = 0; j < 16; ++j)
    //     {
    //         std::cout << static_cast<int>(result_vector[i * 16 + j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }
}
