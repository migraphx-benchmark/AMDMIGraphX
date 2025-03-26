/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/stringutils.hpp>

TEST_CASE(mean_variance_norm_val_test)
{
    migraphx::program p = read_onnx("mean_variance_norm_val_test.onnx");

    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> result_vector(9);
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // example from: https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/meanvariancenormalization.py
    const std::vector<float> expected_result = 
    {
        1.35464, 0.330535, -1.54508, 
        -1.21068, -0.892595, 0.298881, 
        0.380831, 0.818088, 0.858656, 
        
        -1.10606, -0.0555288, -0.783103, 
        0.832814, -1.25028, 0.674679, 
        0.766937, 0.911387, -1.64636, 
        
        -0.234028, 1.60921, 0.429406, 
        1.29061, 1.18602, -0.929458, 
        0.0721332, -0.38174, -1.77993
    };
    EXPECT(migraphx::verify::verify_rms_range(result_vector, expected_result));    
}
