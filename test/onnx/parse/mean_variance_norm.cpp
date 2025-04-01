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

 #include <onnx_test.hpp>
 #include <onnx_test_utils.hpp>

TEST_CASE(mean_variance_norm_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    std::vector<size_t> dims{3, 4, 5, 6};
    migraphx::shape s1{migraphx::shape::float_type, dims};

    const float eps_default = 1e-7f;

    auto X          = mm->add_parameter("x", s1);

    auto E_X        = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), X);
    auto E_sqr_X    = add_common_op(*mm, migraphx::make_op("mul"), {E_X, E_X});
    auto X_sqr      = add_common_op(*mm, migraphx::make_op("mul"), {X, X});
    auto E_X_sqr    = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), X_sqr);
    auto std_sqr    = add_common_op(*mm, migraphx::make_op("sub"), {E_X_sqr, E_sqr_X});
    auto std        = add_common_op(*mm, migraphx::make_op("sqrt"), {std_sqr});
    auto numerator  = add_common_op(*mm, migraphx::make_op("sub"), {X, E_X});
    auto eps_literal = mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {eps_default}});
    auto denominator= add_common_op(*mm, migraphx::make_op("add"), {std, eps_literal});
    auto Y          = add_common_op(*mm, migraphx::make_op("div"), {numerator, denominator});

    mm->add_return({Y});

    migraphx::onnx_options options;
    auto prog = read_onnx("mean_variance_norm_test.onnx", options);
    EXPECT(p == prog);
}
