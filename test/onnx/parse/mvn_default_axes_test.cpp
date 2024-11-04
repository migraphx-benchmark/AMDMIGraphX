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

#include <onnx_test.hpp>
#include <onnx_test_utils.hpp>

TEST_CASE(mvn_default_axes_test)
{
    // mvn_n_rank_test({0, 2, 3}, {2, 2, 2, 2}, optimize_onnx("mvn_default_axes_test.onnx"));

    std::vector<int64_t> axes{0, 2, 3};
    std::vector<size_t> input_shape{2, 2, 2, 2};
    const float exponent{2.f};
    const float epsilon{1e-09};
    migraphx::program prog = optimize_onnx("mvn_default_axes_test.onnx");

    using migraphx::make_op;
    migraphx::program p;
    auto* main_module = p.get_main_module();

    auto data =
        main_module->add_parameter("data", {migraphx::shape::float_type, std::move(input_shape)});

    auto exp = main_module->add_literal(migraphx::literal{data->get_shape(), {exponent}});
    auto eps = main_module->add_literal(migraphx::literal{data->get_shape(), {epsilon}});

    auto x_rm =
        main_module->add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), data);
    auto ex_sq = add_common_op(*main_module, make_op("pow"), {x_rm, exp});
    auto x_sq  = add_common_op(*main_module, make_op("pow"), {data, exp});
    auto e_xsq =
        main_module->add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), x_sq);
    auto variance      = add_common_op(*main_module, make_op("sub"), {e_xsq, ex_sq});
    auto std           = add_common_op(*main_module, make_op("sqrt"), {variance});
    auto x_variance    = add_common_op(*main_module, make_op("sub"), {data, x_rm});
    auto processed_std = add_common_op(*main_module, make_op("add"), {std, eps});
    // auto y = info.add_instruction(migraphx::make_op("div"), x_variance, processed_std);
    auto y = add_common_op(*main_module, make_op("div"), {x_variance, processed_std});

    //EXPECT(p == prog);
    EXPECT(p.sort() == prog.sort());
}
