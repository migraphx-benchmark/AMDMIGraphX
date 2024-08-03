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

#include <onnx_test.hpp>

TEST_CASE(convinteger_no_bias_mismatched_data_inputs_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto data   = mm->add_parameter("0", {migraphx::shape::int8_type, {1, 3, 32, 32}});
    auto weight = mm->add_parameter("1", {migraphx::shape::uint8_type, {1, 3, 5, 5}});

    mm->add_literal(migraphx::literal{migraphx::shape{data->get_shape().type(), {1}, {0}}, {0}});
    mm->add_literal(
        migraphx::literal{migraphx::shape{weight->get_shape().type(), {1}, {0}}, {128}});

    // shift uint8 input
    auto int8_shift2 =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type}, {-128}});

    // shift uint8 input
    auto unshifted_input_half = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), weight);

    auto mbr2 = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", weight->get_shape().lens()}}),
        int8_shift2);

    auto input_shifted_half =
        mm->add_instruction(migraphx::make_op("add"), unshifted_input_half, mbr2);

    weight = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}),
        input_shifted_half);

    mm->add_instruction(migraphx::make_op("quant_convolution"), data, weight);

    auto prog = optimize_onnx("convinteger_mismatched_input_types_test.onnx");
    mm->sort();
    prog.get_main_module()->sort();
    EXPECT(p == prog);
}
