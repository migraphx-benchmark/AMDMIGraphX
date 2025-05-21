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
#include "migraphx/compile_options.hpp"
#include "migraphx/module.hpp"
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/register_op.hpp>

#include <test.hpp>

TEST_CASE(cos_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data{-1, 0, 1};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("cos"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return cosf(n); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(cos_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("cos"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-1, 0, 1};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return cosf(n); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

namespace migraphx {
struct unpack_masks
{
    std::string name() const { return "gpu::unpack_masks"; }

    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        const auto block_row_ind_lens = inputs[0].lens();
        const auto max_blocks         = block_row_ind_lens[1] - 1;
        return shape{shape::bool_type, {block_row_ind_lens[0], max_blocks, max_blocks}};
    }
};
MIGRAPHX_REGISTER_OP(unpack_masks);
} // namespace migraphx

void print_mat(
    const std::vector<bool>& data, int32_t batches, int32_t channels, int32_t rows, int32_t cols)
{
    for(int32_t i = 0; i < batches; ++i)
    {
        std::cout << "Batch " << i + 1 << std::endl;
        std::cout << "[";
        for(int32_t j = 0u; j < channels; ++j)
        {
            std::cout << "[";
            for(int32_t k = 0u; k < rows; ++k)
            {
                for(int32_t l = 0u; l < cols; ++l)
                {
                    std::cout << data[i * channels * rows * cols + j * rows * cols + k * cols + l]
                              << " ";
                }
                std::cout << "]";
                if(k < (rows - 1))
                    std::cout << "\n";
            }
            std::cout << "]\n";
        }
        std::cout << std::endl;
    }
}

TEST_CASE(bla)
{
    using namespace migraphx;

    migraphx::program p;
    auto* mm = p.get_main_module();

    const size_t num_layouts       = 2;
    const size_t max_blocks        = 4;
    const size_t max_nnz_blocks    = 9;
    const size_t sparse_block_size = 2;
    const size_t repeats           = 3;
    const size_t batch_size        = 2;

    shape block_row_indices_shape(shape::int32_type, {num_layouts, max_blocks + 1});
    shape block_col_indices_shape(shape::int32_type, {num_layouts, max_nnz_blocks});

    // clang-format off
    std::vector<int> bri_val{/*layout 1*/ 0, 1, 3, 6, 9,
                             /*layout 2*/ 0, 1, 3, 5, 8};

    std::vector<int> bci_val{/*layout 1*/ 0, 0, 1, 0, 1, 2, 0, 2, 3,
                             /*layout 2*/ 0, 0, 1, 1, 2, 1, 2, 3, -1};
    // clang-format on

    auto bri = mm->add_literal(literal{block_row_indices_shape, bri_val});
    auto bci = mm->add_literal(literal{block_col_indices_shape, bci_val});

    auto block_mask = mm->add_instruction(unpack_masks{}, {bri, bci});
    // {num_layouts, max_blocks, max_blocks} -> {num_layouts, max_blocks, 1, max_blocks, 1}
    // Want block_mask to go from:
    // {num_layouts, max_blocks, max_blocks}
    // to:
    // {batch_size, num_layouts * head_layout_factor, max_blocks * block_size, max_blocks *
    // block_size} Where head_layout_factor is (num_heads + num_layouts - 1) / num_layouts In
    // dimension 1(num_layouts * head_layout_factor) the layouts need to be repeated, that is:
    // {layout_1, layout_2, ..., layout_n} -> {layout_1, layout_2, ..., layout_n, layout_1,
    // layout_2, ..., layout_n, ...}
    auto expanded_lens = block_mask->get_shape().lens();
    expanded_lens.insert(expanded_lens.begin(), batch_size);
    expanded_lens[1] *= repeats;
    expanded_lens[2] *= sparse_block_size;
    expanded_lens[3] *= sparse_block_size;
    block_mask   = mm->add_instruction(make_op("unsqueeze", {{"axes", {0, 3, 5}}}), block_mask);
    auto bc_lens = block_mask->get_shape().lens();
    bc_lens[0]   = repeats;
    bc_lens[3]   = sparse_block_size;
    bc_lens[5]   = sparse_block_size;
    bc_lens.insert(bc_lens.begin(), batch_size);
    block_mask =
        mm->add_instruction(make_op("multibroadcast", {{"out_lens", bc_lens}}), block_mask);
    block_mask = mm->add_instruction(make_op("reshape", {{"dims", expanded_lens}}), block_mask);

    mm->add_return({block_mask});
    std::cout << p << std::endl;

    compile_options opts;
    opts.offload_copy = true;
    p.compile(make_target("gpu"), opts);
    std::cout << p << std::endl;

    auto res = p.eval({}).front();
    std::vector<bool> out;
    res.visit([&](auto r) { out.assign(r.begin(), r.end()); });

    print_mat(out,
              batch_size,
              num_layouts * repeats,
              sparse_block_size * max_blocks,
              sparse_block_size * max_blocks);
}

TEST_CASE(bla_mlir)
{
    using namespace migraphx;

    migraphx::program p;
    auto* mm = p.get_main_module();

    shape xs{shape::float_type, {1024, 1024}};
    shape ys{shape::float_type, {1024, 1024}};
    shape ss{shape::float_type, {1}};
    auto x_param     = mm->add_parameter("x", xs);
    auto y_param     = mm->add_parameter("y", ys);
    auto scale_param = mm->add_parameter("scale", ss);

    auto dot1  = mm->add_instruction(make_op("dot"), x_param, y_param);
    auto scale = mm->add_instruction(
        make_op("multibroadcast", {{"out_lens", dot1->get_shape().lens()}}), scale_param);
    dot1 = mm->add_instruction(make_op("mul"), dot1, scale);
    auto softmax = mm->add_instruction(make_op("softmax"), dot1);
    auto dot2    = mm->add_instruction(make_op("dot"), softmax, x_param);
    mm->add_return({dot2});

    compile_options opts;
    opts.offload_copy = true;
    p.compile(make_target("gpu"), opts);
    std::cout << p << std::endl;
}
