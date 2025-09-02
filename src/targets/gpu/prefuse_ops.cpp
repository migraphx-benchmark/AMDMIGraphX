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
#include "migraphx/instruction.hpp"
#include "migraphx/literal.hpp"
#include "migraphx/make_op.hpp"
#include "migraphx/module.hpp"
#include <cstdint>
#include <migraphx/matcher.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/gpu/prefuse_ops.hpp>
#include <migraphx/gpu/gemm_softmax_gemm.hpp>
#include <migraphx/match/layernorm.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/op/group_query_attention.hpp>
#include <migraphx/op/sparse_attention.hpp>
#ifdef MIGRAPHX_USE_COMPOSABLEKERNEL
#include <migraphx/gpu/ck.hpp>
#endif
#include <migraphx/gpu/fuse_mlir.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_LAYERNORM_FUSION);

namespace {

template <class Derived, std::size_t N>
struct layernorm_base
{
    float epsilon = 1e-12f;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.epsilon, "epsilon"));
    }
    shape compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const
    {
        std::size_t nargs = N;
        if(not mods.empty())
        {
            auto* pm = mods.front();
            nargs += pm->get_parameter_names().size() - 1;
        }
        check_shapes{inputs, static_cast<const Derived&>(*this)}.has(nargs);
        auto s = inputs.front();
        auto t = s.type();
        if(not mods.empty())
            t = mods.front()->get_output_shapes().front().type();

        // Scalar output if all inputs are scalar
        if(inputs.front().elements() == 1 and
           all_of(inputs, [](const auto& ss) { return ss.scalar(); }))
            return inputs.front();
        auto l_s = shape::from_permutation(
            t, s.lens(), find_permutation(std::vector<shape>(inputs.begin(), inputs.begin() + N)));
        // just prelayernorm or preadd_layernorm
        if(nargs <= N)
            return l_s;
        // else, layernorm + pointwise fusion, preserve layout of fused op
        std::vector<shape> lp_s(inputs.begin() + N, inputs.end());
        lp_s.insert(lp_s.begin(), l_s);
        return shape::from_permutation(t, s.lens(), find_permutation(lp_s));
    }
};

struct layernorm : layernorm_base<layernorm, 1>
{

    std::string name() const { return "gpu::prelayernorm"; }
};
MIGRAPHX_REGISTER_OP(layernorm);

struct add_layernorm : layernorm_base<add_layernorm, 2>
{
    std::string name() const { return "gpu::preadd_layernorm"; }
};
MIGRAPHX_REGISTER_OP(add_layernorm);

struct find_layernorm
{
    auto matcher() const { return match::layernorm(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];
        float eps  = 0;
        if(contains(r.instructions, "eps"))
            eps = r.instructions["eps"]->eval().at<float>();

        m.replace_instruction(ins, layernorm{eps}, x_ins);
    }
};

struct find_add_layernorm
{
    auto matcher() const
    {
        return match::name("gpu::prelayernorm")(
            match::args(match::name("add")(match::used_once()).bind("add")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins     = r.result;
        auto add_ins = r.instructions["add"];
        auto op      = any_cast<layernorm>(ins->get_operator());

        m.replace_instruction(ins, add_layernorm{op.epsilon}, add_ins->inputs());
    }
};

struct pre_gemm_softmax_gemm : gemm_softmax_gemm
{
    std::string name() const { return "gpu::pre_gemm_softmax_gemm"; }
};
MIGRAPHX_REGISTER_OP(pre_gemm_softmax_gemm);

auto is_ck_gemm()
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
#ifdef MIGRAPHX_USE_COMPOSABLEKERNEL
        if(not enabled(MIGRAPHX_ENABLE_CK{}))
            return false;
        if(ins->name() != "dot")
            return false;
        if(not pre_gemm_softmax_gemm::is_ck_supported_type(ins->get_shape().type()))
            return false;
        return true;
#else
        (void)ins;
        return false;
#endif
    });
}

auto is_test_gemm(bool enable_attention)
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
        if(ins->name() != "dot")
            return false;
        return enable_attention;
    });
}

auto is_bias_supported()
{
    return match::make_basic_pred_matcher([=](instruction_ref) {
#ifdef MIGRAPHX_USE_COMPOSABLEKERNEL
        return not enabled(MIGRAPHX_ENABLE_CK{});
#else
        return true;
#endif
    });
}

struct find_gemm_softmax_gemm
{
    bool enable_attention = false;

    auto matcher() const
    {
        auto gemm1 = match::skip(match::name("contiguous"))(match::name("dot")(
            match::any_of(is_ck_gemm(), is_test_gemm(enable_attention)).bind("gemm1")));
        auto mul   = match::name("mul")(
            match::nargs(2), match::either_arg(0, 1)(match::is_constant().bind("scale"), gemm1));
        auto where = match::name("where")(match::arg(2)(match::is_constant().bind("select_const")),
                                          match::arg(1)(mul),
                                          match::arg(0)(match::any().bind("select_cond")));
        auto add =
            match::name("add")(is_bias_supported(),
                               match::nargs(2),
                               match::either_arg(0, 1)(match::none_of(mul).bind("bias"), mul));
        auto softmax = match::name("softmax")(match::arg(0)(match::any_of(mul, add, gemm1, where)))
                           .bind("softmax");

        return match::name("dot")(
            match::any_of(is_ck_gemm(), is_test_gemm(enable_attention)).bind("gemm2"))(
            match::arg(0)(softmax));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins       = r.result;
        auto gemm2_ins = r.instructions["gemm2"];
        auto gemm1_ins = r.instructions["gemm1"];

        float scale = 1.0;
        if(contains(r.instructions, "scale"))
        {
            auto scale_lit = r.instructions["scale"];
            // CK only supports single-valued scale
            scale_lit->eval().visit([&](const auto s) {
                // CK only supports single-valued scale
                if(not std::all_of(
                       s.begin() + 1, s.end(), [&](auto v) { return float_equal(v, s.front()); }))
                    return;
                scale = s.front();
            });
        }

        auto inputs = gemm1_ins->inputs(); // A, B
        if(contains(r.instructions, "select_cond"))
        {
            inputs.push_back(r.instructions["select_cond"]);
            inputs.push_back(r.instructions["select_const"]);
        }
        if(contains(r.instructions, "bias"))
        {
            inputs.push_back(r.instructions["bias"]);
        }

        inputs.push_back(gemm2_ins->inputs().back()); // B1

        mpm.get_module().replace_instruction(
            ins, pre_gemm_softmax_gemm{gemm2_ins->get_operator(), scale}, inputs);
    }
};

struct gpu_compute_attention_probabilities : op::group_query_attention
{
    std::string name() const { return "gpu::compute_attention_probabilities"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        auto query_lens        = inputs.front().lens();
        auto present_kv_seqlen = inputs.at(1).lens().at(2);
        std::vector<std::size_t> output_lens{
            query_lens.at(0), num_heads, query_lens.at(2), present_kv_seqlen};
        shape output_shape{inputs.front().type(), output_lens};
        return output_shape;
    }
};
MIGRAPHX_REGISTER_OP(gpu_compute_attention_probabilities);

struct gpu_compute_attention_scores : op::group_query_attention
{
    std::string name() const { return "gpu::compute_attention_scores"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        auto query_lens = inputs.front().lens();
        std::size_t q_hidden_size =
            (query_lens[1] * query_lens[3] * num_heads) / (num_heads + 2 * kv_num_heads);
        std::vector<std::size_t> output_lens{query_lens.at(0), query_lens.at(2), q_hidden_size};
        shape output_shape{inputs.front().type(), output_lens};
        return output_shape;
    }
};
MIGRAPHX_REGISTER_OP(gpu_compute_attention_scores);

struct gpu_gqa_rotary_embedding : op::group_query_attention
{
    std::string name() const { return "gpu::gqa_rotary_embedding"; }

    shape compute_shape(std::vector<shape> inputs) const { return inputs.front(); }
};
MIGRAPHX_REGISTER_OP(gpu_gqa_rotary_embedding);

struct gpu_gqa_softmax : op::group_query_attention
{
    std::string name() const { return "gpu::gqa_softmax"; }

    shape compute_shape(std::vector<shape> inputs) const { return inputs.at(2); }
};
MIGRAPHX_REGISTER_OP(gpu_gqa_softmax);

struct gpu_sparse_attn_softmax : op::sparse_attention
{
    std::string name() const { return "gpu::sparse_attn_softmax"; }

    shape compute_shape(std::vector<shape> inputs) const { return inputs.at(2); }
};
MIGRAPHX_REGISTER_OP(gpu_sparse_attn_softmax);

struct gpu_concat_past_present : op::group_query_attention
{
    std::string name() const { return "gpu::concat_past_present"; }

    shape compute_shape(std::vector<shape> inputs) const { return inputs[0]; }
};
MIGRAPHX_REGISTER_OP(gpu_concat_past_present);

struct find_group_query_attention
{
    auto matcher() const { return match::name("group_query_attention"); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins    = r.result;
        auto inputs = ins->inputs();
        auto v      = ins->get_operator().to_value();

        auto num_heads          = v.at("num_heads").to<std::size_t>();
        auto kv_num_heads       = v.at("kv_num_heads").to<std::size_t>();
        auto do_rotary          = v.at("do_rotary").to<bool>();
        auto local_window_size  = v.at("local_window_size").to<int>();
        auto rotary_interleaved = v.at("rotary_interleaved").to<bool>();
        auto scale              = v.at("scale").to<float>();

        auto q_shape                      = inputs[0]->get_shape();
        auto q_lens                       = q_shape.lens();
        const std::size_t batch_size      = q_lens[0];
        const std::size_t sequence_length = q_lens[1];
        std::size_t q_hidden_size         = q_lens[2];
        std::size_t head_size             = q_hidden_size / (num_heads + 2 * kv_num_heads);

        std::vector<std::size_t> bsnh{
            batch_size, sequence_length, num_heads + 2 * kv_num_heads, head_size};

        auto transposed_qkv = mpm.get_module().insert_instruction(
            ins, make_op("reshape", {{"dims", bsnh}}), inputs.at(0));

        transposed_qkv = mpm.get_module().insert_instruction(
            ins, make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), transposed_qkv);

        auto rotary_qkv = transposed_qkv;
        if(do_rotary)
        {
            std::vector<instruction_ref> rotary_inputs{
                transposed_qkv, inputs.at(5), inputs.at(7), inputs.at(8)};
            rotary_qkv =
                mpm.get_module().insert_instruction(ins,
                                                    gpu_gqa_rotary_embedding{do_rotary,
                                                                             kv_num_heads,
                                                                             local_window_size,
                                                                             num_heads,
                                                                             rotary_interleaved,
                                                                             scale},
                                                    rotary_inputs);
        }

        auto pres_k = inputs.at(3);
        auto pres_v = inputs.at(4);
        std::vector<instruction_ref> concat_inputs{rotary_qkv, pres_k, pres_v, inputs.at(5)};

        auto concat = mpm.get_module().insert_instruction(
            ins,
            gpu_concat_past_present{
                do_rotary, kv_num_heads, local_window_size, num_heads, rotary_interleaved, scale},
            concat_inputs);
        auto id =
            mpm.get_module().insert_instruction(ins, make_op("identity"), concat, pres_k, pres_v);

        std::vector<instruction_ref> attn_probs_inputs{id, pres_k, pres_v, inputs.at(5)};
        auto attn_probs = mpm.get_module().insert_instruction(
            ins,
            gpu_compute_attention_probabilities{
                do_rotary, kv_num_heads, local_window_size, num_heads, rotary_interleaved, scale},
            attn_probs_inputs);

        std::vector<instruction_ref> softmax_inputs{rotary_qkv, pres_k, attn_probs, inputs.at(5)};
        auto softmax = mpm.get_module().insert_instruction(
            ins,
            gpu_gqa_softmax{
                do_rotary, kv_num_heads, local_window_size, num_heads, rotary_interleaved, scale},
            softmax_inputs);
        std::vector<instruction_ref> new_inputs{rotary_qkv, pres_k, pres_v, inputs.at(5), softmax};

        auto get_tuple_elm_0 = std::next(ins);
        auto get_tuple_elm_1 = std::next(get_tuple_elm_0);
        auto get_tuple_elm_2 = std::next(get_tuple_elm_1);
        mpm.get_module().replace_instruction(get_tuple_elm_2, pres_v);
        mpm.get_module().replace_instruction(get_tuple_elm_1, pres_k);
        mpm.get_module().replace_instruction(
            get_tuple_elm_0,
            gpu_compute_attention_scores{
                do_rotary, kv_num_heads, local_window_size, num_heads, rotary_interleaved, scale},
            new_inputs);
    }
};

struct find_sparse_attention
{
    auto matcher() const { return match::name("sparse_attention"); }

    void apply(module& mod, const match::matcher_result& r) const
    {
        auto ins    = r.result;
        auto inputs = ins->inputs();
        auto op_v   = ins->get_operator().to_value();

        auto do_rotary          = op_v.at("do_rotary").to<bool>();
        auto rotary_interleaved = op_v.at("rotary_interleaved").to<bool>();
        auto num_heads          = op_v.at("num_heads").to<size_t>();
        auto kv_num_heads       = op_v.at("kv_num_heads").to<size_t>();
        auto sparse_block_size  = op_v.at("sparse_block_size").to<size_t>();
        auto scale              = op_v.at("scale").to<float>();

        auto qkv                = inputs.at(0);
        auto past_key           = inputs.at(3);
        auto past_val           = inputs.at(4);
        auto block_row_indices  = inputs.at(5);
        auto block_col_indices  = inputs.at(6);
        auto key_total_seq_lens = inputs.at(8);
        // GroupQueryAttention expects this input to contain total_sequence_lengths - 1 values
        // SparseAttention expects it to contain total_sequence_lengths values
        // Decrement it to make it compatible with GQA kernels
        auto dec_key_total_seq_lens = decrement_key_total_seq_lens(mod, ins, key_total_seq_lens);

        auto batch_size      = qkv->get_shape().lens()[0];
        auto sequence_length = qkv->get_shape().lens()[2];

        if(do_rotary)
        {
            qkv = mod.insert_instruction(
                ins,
                gpu_gqa_rotary_embedding{
                    do_rotary, kv_num_heads, -1, num_heads, rotary_interleaved, scale},
                {qkv, dec_key_total_seq_lens, inputs.at(9), inputs.at(10)});
        }

        auto concat = mod.insert_instruction(
            ins,
            gpu_concat_past_present{
                do_rotary, kv_num_heads, -1, num_heads, rotary_interleaved, scale},
            {qkv, past_key, past_val, dec_key_total_seq_lens});
        concat = mod.insert_instruction(ins, make_op("identity"), concat, past_key, past_val);

        auto q = mod.insert_instruction(
            ins, make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {num_heads}}}), qkv);
        auto k = transform_kv(mod, ins, num_heads, kv_num_heads, past_key);
        auto v = transform_kv(mod, ins, num_heads, kv_num_heads, past_val);

        auto attn_probs = attention_probabilities(mod, ins, q, k, scale);
        auto bnsm       = attn_probs->get_shape().lens();

        auto [mask, expanded_mask] = make_block_masks(
            mod, ins, block_row_indices, block_col_indices, sparse_block_size, num_heads, bnsm);
        // TODO move this into make_block_masks
        expanded_mask = mod.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::bool_type}}), expanded_mask);
        auto causal_mask = make_causal_mask(mod, ins, shape::uint32_type, bnsm, key_total_seq_lens);
        auto final_mask =
            mod.insert_instruction(ins, make_op("logical_and"), expanded_mask, causal_mask);
        auto ninf = mod.insert_literal(
            ins,
            {{attn_probs->get_shape().type(), {1}}, {-std::numeric_limits<float>::infinity()}});
        ninf = mod.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", attn_probs->get_shape().lens()}}), ninf);
        attn_probs = mod.insert_instruction(ins, make_op("where"), final_mask, attn_probs, ninf);
        auto softmax = mod.insert_instruction(ins, make_op("softmax", {{"axis", 3}}), attn_probs);

        // auto softmax = insert_softmax(mod,
        //                               ins,
        //                               do_rotary,
        //                               rotary_interleaved,
        //                               num_heads,
        //                               kv_num_heads,
        //                               scale,
        //                               sparse_block_size,
        //                               qkv,
        //                               past_key,
        //                               attn_probs,
        //                               key_total_seq_lens,
        //                               mask);

        auto attn_scores = attention_scores(mod, ins, softmax, v);

        auto&& outputs = ins->outputs();
        mod.replace_instruction(outputs[0], attn_scores);
        mod.replace_instruction(outputs[1], past_key);
        mod.replace_instruction(outputs[2], past_val);
    }

    instruction_ref
    decrement_key_total_seq_lens(module& mod, instruction_ref ins, instruction_ref ktsl) const
    {
        auto seq_len_lit =
            mod.insert_literal(ins, migraphx::literal{shape{shape::int32_type, {1}}, {1}});
        seq_len_lit = mod.insert_instruction(
            ins,
            migraphx::make_op("multibroadcast", {{"out_lens", ktsl->get_shape().lens()}}),
            seq_len_lit);
        return mod.insert_instruction(ins, migraphx::make_op("sub"), ktsl, seq_len_lit);
    }

    instruction_ref transform_kv(
        module& mod, instruction_ref ins, int num_heads, int kv_num_heads, instruction_ref kv) const
    {
        auto num_heads_ratio = num_heads / kv_num_heads;
        if(num_heads_ratio > 1)
        {
            auto kv_final_lens = kv->get_shape().lens();
            kv_final_lens[1]   = num_heads;
            kv           = mod.insert_instruction(ins, make_op("unsqueeze", {{"axes", {2}}}), kv);
            auto kv_lens = kv->get_shape().lens();
            kv_lens[2]   = num_heads_ratio;
            kv =
                mod.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", kv_lens}}), kv);
            kv = mod.insert_instruction(ins, make_op("reshape", {{"dims", kv_final_lens}}), kv);
        }

        return kv;
    }

    instruction_ref unpack_block_masks(module& mod,
                                       instruction_ref ins,
                                       instruction_ref block_row_ind,
                                       instruction_ref block_col_ind) const
    {
        const uint32_t num_layouts = block_row_ind->get_shape().lens()[0];
        const uint32_t mat_dim     = block_row_ind->get_shape().lens()[1] - 1;
        const uint32_t max_nnz     = block_col_ind->get_shape().lens()[1];
        const auto dtype           = block_row_ind->get_shape().type();
        const auto out_type        = shape::uint8_type;

        block_col_ind =
            mod.insert_instruction(ins, make_op("unsqueeze", {{"axes", {1}}}), block_col_ind);
        auto col_idx_lens = block_col_ind->get_shape().lens();
        col_idx_lens[1]   = mat_dim;
        block_col_ind     = mod.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", col_idx_lens}}), block_col_ind);

        const auto make_bounds = [&](const auto start, const auto end) {
            auto bounds = mod.insert_instruction(
                ins,
                make_op("slice", {{"axes", {1}}, {"starts", {start}}, {"ends", {end}}}),
                block_row_ind);
            bounds = mod.insert_instruction(ins, make_op("unsqueeze", {{"axes", {2}}}), bounds);
            auto bounds_lens = bounds->get_shape().lens();
            bounds_lens[2]   = max_nnz;
            return mod.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", bounds_lens}}), bounds);
        };
        auto lower_bounds = make_bounds(0, mat_dim);
        auto upper_bounds = make_bounds(1, mat_dim + 1);

        std::vector<uint32_t> column_indices_vals(max_nnz);
        std::iota(column_indices_vals.begin(), column_indices_vals.end(), 0);
        auto column_indices =
            mod.insert_literal(ins, {shape{dtype, {1, 1, max_nnz}}, column_indices_vals});
        column_indices = mod.insert_instruction(
            ins,
            make_op("multibroadcast", {{"out_lens", lower_bounds->get_shape().lens()}}),
            column_indices);

        auto gte = mod.insert_instruction(ins, make_op("less"), column_indices, lower_bounds);
        gte      = mod.insert_instruction(ins, make_op("not"), gte);
        gte      = mod.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::bool_type}}), gte);
        auto lt = mod.insert_instruction(ins, make_op("less"), column_indices, upper_bounds);
        lt      = mod.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::bool_type}}), lt);
        auto where_predicate = mod.insert_instruction(ins, make_op("logical_and"), gte, lt);
        auto updates         = mod.insert_instruction(
            ins, make_op("convert", {{"target_type", out_type}}), where_predicate);

        auto out_of_bound_idx =
            mod.insert_literal(ins, {shape{dtype, {1, 1, 1}}, std::vector<uint32_t>{mat_dim}});
        out_of_bound_idx = mod.insert_instruction(
            ins,
            make_op("multibroadcast", {{"out_lens", block_col_ind->get_shape().lens()}}),
            out_of_bound_idx);

        auto indices = mod.insert_instruction(
            ins, make_op("where"), where_predicate, block_col_ind, out_of_bound_idx);

        auto out_mat =
            mod.insert_literal(ins, {shape{out_type, {1, 1, 1}}, std::vector<uint32_t>(0)});
        out_mat = mod.insert_instruction(
            ins,
            make_op("multibroadcast", {{"out_lens", {num_layouts, mat_dim, mat_dim}}}),
            out_mat);

        return mod.insert_instruction(
            ins,
            make_op("scatter_none", {{"axis", 2}, {"skip_out_of_bounds", true}}),
            out_mat,
            indices,
            updates);
    }

    std::tuple<instruction_ref, instruction_ref>
    make_block_masks(module& mod,
                     instruction_ref ins,
                     instruction_ref block_row_ind,
                     instruction_ref block_col_ind,
                     size_t sparse_block_size,
                     size_t num_heads,
                     const std::vector<size_t>& bnsm) const
    {
        auto masks = unpack_block_masks(mod, ins, block_row_ind, block_col_ind);
        // Want masks to go from:
        // {num_layouts, max_blocks, max_blocks}
        // to:
        // {batch_size, num_layouts * head_layout_factor, max_blocks * block_size, max_blocks *
        // block_size}
        // Where head_layout_factor is (num_heads + num_layouts - 1) / num_layouts
        // In dimension 1(num_layouts * head_layout_factor) the layouts need to be repeated, that
        // is: {layout_1, layout_2, ..., layout_n} -> {layout_1, layout_2, ..., layout_n, layout_1,
        // layout_2, ..., layout_n, ...}
        auto num_layouts        = masks->get_shape().lens()[0];
        auto head_layout_factor = (num_heads + num_layouts - 1) / num_layouts;
        auto expanded_lens = masks->get_shape().lens();
        expanded_lens.insert(expanded_lens.begin(), bnsm[0]);
        expanded_lens[1] *= head_layout_factor;
        expanded_lens[2] *= sparse_block_size;
        expanded_lens[3] *= sparse_block_size;
        auto expanded_masks =
            mod.insert_instruction(ins, make_op("unsqueeze", {{"axes", {0, 3, 5}}}), masks);

        auto bc_lens = expanded_masks->get_shape().lens();
        bc_lens[0]   = head_layout_factor;
        bc_lens[3]   = sparse_block_size;
        bc_lens[5]   = sparse_block_size;
        bc_lens.insert(bc_lens.begin(), bnsm[0]);
        expanded_masks = mod.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", bc_lens}}), expanded_masks);

        expanded_masks = mod.insert_instruction(
            ins, make_op("reshape", {{"dims", expanded_lens}}), expanded_masks);

        std::vector<int64_t> axes;
        std::vector<int64_t> starts;
        std::vector<int64_t> ends;
        if(expanded_masks->get_shape().lens()[2] > bnsm[2])
        {
            axes.push_back(2);
            starts.push_back(bnsm[2] == 1 ? -2 : bnsm[2] - 1);
            ends.push_back(bnsm[2] == 1 ? -1 : bnsm[2]);
        }
        if(expanded_masks->get_shape().lens()[3] > bnsm[3])
        {
            axes.push_back(3);
            starts.push_back(0);
            ends.push_back(bnsm[3]);
        }
        if(not axes.empty())
        {
            expanded_masks = mod.insert_instruction(
                expanded_masks,
                make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}),
                expanded_masks);
        }

        return std::make_tuple(masks, expanded_masks);
    }

    instruction_ref make_causal_mask(module& mod,
                                     instruction_ref ins,
                                     shape::type_t dtype,
                                     const std::vector<size_t>& bnsm,
                                     instruction_ref ktsl) const
    {
        std::vector<size_t> causal_lens_vals(bnsm[2]);
        // TODO: Starting value would have to be offset when sequence length == 1

        dtype = ktsl->get_shape().type();
        std::iota(causal_lens_vals.begin(), causal_lens_vals.end(), 0);
        // Have to do literal->reshape->broadcast instead of literal with appropriate shape ->
        // broadcast to avoid simplify algebra messing up
        auto causal_lens = mod.insert_literal(ins, {{dtype, {bnsm[2]}}, causal_lens_vals});
        causal_lens =
            mod.insert_instruction(ins, make_op("reshape", {{"dims", {bnsm[2], 1}}}), causal_lens);
        causal_lens = mod.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", bnsm}}), causal_lens);
        auto sl = mod.insert_literal(ins, {{ktsl->get_shape().type(), {1}}, {bnsm[2]}});
        sl      = mod.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", ktsl->get_shape().lens()}}), sl);
        auto psl = mod.insert_instruction(ins, make_op("sub"), ktsl, sl);
        psl = mod.insert_instruction(ins, make_op("reshape", {{"dims", {bnsm[0], 1, 1, 1}}}), psl);
        psl = mod.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", bnsm}}), psl);
        causal_lens = mod.insert_instruction(ins, make_op("add"), causal_lens, psl);

        std::vector<size_t> column_indices_vals(bnsm[3]);
        std::iota(column_indices_vals.begin(), column_indices_vals.end(), 0);
        auto column_indices =
            mod.insert_literal(ins, {{dtype, {1, 1, 1, bnsm[3]}}, column_indices_vals});
        column_indices = mod.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", bnsm}}), column_indices);

        auto causal_mask =
            mod.insert_instruction(ins, make_op("greater"), column_indices, causal_lens);
        causal_mask = mod.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::bool_type}}), causal_mask);
        return causal_mask = mod.insert_instruction(ins, make_op("not"), causal_mask);
    }

    instruction_ref attention_probabilities(
        module& mod, instruction_ref ins, instruction_ref q, instruction_ref k, float scale) const
    {
        k = mod.insert_instruction(ins, make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);
        // TODO: This does not handle key_total_seq_lens at all, if it's different to
        // max_cache_sequence_length, behavior will not be as expected
        // Could set elements for which idx > total_sequence_length to 0, thus nullifying their
        // effect on the gemm.
        auto attn_probs = mod.insert_instruction(ins, make_op("dot"), q, k);
        scale           = float_equal(scale, 0.0f)
                              ? 1.0f / std::sqrt(static_cast<float>(q->get_shape().lens()[3]))
                              : scale;
        auto scale_lit  = mod.insert_literal(ins, literal{shape{shape::float_type, {1}}, {scale}});
        scale_lit       = mod.insert_instruction(
            ins,
            make_op("multibroadcast", {{"out_lens", attn_probs->get_shape().lens()}}),
            scale_lit);
        return mod.insert_instruction(ins, make_op("mul"), attn_probs, scale_lit);
    }

    instruction_ref insert_softmax(module& mod,
                                   instruction_ref ins,
                                   bool do_rotary,
                                   bool rotary_interleaved,
                                   size_t num_heads,
                                   size_t kv_num_heads,
                                   float scale,
                                   size_t sparse_block_size,
                                   instruction_ref qkv,
                                   instruction_ref past_key,
                                   instruction_ref attn_probs,
                                   instruction_ref key_total_seq_lens,
                                   instruction_ref mask) const
    {
        auto softmax = mod.insert_instruction(
            ins,
            gpu_sparse_attn_softmax{
                do_rotary, rotary_interleaved, num_heads, kv_num_heads, scale, sparse_block_size},
            {qkv, past_key, attn_probs, key_total_seq_lens, mask});

        return softmax;
    }

    instruction_ref attention_scores(module& mod,
                                     instruction_ref ins,
                                     instruction_ref softmax,
                                     instruction_ref v) const
    {
        auto attn_scores = mod.insert_instruction(ins, make_op("dot"), softmax, v);
        attn_scores      = mod.insert_instruction(
            ins, make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), attn_scores);
        return mod.insert_instruction(
            ins,
            make_op("reshape", {{"dims", ins->outputs()[0]->get_shape().lens()}}),
            attn_scores);
    }
};

} // namespace

void prefuse_ops::apply(module_pass_manager& mpm) const
{
    if(not enabled(MIGRAPHX_DISABLE_LAYERNORM_FUSION{}))
    {
        match::find_matches(mpm.get_module(), find_layernorm{});
        mpm.run_pass(dead_code_elimination{});
        match::find_matches(mpm.get_module(), find_add_layernorm{});
    }
    match::find_matches(mpm, find_gemm_softmax_gemm{enable_attention});
    match::find_matches(mpm, find_group_query_attention{});
    match::find_matches(mpm.get_module(), find_sparse_attention{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
