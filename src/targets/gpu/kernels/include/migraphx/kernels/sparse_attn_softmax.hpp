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
#ifndef MIGRAPHX_GUARD_KERNELS_SPARSE_ATTN_SOFTMAX_HPP
#define MIGRAPHX_GUARD_KERNELS_SPARSE_ATTN_SOFTMAX_HPP

#include "migraphx/kernels/float8.hpp"
#include "migraphx/kernels/gqa_softmax.hpp"
#include <migraphx/kernels/group_query_attention.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

template <size_t SparseBlockSize, class AttnProbs, class SeqLensK, class Params>
__device__ void sparse_attn_calculate_softmax(AttnProbs attention_probs,
                                              SeqLensK seqlens_k,
                                              Params params,
                                              index_int idx)
{
    (void)attention_probs;
    (void)seqlens_k;
    (void)params;
    (void)idx;
}

template <size_t SparseBlockSize,
          class Output,
          class Input,
          class PresentKey,
          class Probs,
          class SeqLensK,
          class Mask,
          class Params>
__device__ void sparse_attn_softmax(
    Output output, Input, PresentKey, Probs, SeqLensK seqlens_k, Mask mask, Params params)
{
    (void)mask;
    const index_int elements        = params.batch_size * params.num_heads * params.sequence_length;
    constexpr index_int num_layouts = mask.get_shape().lens[0];
    make_index().global_stride(elements, [&](auto idx) {
        const index_int batch_idx = idx / (params.num_heads * params.sequence_length);
        // (idx - batch_idx * batch_size) / num_heads
        const index_int head_idx =
            idx % (params.num_heads * params.sequence_length) / params.sequence_length;
        const index_int seq_idx = idx % params.sequence_length;
        const index_int key_total_seq_len = seqlens_k[batch_idx];
        // TODO just do key_total_seq_len - params.sequence_length
        const index_int past_seq_len   = params.sequence_length == 1 ? key_total_seq_len - 1 : 0;
        // constexpr index_int max_blocks = mask.get_shape().lens[1];
        const auto layout_idx          = head_idx % num_layouts;

        const auto q_idx         = seq_idx;
        const auto causal_length = past_seq_len + q_idx + 1;
        const auto q_abs_idx     = q_idx + past_seq_len;
        // printf("idx=%u, batch_idx=%u, head_idx=%u, seq_idx=%u, causal_len=%u\n", idx, batch_idx, head_idx, seq_idx, causal_length);
        for(index_int i = 0; i < causal_length; ++i)
        {
            const index_int mask_row    = q_abs_idx / SparseBlockSize;
            const index_int mask_column = i / SparseBlockSize;
            if(not mask[make_array(layout_idx, mask_row, mask_column)])
            {
                output[make_array(batch_idx, head_idx, seq_idx, i)] =
                    numeric_lowest<typename Output::type>();
            }
        }
        auto it = output.begin() + output.get_shape().index(make_array(batch_idx, head_idx, seq_idx, 0));
        // if(idx==0) {
        //     auto i = output.begin_at(make_array(0u, 0u, 0u, 0u));
        //     auto single = output.get_shape().index(make_array(0u, 0u, 0u, 0u));
        //     printf("val0=%f, it0=%f, single=%u\n", output[make_array(0u, 0u, 0u, 0u)], *i, single);
        //     i = output.begin_at(make_array(0u, 1u, 0u, 0u));
        //     single = output.get_shape().index(make_array(0u, 1u, 0u, 0u));
        //     printf("val1=%f, it1=%f, single=%u\n", output[make_array(0u, 1u, 0u, 0u)], *i, single);
        //     printf("val2=%f\n", output[make_array(0u, 2u, 0u, 0u)]);
        //     printf("val3=%f\n", output[make_array(0u, 3u, 0u, 0u)]);
        // }
        softmax_inplace(it, 1, causal_length);
        for(index_int i = causal_length; i < key_total_seq_len; ++i) {
            output[make_array(batch_idx, head_idx, seq_idx, i)] = 0;
        }
    });
}

} // namespace migraphx
#endif
