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
#include "migraphx/instruction_ref.hpp"
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_multi_head_attention : op_parser<parse_multi_head_attention>
{

    std::vector<op_desc> operators() const { return {{"MultiHeadAttention"}}; }

    void unpack_qkv(const onnx_parser::node_info& info,
                    instruction_ref& query,
                    instruction_ref& key,
                    instruction_ref& value) const
    {
        // (batch_size, q_sequence_length, num_heads, 3, head_size) ->
        // (3, batch_size, q_sequence_length, num_heads, head_size)
        auto qkv_packed =
            info.add_instruction(make_op("transpose", {{"permutation", {3, 0, 1, 2, 4}}}), query);
        query = info.add_instruction(
            make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), qkv_packed);
        query = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), query);
        key   = info.add_instruction(
            make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), qkv_packed);
        key   = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), key);
        value = info.add_instruction(
            make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), qkv_packed);
        value = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), value);
    }

    void unpack_kv(const onnx_parser::node_info& info,
                   instruction_ref& key,
                   instruction_ref& value) const
    {
        // (batch_size, kv_sequence_length, num_heads, 2, head_size) ->
        // (2, batch_size, kv_sequence_length, num_heads, head_size)
        auto kv_packed =
            info.add_instruction(make_op("transpose", {{"permutation", {3, 0, 1, 2, 4}}}), key);
        key = info.add_instruction(
            make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), kv_packed);
        key   = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), key);
        value = info.add_instruction(
            make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), kv_packed);
        value = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), value);
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        if(not contains(info.attributes, "num_heads"))
            MIGRAPHX_THROW("MultiHeadAttention: num_heads attribute is required");

        if(args.size() < 1 or args.size() > 4)
            MIGRAPHX_THROW("MultiHeadAttention: Wrong number of inputs. Only query, key, value and "
                           "bias are supported.");

        auto query                = args[0];
        auto q_dim                = query->get_shape().ndim();
        auto q_lens               = query->get_shape().lens();
        int64_t batch_size        = q_lens[0];
        int64_t q_sequence_length = q_lens[1];
        int64_t num_heads         = parser.parse_value(info.attributes.at("num_heads")).at<int>();
        int64_t head_size, hidden_size_v;

        if(q_dim != 3 and q_dim != 5)
            MIGRAPHX_THROW(
                "MultiHeadAttention: Input query needs to be 3 or 5 dimensional, current: " +
                std::to_string(q_dim));

        instruction_ref key, value;
        bool transpose_kv = true;
        if(q_dim == 5)
        {
            if(q_lens[3] != 3)
                MIGRAPHX_THROW("MultiHeadAttention: Packed qkv tensor 4th dimension needs to be 3");

            // Packed QKV: (batch_size, q_sequence_length, num_heads, 3, head_size)
            head_size     = q_lens[4];
            hidden_size_v = head_size * num_heads;

            unpack_qkv(info, query, key, value);

            if(args.size() > 3)
            {
                auto q_bias = info.add_instruction(
                    make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {hidden_size_v}}}),
                    args[3]);
                auto k_bias = info.add_instruction(make_op("slice",
                                                           {{"axes", {0}},
                                                            {"starts", {hidden_size_v}},
                                                            {"ends", {2 * hidden_size_v}}}),
                                                   args[3]);
                auto v_bias = info.add_instruction(make_op("slice",
                                                           {{"axes", {0}},
                                                            {"starts", {2 * hidden_size_v}},
                                                            {"ends", {3 * hidden_size_v}}}),
                                                   args[3]);
                q_bias      = info.add_instruction(
                    make_op("reshape", {{"dims", {num_heads, head_size}}}), q_bias);
                k_bias = info.add_instruction(
                    make_op("reshape", {{"dims", {num_heads, head_size}}}), k_bias);
                v_bias = info.add_instruction(
                    make_op("reshape", {{"dims", {num_heads, head_size}}}), v_bias);

                query = info.add_common_op("add", query, q_bias);
                key   = info.add_common_op("add", key, k_bias);
                value = info.add_common_op("add", value, v_bias);
            }
        }
        else // q_dim == 3
        {
            if(args.size() < 2)
                MIGRAPHX_THROW(
                    "MultiHeadAttention: Wrong number of inputs, key and value are missing.");

            // Query: (batch_size, q_sequence_length, hidden_size)
            int64_t hidden_size = q_lens[2];
            head_size           = hidden_size / num_heads;

            if(args.size() > 3)
            {
                auto q_bias = info.add_instruction(
                    make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {hidden_size}}}),
                    args[3]);
                query = info.add_common_op("add", query, q_bias);
            }

            std::vector<int64_t> q_dims{batch_size, q_sequence_length, num_heads, head_size};
            query = info.add_instruction(make_op("reshape", {{"dims", q_dims}}), query);

            key        = args[1];
            auto k_dim = key->get_shape().ndim();

            if(k_dim < 3 or k_dim > 5)
                MIGRAPHX_THROW(
                    "MultiHeadAttention: Input key needs to be 3, 4 or 5 dimensional, current: " +
                    std::to_string(k_dim));

            if(k_dim == 5)
            {
                if(args[1]->get_shape().lens()[3] != 2)
                    MIGRAPHX_THROW(
                        "MultiHeadAttention: Packed kv tensor 4th dimension needs to be 2");

                // Packed KV: (batch_size, kv_sequence_length, num_heads, 2, head_size)
                hidden_size_v = key->get_shape().lens()[4] * num_heads;
                unpack_kv(info, key, value);
            }
            else
            {
                if(args.size() < 3)
                    MIGRAPHX_THROW("MultiHeadAttention: Wrong number of inputs, value is missing.");

                value = args[2];
                if(k_dim == 4)
                {
                    if(value->get_shape().ndim() != 4)
                        MIGRAPHX_THROW("MultiHeadAttention: Value should be 4 dimensional.");

                    // Key: (batch_size, num_heads, kv_sequence_length, head_size)
                    // Value: (batch_size, num_heads, kv_sequence_length, head_size_v)
                    hidden_size_v = value->get_shape().lens()[3] * num_heads;
                    transpose_kv  = false;
                }
                else // k_dim == 3
                {
                    if(value->get_shape().ndim() != 3)
                        MIGRAPHX_THROW("MultiHeadAttention: Value should be 3 dimensional.");

                    // Key: (batch_size, kv_sequence_length, hidden_size)
                    // Value: (batch_size, kv_sequence_length, hidden_size_v)
                    hidden_size_v       = value->get_shape().lens()[2];
                    int64_t head_size_v = hidden_size_v / num_heads;

                    if(args.size() > 3)
                    {
                        auto k_bias = info.add_instruction(make_op("slice",
                                                                   {{"axes", {0}},
                                                                    {"starts", {hidden_size}},
                                                                    {"ends", {2 * hidden_size}}}),
                                                           args[3]);
                        auto v_bias = info.add_instruction(
                            make_op("slice",
                                    {{"axes", {0}},
                                     {"starts", {2 * hidden_size}},
                                     {"ends", {2 * hidden_size + hidden_size_v}}}),
                            args[3]);
                        key   = info.add_common_op("add", key, k_bias);
                        value = info.add_common_op("add", value, v_bias);
                    }

                    int64_t kv_sequence_length = key->get_shape().lens()[1];
                    std::vector<int64_t> k_dims{
                        batch_size, kv_sequence_length, num_heads, head_size};
                    std::vector<int64_t> v_dims{
                        batch_size, kv_sequence_length, num_heads, head_size_v};
                    key   = info.add_instruction(make_op("reshape", {{"dims", k_dims}}), key);
                    value = info.add_instruction(make_op("reshape", {{"dims", v_dims}}), value);
                }
            }
        }

        // Target shape: (batch_size, num_heads, sequence_length, head_size)
        std::vector<int64_t> perm{0, 2, 1, 3};
        query = info.add_instruction(make_op("transpose", {{"permutation", perm}}), query);
        if(transpose_kv)
        {
            key   = info.add_instruction(make_op("transpose", {{"permutation", perm}}), key);
            value = info.add_instruction(make_op("transpose", {{"permutation", perm}}), value);
        }

        float scale = 1 / std::sqrt(head_size);
        if(contains(info.attributes, "scale"))
            scale = parser.parse_value(info.attributes.at("scale")).at<float>();

        auto scale_literal = info.add_literal(
            migraphx::literal{migraphx::shape{query->get_shape().type()}, {scale}});

        auto key_transposed =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), key);

        auto result = info.add_instruction(make_op("dot"), query, key_transposed);
        result      = info.add_common_op("mul", result, scale_literal);
        result      = info.add_instruction(make_op("softmax", {{"axis", -1}}), result);
        result      = info.add_instruction(make_op("dot"), result, value);
        result      = info.add_instruction(make_op("transpose", {{"permutation", perm}}), result);
        result      = info.add_instruction(
            make_op("reshape", {{"dims", {batch_size, q_sequence_length, hidden_size_v}}}), result);

        return result;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
