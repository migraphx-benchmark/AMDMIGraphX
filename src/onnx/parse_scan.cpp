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
#include "migraphx/argument.hpp"
#include "migraphx/errors.hpp"
#include "migraphx/instruction_ref.hpp"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_scan : op_parser<parse_scan>
{
    std::vector<op_desc> operators() const { return {{"Scan"}}; }

    std::vector<instruction_ref> parse(const op_desc& opd,
                                       onnx_parser& parser,
                                       onnx_parser::node_info info,
                                       std::vector<instruction_ref> args) const
    {
        // NOTE Version 8 of the operator differs to all the later versions
        if(not contains(info.attributes, "body"))
            MIGRAPHX_THROW("Scan: body attribute required");

        if(not contains(info.attributes, "num_scan_inputs"))
            MIGRAPHX_THROW("Scan: num_scan_inputs attribute required");

        const auto& body = info.attributes["body"].g();
        auto sub_mod     = parser.prog.create_module(info.name + "_scan");
        (void)parser.parse_graph(sub_mod, body);

        auto sub_mod_output_shapes = sub_mod->get_output_shapes();
        const auto M               = info.attributes["num_scan_inputs"].i();
        const auto N               = args.size() - M;
        const auto K               = sub_mod_output_shapes.size() - N;

        // NOTE Does not apply to opset 8 version
        if(sub_mod->get_parameter_names().size() != N + M)
            MIGRAPHX_THROW("Lorem ipsum");

        // SCAN INPUT AXES
        auto scan_input_axes = parse_vector_attribute(info, "scan_input_axes", M);
        if(scan_input_axes.empty())
            scan_input_axes = std::vector<int64_t>(M, 0);
        else
        {
            std::vector<int64_t> ndims;
            ndims.reserve(M);
            std::transform(args.begin() + N,
                           args.end(),
                           std::back_inserter(ndims),
                           [](instruction_ref arg) { return arg->get_shape().ndim(); });
            normalize_axes(scan_input_axes, ndims);
        }

        size_t num_iters = args[N]->get_shape().lens()[scan_input_axes[0]];
        for(auto i = 1; i < M; ++i)
        {
            if(args[i]->get_shape().lens()[scan_input_axes[i]] != num_iters)
                MIGRAPHX_THROW("Lorem ipsum");
        }
        // SCAN INPUT AXES

        // SCAN INPUT DIRECTIONS
        auto scan_input_directions = parse_vector_attribute(info, "scan_input_directions", M);
        if(scan_input_directions.empty())
            scan_input_directions = std::vector<int64_t>(M, 0);
        else if(any_of(scan_input_directions, [](auto i) { return i != 0 and i != 1; }))
        {
            MIGRAPHX_THROW(
                "Scan: scan_input_directions may contain only 1s and 0s, actual values: " +
                to_string_range(scan_input_directions));
        }
        // SCAN INPUT DIRECTIONS

        // SCAN OUTPUT AXES
        auto scan_output_axes = parse_vector_attribute(info, "scan_output_axes", K);
        if(scan_output_axes.empty())
            scan_output_axes = std::vector<int64_t>(K, 0);
        else
        {
            std::vector<int64_t> ndims;
            ndims.reserve(K);
            std::transform(sub_mod_output_shapes.begin() + N,
                           sub_mod_output_shapes.end(),
                           std::back_inserter(ndims),
                           [](const shape& sh) { return sh.ndim() + 1; });
            normalize_axes(scan_output_axes, ndims);
        }
        // SCAN OUTPUT AXES

        // SCAN OUTPUT DIRECTIONS
        auto scan_output_directions = parse_vector_attribute(info, "scan_output_directions", K);
        if(scan_output_directions.empty())
            scan_output_directions = std::vector<int64_t>(K, 0);
        else if(any_of(scan_output_directions, [](auto i) { return i != 0 and i != 1; }))
        {
            MIGRAPHX_THROW(
                "Scan: scan_output_directions may contain only 1s and 0s, actual values: " +
                to_string_range(scan_output_directions));
        }
        // SCAN OUTPUT DIRECTIONS

        std::vector<instruction_ref> alt_args(args.begin(), args.begin() + N);
        for(int64_t i = 0; i < num_iters; ++i)
        {
            for(auto j = 0; j < M; ++j)
            {
                auto dir   = scan_input_directions[j];
                auto idx   = (1 - dir) * i + dir * (num_iters - 1 - i);
                auto axis  = scan_input_axes[j];
                auto slice = info.add_instruction(
                    make_op("slice", {{"axes", {axis}}, {"starts", {idx}}, {"ends", {idx + 1}}}),
                    args[N + j]);
                alt_args.push_back(
                    info.add_instruction(make_op("squeeze", {{"axes", {axis}}}), slice));
            }
        }

        // TODO check that alt_args shapes match sub_mod input parameter shapes

        auto scan = info.add_instruction(
            make_op("scan",
                    {{"iterations", num_iters}, {"num_scan_inputs", M}, {"num_state_vars", N}}),
            alt_args,
            {sub_mod});

        std::vector<instruction_ref> ret;
        ret.reserve(N + K);
        for(auto i = 0; i < N + K; ++i)
        {
            auto ins = info.add_instruction(make_op("get_tuple_elem", {{"index", i}}), scan);
            if(i >= N)
            {
                auto scan_axis = scan_output_axes[i - N];
                ins = info.add_instruction(make_op("unsqueeze", {{"axes", {scan_axis}}}), ins);
            }
            ret.push_back(ins);
        }

        for(auto i = 1; i < num_iters; ++i)
        {
            for(auto j = 0; j < K; ++j)
            {
                auto tuple_idx = N + i * K + j;
                auto get =
                    info.add_instruction(make_op("get_tuple_elem", {{"index", tuple_idx}}), scan);
                auto scan_axis = scan_output_axes[j];
                auto usq = info.add_instruction(make_op("unsqueeze", {{"axes", {scan_axis}}}), get);
                std::vector concat_args{usq, usq};
                concat_args[scan_output_directions[j]] = ret[N + j];
                auto concat =
                    info.add_instruction(make_op("concat", {{"axis", scan_axis}}), concat_args);
                ret[N + j] = concat;
            }
        }

        return ret;
    }

    void normalize_axes(std::vector<int64_t>& axes, const std::vector<int64_t>& ndims) const
    {
        auto normalize_axis = [=](int64_t axis, int64_t ndim) {
            if(axis < -ndim or axis >= ndim)
                MIGRAPHX_THROW("Scan: Axis value {" + to_string(axis) + "} out of range [" +
                               to_string(-ndim) + ", " + to_string(ndim) + ")");

            return axis < 0 ? ndim + axis : axis;
        };

        std::transform(axes.begin(), axes.end(), ndims.begin(), axes.begin(), normalize_axis);
    }

    std::vector<int64_t> parse_vector_attribute(onnx_parser::node_info& info,
                                                const std::string& attr_name,
                                                size_t expected_size) const
    {
        if(not contains(info.attributes, attr_name))
            return {};

        std::vector<int64_t> res;
        auto&& attr = info.attributes[attr_name].ints();
        if(attr.size() != expected_size)
            MIGRAPHX_THROW("Scan: " + attr_name + " size is " + to_string(attr.size()) +
                           ", should be " + to_string(expected_size));
        res.assign(attr.begin(), attr.end());

        return res;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
