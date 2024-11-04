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

#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_mean_variance_normalization : op_parser<parse_mean_variance_normalization>
{

    std::vector<op_desc> operators() const { return {{"MeanVarianceNormalization"}}; }

    instruction_ref parse(const op_desc&,
                          const onnx_parser&,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        auto x         = args[0];
        auto x_shape   = x->get_shape();
        auto x_type    = x_shape.type();
        auto data_rank = x_shape.ndim();

        const float exponent{2};
        const float epsilon{1e-09};
        std::vector<int64_t> axes{0, 2, 3};

        // if(contains(info.attributes, "axes"))
        // {
        //     auto axes_key = info.attributes.find("axes");
        //     // std::vector<int> axesVector(axes_key->second.AppendToString,
        //     axes_key->second.end()); std::string temporary_axes_values = ""; std::vector<int64_t>
        //     temp_axes{axes_key->second.INTS}; axes = temp_axes;
        // }

        if(contains(info.attributes, "axes"))
        {
            const auto& axes_attr = info.attributes["axes"].ints();
            axes.assign(axes_attr.begin(), axes_attr.end());
            std::cout << "Axes after reading from attributes: " << std::endl;
            for(auto axe : axes)
            {
                std::cout << axe << " " << std::endl;
            }
        }

        else if(data_rank != 4)
        {
            MIGRAPHX_THROW(
                "Input tensor needs to be rank 4 when axes is not specified. Instead it is rank " +
                std::to_string(data_rank));
        }

        if(axes.size() != data_rank - 1)
        {
            MIGRAPHX_THROW("Length of axes array needs to be equal to input tensor rank - 1. "
                           "Current length: " +
                           std::to_string(axes.size()));
        }

        auto exp = info.add_literal(migraphx::literal{x_shape.type(), {exponent}});
        auto eps = info.add_literal(migraphx::literal{x_shape.type(), {epsilon}});

        auto x_rm = info.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), x);
        // auto exponent_bcast = info.add_instruction(migraphx::make_op("multibroadcast",
        // {{"out_lens", x_rm->get_shape().lens()}}), exp);
        auto ex_sq = info.add_common_op("pow", x_rm, exp);
        // auto exponent_bcast_x = info.add_instruction(migraphx::make_op("multibroadcast",
        // {{"out_lens", x->get_shape().lens()}}), exp);
        auto x_sq  = info.add_common_op("pow", x, exp);
        auto e_xsq = info.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), x_sq);
        // auto variance      = info.add_common_op("sub", e_xsq, ex_sq);
        auto variance      = info.add_common_op("sub", e_xsq, ex_sq);
        auto std           = info.add_common_op("sqrt", variance);
        auto x_variance    = info.add_common_op("sub", x, x_rm);
        auto processed_std = info.add_common_op("add", std, eps);
        // auto y = info.add_instruction(migraphx::make_op("div"), x_variance, processed_std);
        auto y = info.add_common_op("div", x_variance, processed_std);

        return y;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
