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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct mean_variance_norm : op_parser<mean_variance_norm>
{
    std::set<shape::type_t> valid_types = {
        shape::bf16_type, shape::double_type, shape::float_type, shape::half_type};

    std::vector<op_desc> operators() const
    {
        return {{"MeanVarianceNormalization", "mean_variance_norm"}}; 
    }

    instruction_ref parse(const op_desc& opd,
        const onnx_parser& parser,
        const onnx_parser::node_info& info,
        std::vector<instruction_ref> args) const
    {
        const auto dtype         = args[0]->get_shape().type();
        const auto literal_dtype = dtype;

        if(not contains(valid_types, dtype))
        {
            MIGRAPHX_THROW(opd.onnx_name + ": invalid output type: " + std::to_string(dtype) +
                           ". Valid types are (bfloat16), (double), (float) and (half)");
        }

        const auto& x           = args[0];

        const auto eps_default  = 1e-7f;
        const auto axes_default = std::vector<size_t>{0, 2, 3};
        
        auto eps = eps_default;
        if (contains(info.attributes, "epsilon"))
        {
            eps = parser.parse_value(info.attributes.at("epsilon")).at<float>();
        }

        auto axes = axes_default;
        auto axes_min_size = axes.size();
        if (contains(info.attributes, "axes"))
        {
            axes.assign(info.attributes.at("axes").ints().begin(), info.attributes.at("axes").ints().end());
            axes_min_size = axes.size();
        }

        if (x->get_shape().ndim() < axes_min_size)
        {
            MIGRAPHX_THROW(opd.onnx_name + ": input dimension has value: " + std::to_string(x->get_shape().ndim()) + 
                        ". It sould be greater or equal to: " + std::to_string(axes_min_size));
        } 
        
        auto expected_val_x     = info.add_instruction(make_op("reduce_mean", {{"axes", axes}}), x);
        auto expected_val_sqr_x = info.add_common_op("mul", expected_val_x, expected_val_x);
        auto x_sqr              = info.add_common_op("mul", x, x);
        auto expected_val_x_sqr = info.add_instruction(make_op("reduce_mean", {{"axes", axes}}), x_sqr);
        auto std_sqr            = info.add_common_op("sub", expected_val_x_sqr, expected_val_sqr_x);
        auto std                = info.add_common_op("sqrt", std_sqr);
        auto numerator          = info.add_common_op("sub", x, expected_val_x);
        auto eps_literal        = info.add_literal(literal{shape{literal_dtype}, {eps}});
        auto denominator        = info.add_common_op("add", std, eps_literal);
        auto y                  = info.add_common_op("div", numerator, denominator);

        return y;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
