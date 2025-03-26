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
        shape::bf16_type, shape::double_type, shape::float_type};

    std::vector<op_desc> operators() const
    {
        return {{"MeanVarianceNormalization", "mean_variance_norm"}}; 
    }

    instruction_ref parse(const op_desc& opd,
        const onnx_parser& parser,
        const onnx_parser::node_info& info,
        std::vector<instruction_ref> args) const
    {
        const auto& X                  = args[0];

        const auto Epsilon_default            = 1e-9f;
        const auto axes_default = std::vector<size_t>{0, 2, 3};
        
        auto Epsilon = Epsilon_default;
        if (contains(info.attributes, "epsilon"))
        {
            Epsilon = parser.parse_value(info.attributes.at("epsilon")).at<float>();
        }

        auto axes = axes_default;
        auto axes_min_size = axes.size();
        if (contains(info.attributes, "axes"))
        {
            axes.assign(info.attributes.at("axes").ints().begin(), info.attributes.at("axes").ints().end());
            axes_min_size = axes.size();
        }
        assert(X->get_shape().ndim() >= axes_min_size);
        
        const auto dtype         = args[0]->get_shape().type();
        const auto literal_dtype = dtype;

        if(not contains(valid_types, dtype))
        {
            MIGRAPHX_THROW(opd.onnx_name + ": invalid output type: " + std::to_string(dtype) +
                           ". Valid types are (bfloat16), (double), and (float).");
        }

        auto E_X        = info.add_instruction(make_op("reduce_mean", {{"axes", axes}}), X);
        auto E_sqr_X    = info.add_common_op("mul", E_X, E_X);
        auto X_sqr      = info.add_common_op("mul", X, X);
        auto E_X_sqr    = info.add_instruction(make_op("reduce_mean", {{"axes", axes}}), X_sqr);
        auto std_sqr    = info.add_common_op("sub", E_X_sqr, E_sqr_X);
        auto std        = info.add_common_op("sqrt", std_sqr);
        auto numerator  = info.add_common_op("sub", X, E_X);
        auto Epsilon_literal= info.add_literal(literal{shape{literal_dtype}, {Epsilon}});
        auto denominator= info.add_common_op("add", std, Epsilon_literal);
        auto Y          = info.add_common_op("div", numerator, denominator);

        return Y;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
