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
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_bitwise_not : op_parser<parse_bitwise_not>
{
    std::vector<op_desc> operators() const { return {{"BitwiseNot"}}; }

    instruction_ref parse(const op_desc&,
                          const onnx_parser&,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        if(auto num_args = args.size(); num_args != 1)
        {
            MIGRAPHX_THROW("BitwiseNot: Unary operator requires 1 argument, " +
                           std::to_string(num_args) + " provided");
        }

        const auto& x = args[0];
        switch(x->get_shape().type())
        {
        case shape::type_t::int8_type:
        case shape::type_t::int16_type:
        case shape::type_t::int32_type:
        case shape::type_t::int64_type:
        case shape::type_t::uint8_type:
        case shape::type_t::uint16_type:
        case shape::type_t::uint32_type:
        case shape::type_t::uint64_type: break;
        default: MIGRAPHX_THROW("BitwiseNot: Only integral types are suppored");
        }

        std::cout << "BITWISE NOT PARSE" << std::endl;
        return info.add_instruction(migraphx::make_op("bitwise_not"), x);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
