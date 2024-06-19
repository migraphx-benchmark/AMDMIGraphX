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

#include <builders.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op_builders {

instruction_ref binary_op(module_wrapper mod,
                          const std::vector<instruction_ref>& args,
                          const std::string& op_name,
                          std::optional<uint64_t> broadcasted,
                          std::optional<uint64_t> axis)
{

    if(not broadcasted.has_value() or not axis.has_value())
    {
        return mod.add_broadcastable_binary_op(op_name, args[0], args[1]);
    }

    if(broadcasted.value() != 0)
    {
        if(std::any_of(args.cbegin(), args.cend(), [](auto a) { return a->get_shape().dynamic(); }))
        {
            MIGRAPHX_THROW("Binary op broadcast attribute not supported for dynamic input shapes");
        }
        auto l = mod.add_instruction(
            make_op("broadcast",
                    {{"axis", axis.value()}, {"out_lens", args[0]->get_shape().lens()}}),
            args[1]);
        return mod.add_instruction(make_op(op_name), args[0], l);
    }
    return mod.add_instruction(make_op(op_name), args);
}

} // namespace op_builders
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
