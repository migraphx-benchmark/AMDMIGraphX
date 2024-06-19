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

#pragma once

#include <migraphx/instruction_ref.hpp>
#include <migraphx/module.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op_builders {

struct module_wrapper
{
    module* mod = nullptr;

    instruction_ref make_contiguous(instruction_ref ins) const;
    instruction_ref add_bias(const std::vector<instruction_ref>& args,
                             instruction_ref curr_ins,
                             uint64_t axis) const;

    instruction_ref add_broadcastable_binary_op(const std::string& op_name,
                                                instruction_ref arg0,
                                                instruction_ref arg1) const;

    instruction_ref add_common_op(const std::string& op_name,
                                  std::vector<instruction_ref> inputs) const;

    template <class... Ts>
    instruction_ref add_common_op(const std::string& op_name, Ts... xs) const
    {
        return add_common_op(op_name, {xs...});
    }

    instruction_ref add_instruction(const operation& op,
                                    const std::vector<instruction_ref>& args) const;

    instruction_ref add_instruction(const operation& op,
                                    const std::vector<instruction_ref>& args,
                                    const std::vector<module_ref>& mods) const;

    template <class... Ts>
    instruction_ref add_instruction(const operation& op, Ts... xs) const
    {
        return add_instruction(op, {xs...});
    }
    instruction_ref add_literal(literal l) const;
    template <class... Ts>
    instruction_ref add_literal(Ts&&... xs) const
    {
        return add_literal(literal{std::forward<Ts>(xs)...});
    }
};

} // namespace op_builders
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
