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

#include <module_wrapper.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op_builders {

instruction_ref module_wrapper::make_contiguous(instruction_ref ins) const
{
    auto attr       = ins->get_operator().to_value();
    std::string key = "require_std_shape";
    if((attr.get(key, false)) or (not ins->get_shape().standard()))
    {
        return add_instruction(make_op("contiguous"), ins);
    }

    return ins;
}

instruction_ref module_wrapper::add_bias(const std::vector<instruction_ref>& args,
                                         instruction_ref curr_ins,
                                         uint64_t axis) const
{
    if(args.size() == 3)
    {
        instruction_ref bias_bcast;
        // if curr_ins has a dynamic output shape use 2 input broadcast
        if(curr_ins->get_shape().dynamic())
        {
            bias_bcast =
                mod->add_instruction(make_op("broadcast", {{"axis", axis}}), args[2], curr_ins);
        }
        else
        {
            bias_bcast = mod->add_instruction(
                make_op("broadcast", {{"axis", axis}, {"out_lens", curr_ins->get_shape().lens()}}),
                args[2]);
        }
        return mod->add_instruction(make_op("add"), curr_ins, bias_bcast);
    }
    return curr_ins;
}

instruction_ref module_wrapper::add_broadcastable_binary_op(const std::string& op_name,
                                                            instruction_ref arg0,
                                                            instruction_ref arg1) const
{
    return this->add_common_op(op_name, arg0, arg1);
}

/**
 * @brief A wrapper for insert_common_args(), which constructs an argument list
 * and inserts multibroadcast and convert ops to match inputs to a common shape and type
 * as required.  The requested operation is placed after the added multibroadcast and convert ops,
 * if any, so that their results are transparent to the programmer.
 *
 * Use add_common_op() to match input sizes when inputs may be
 *  either static or dynamic.
 *
 * @param op_name               string; Name of operation (op) to add; valid names are the same as
 * for make_op()
 *
 * @param inputs                vector of instruction_ref.  List of instructions for the new
 * operator.  Multibroadcast and convert operations, if needed, are deduced from these too.
 *
 * @return instruction_ref      Returns an instruction_ref which is the result of the requested
 * operation.
 *
 */
instruction_ref module_wrapper::add_common_op(const std::string& op_name,
                                              std::vector<instruction_ref> inputs) const
{
    return migraphx::add_common_op(*mod, make_op(op_name), std::move(inputs));
}

instruction_ref module_wrapper::add_instruction(const operation& op,
                                                const std::vector<instruction_ref>& args) const
{
    return mod->add_instruction(op, args);
}

instruction_ref module_wrapper::add_instruction(const operation& op,
                                                const std::vector<instruction_ref>& args,
                                                const std::vector<module_ref>& mods) const
{
    return mod->add_instruction(op, args, mods);
}

instruction_ref module_wrapper::add_literal(literal l) const
{
    return mod->add_literal(std::move(l));
}

} // namespace op_builders
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
