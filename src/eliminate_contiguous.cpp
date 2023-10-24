/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/op/contiguous.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/par_for.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_ELIMINATE_CONTIGUOUS)

static bool try_compute_shape(instruction_ref ins,
                              const std::vector<shape>& inputs,
                              const std::vector<module_ref>& mods)
{
    try
    {
        shape new_shape = ins->get_operator().compute_shape(inputs, mods);

        // Cannot tell if a dynamic shape will need to be made contiguous
        if(new_shape.dynamic())
        {
            return false;
        }

        // If the output shape is a standard shape, no need to try its output
        if(new_shape.standard())
        {
            return true;
        }

        // if no changes for the shape, the contiguous can also be removed
        if(new_shape == ins->get_shape())
        {
            return true;
        }

        auto outputs = ins->outputs();
        // If the current instruction has no output, it means it is the last
        // instruction and generates a non-standard output shape, and the last
        // output shape is different from the case with the contiguous operator
        if(outputs.empty())
        {
            return false;
        }

        for(auto output : outputs)
        {
            auto args = output->inputs();
            std::vector<shape> input_shapes(args.size());
            std::transform(args.begin(), args.end(), input_shapes.begin(), [&](auto& arg) {
                return (arg == ins) ? new_shape : arg->get_shape();
            });

            if(not try_compute_shape(output, input_shapes, output->module_inputs()))
            {
                return false;
            }
        }
    }
    catch(const std::exception& e)
    {
        if(enabled(MIGRAPHX_TRACE_ELIMINATE_CONTIGUOUS{}))
        {
            std::cout << "Exception: " << e.what() << std::endl;
        }
        return false;
    }
    catch(...)
    {
        if(enabled(MIGRAPHX_TRACE_ELIMINATE_CONTIGUOUS{}))
        {
            std::cout << "Unknown exception" << std::endl;
        }
        return false;
    }

    return true;
}

static bool try_compute_shape_print(instruction_ref ins,
                                    const std::vector<shape>& inputs,
                                    const std::vector<module_ref>& mods,
                                    int32_t level,
                                    bool expect_standard_shape)
{
    // std::cout << std::endl << "$$$$$$ try_compute_shape ( " << level << ") $$$$$$" << std::endl;
    try
    {
        // std::cout << "$$ ins name: " << ins->name() << " shape: " << ins->get_shape() << std::endl;
        shape new_shape = ins->get_operator().compute_shape(inputs, mods);
        // std::cout << "$$ new shape: " << new_shape << std::endl;

        // Cannot tell if a dynamic shape will need to be made contiguous
        if(new_shape.dynamic())
        {
            // std::cout << "$$ ret false: dynamic" << std::endl;
            return false;
        }

        // If the output shape is a standard shape, no need to try its output
        if(new_shape.standard())
        {
            // std::cout << "$$ ret true: standard" << std::endl;
            return true;
        }
        else if (expect_standard_shape)
        {
            return false;
        }

        // if no changes for the shape, the contiguous can also be removed
        if(new_shape == ins->get_shape())
        {
            // std::cout << "$$ ret true: new_shape == ins->get_shape()" << std::endl;
            return true;
        }

        auto outputs = ins->outputs();
        // If the current instruction has no output, it means it is the last
        // instruction and generates a non-standard output shape, and the last
        // output shape is different from the case with the contiguous operator
        if(outputs.empty())
        {
            // std::cout << "$$ ret false: outputs.empty()" << std::endl;
            return false;
        }

        for(auto output : outputs)
        {
            auto args = output->inputs();
            std::vector<shape> input_shapes(args.size());
            std::transform(args.begin(), args.end(), input_shapes.begin(), [&](auto& arg) {
                return (arg == ins) ? new_shape : arg->get_shape();
            });

            if(not try_compute_shape_print(output, input_shapes, output->module_inputs(), level++, expect_standard_shape))
            {
                // std::cout << "$$ ret false: nested output" << std::endl;
                return false;
            }
        }
    }
    catch(const std::exception& e)
    {
        if(enabled(MIGRAPHX_TRACE_ELIMINATE_CONTIGUOUS{}))
        {
            std::cout << "Exception: " << e.what() << std::endl;
        }
        // std::cout << "$$ ret false: std::exception" << e.what() << std::endl;
        return false;
    }
    catch(...)
    {
        if(enabled(MIGRAPHX_TRACE_ELIMINATE_CONTIGUOUS{}))
        {
            std::cout << "Unknown exception" << std::endl;
        }
        // std::cout << "$$ ret false: Unknown exception" << std::endl;
        return false;
    }

    // std::cout << "$$ ret true: end of function" << std::endl;
    return true;
}

static bool try_compute_shape(instruction_ref ins,
                              const std::vector<instruction_ref>& args,
                              const std::vector<module_ref>& mods)
{
    auto inputs = to_shapes(args);
    return try_compute_shape(ins, inputs, mods);
}

static bool try_compute_shape_print(instruction_ref ins,
                              const std::vector<instruction_ref>& args,
                              const std::vector<module_ref>& mods,
                              int32_t level,
                              bool expect_standard_shape)
{
    auto inputs = to_shapes(args);
    return try_compute_shape_print(ins, inputs, mods, level, expect_standard_shape);
}

template <class F>
static void remove_contiguous(const std::string& op_name, module& m, F f)
{
    auto last = std::prev(m.end());
    std::vector<instruction_ref> const_instructions;
    // std::unordered_map<instruction_ref, instruction_ref> replace_map;
    for(auto ins : iterator_for(m))
    {
        // return instruction should have inputs with standard shape
        if(ins->name() == "@return")
            continue;

        if(ins != last and ins->outputs().empty())
            continue;

        if(not f(ins))
            continue;

        // Make a copy so we can modify it while we iterate
        auto args     = ins->inputs();
        auto new_args = args;
        auto mod_args = ins->module_inputs();

        bool is_gpu_cont = false;
        if(op_name == "gpu::contiguous")
            is_gpu_cont = true;

        if(is_gpu_cont)
        {
            std::cout << "\n+++ ins: " << ins->name() << ", shape: " << ins->get_shape()
                      << ", standard: " << std::boolalpha << ins->get_shape().standard()
                      << std::endl;
        }
        for(auto arg : ins->inputs())
        {
            std::cout << "+++++ ins input: " << arg->name() << ", shape: " << arg->get_shape()
                      << ", standard: " << std::boolalpha << arg->get_shape().standard()
                      << std::endl;
            // if (replace_map.count(arg))
            //     arg = replace_map.at(arg);
            if(arg->name() != op_name)
                continue;
            // if(is_gpu_cont)
            // {
            //     std::cout << "\n+++ ins: " << ins->name() << ", shape: " << ins->get_shape()
            //             << ", standard: " << std::boolalpha << ins->get_shape().standard()
            //             << std::endl;
            //     std::cout << "++++ ins input: " << arg->name()
            //             << ", shape: " << arg->get_shape() << ", standard: " << std::boolalpha
            //             << arg->get_shape().standard() << std::endl;
            //     auto next_ins = std::next(ins);
            //     if(next_ins != last)
            //     {
            //         std::cout << "++++ next_ins name: " << next_ins->name()
            //                 << " shape: " << next_ins->get_shape() << std::endl;
            //     }
            //     auto prev_ins = std::next(ins);
            //     std::cout << "++++ prev_ins name: " << prev_ins->name()
            //             << " shape: " << prev_ins->get_shape() << std::endl;
            // }
            if(enabled(MIGRAPHX_TRACE_ELIMINATE_CONTIGUOUS{}))
            {
                std::cout << "eliminate_contiguous: ";
                m.debug_print(ins);
            }
            auto prev = arg->inputs().front();
            // std::cout << "@@@@ before replace: " << std::endl;
            // for (auto a: new_args)
            // {
            //     std::cout << "@@ name: " << a->name() << " shape: " << a->get_shape() << std::endl;
            // }
            replace(new_args, arg, prev);
            // std::cout << "@@@@ after replace: " << std::endl;
            // for(auto a : new_args)
            // {
            //     std::cout << "@@ name: " << a->name() << " shape: " << a->get_shape() << std::endl;
            // }
            if(try_compute_shape_print(ins, new_args, mod_args, 0, false) /*&& prev->get_shape().standard()*/)
            {
                if(is_gpu_cont)
                {
                    std::cout << "+++++ Replace: " << std::endl;
                    std::cout << "+++++ old:     " << arg->name() << ", shape: " << arg->get_shape() << std::endl;
                    std::cout << "+++++ new:     " << prev->name() << ", shape: " << prev->get_shape() << std::endl;
                }
                // if(is_gpu_cont)
                // {
                //     std::cout << "++++ gpu::contiguous block ++++" << std::endl;
                //     if (not prev->get_shape().standard())
                //     {
                //         std::cout << "NON STANDARD REPLACED" << std::endl;
                //     }
                //     std::cout << "++++ instr name: " << ins->name() << " shape: " << ins->get_shape() << " inputs:" << std::endl;
                //     for(auto a : args)
                //     {
                //         std::cout << "+++++ name: " << a->name() << ", shape: " << a->outputs().front()->get_shape()<< ", standard: " << std::boolalpha
                //                   << a->outputs().front()->get_shape().standard() << std::endl;
                //     }
                //     std::cout << "++++ Replace: " << std::endl;
                //     std::cout << "++++ old:     " << arg->name() << ", shape: " << arg->get_shape() << std::endl;
                //     std::cout << "++++ new:     " << prev->name() << ", shape: " << prev->get_shape() << std::endl;
                //     auto next_ins = std::next(ins);
                //     if(next_ins != last)
                //     {
                //         std::cout << "++++ next_ins name: " << next_ins->name() << " shape: " << next_ins->get_shape() << std::endl;
                //     }

                //     auto _args     = ins->inputs();
                //     auto _new_args = _args;
                //     auto _prev      = arg->inputs().front();
                //     replace(_new_args, arg, _prev);
                //     try_compute_shape_print(ins, new_args, mod_args);
                //     std::cout << "+++++++++++++++++++++++++++++++" << std::endl;
                // }
                // else
                // {
                //     std::cout << "@@ Standard shape" << std::endl;
                // }
                instruction::replace_argument(ins, arg, prev);
                // replace_map[arg] = prev;
            }
            else if(prev->can_eval())
            {
                std::cout << "++ Can eval" << std::endl;
                const_instructions.push_back(arg);
            }
        }
    }

    // Perform static contiguous evaluations in parallel
    std::vector<argument> literals(const_instructions.size());
    par_for(const_instructions.size(), 1, [&](const auto i) {
        auto c    = op::contiguous{};
        auto prev = const_instructions[i]->inputs().front();
        // compute the output contiguous shape from the previous instruction shape
        shape computed_shape                   = c.compute_shape({prev->get_shape()});
        const std::vector<argument>& prev_eval = {prev->eval()};
        // prev_eval should not be used in make_compute_output_shape() as computed_shape is static
        auto co_shape = make_compute_output_shape(pack(c, computed_shape, prev_eval));
        literals[i]   = c.compute(co_shape, prev_eval);
    });

    // Replace static contiguous operations with a literal
    for(size_t i = 0; i < const_instructions.size(); i++)
    {
        auto l = m.add_literal(literals[i].get_shape(), literals[i].data());
        m.replace_instruction(const_instructions[i], l);
    }
}

void eliminate_contiguous::apply(module& m) const
{
    // Skip contiguous from splits first
    remove_contiguous(op_name, m, [](auto ins) {
        if(ins->name() != "slice")
            return true;
        return (ins->inputs().front()->outputs().size() == 1);
    });
    remove_contiguous(op_name, m, [](auto) { return true; });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
