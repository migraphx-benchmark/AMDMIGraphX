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
#ifndef MIGRAPHX_GUARD_OPERATORS_SQUEEZE_HPP
#define MIGRAPHX_GUARD_OPERATORS_SQUEEZE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/dyn_output.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct squeeze
{
    std::vector<int64_t> axes;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axes"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    void normalize_axes(std::vector<int64_t>& reduce_axes, int64_t ndim) const
    {
        for(auto& a : reduce_axes)
        {
            if(a < -ndim or a >= ndim)
                MIGRAPHX_THROW("Invalid axes for reduce: " + to_string(a) + ", input has " +
                               to_string(ndim) + " dimension(s)");

            if(a < 0)
                a += ndim;
        }
    }

    shape compute_shape_for_dynamic_input_shape(const std::vector<shape>& input_shapes) const
    {
        auto input_shape = input_shapes[0];
        if(input_shapes.size() == 1)
        {
            if(any_of(axes, [&](auto axis) { return input_shape.dyn_dims()[axis] != 1; }))
                MIGRAPHX_THROW(
                    "SQUEEZE: dynamic axis dimension should be equal to {1, 1, 0} or {1, 1, 1}");

            std::vector<shape::dynamic_dimension> dyn_dims = {};
            if(axes.empty())
            {
                std::copy_if(input_shape.dyn_dims().cbegin(),
                             input_shape.dyn_dims().cend(),
                             std::back_inserter(dyn_dims),
                             [&](auto dd) { return dd != 1; });
            }
            else
            {
                for(auto i : range(input_shape.ndim()))
                    if(not contains(axes, i))
                        dyn_dims.push_back(input_shape.dyn_dims()[i]);
            }
            return {input_shape.type(), dyn_dims};
        }
        else
        {
            auto axes_shape = input_shapes[1];
            auto dyn_dims   = input_shape.dyn_dims();
            std::cout << input_shape << std::endl;
            auto flat_axes_count = std::count_if(
                dyn_dims.begin(), dyn_dims.end(), [](const auto& dim) { return dim.min == 1; });
            std::cout << "flat_axes_count: " << flat_axes_count << std::endl;
            auto axes_size = axes_shape.dynamic() ? axes_shape.dyn_dims().front().min
                                                  : axes_shape.lens().front();
            std::cout << "axes_size: " << axes_size << std::endl;
            if(flat_axes_count < axes_size)
                MIGRAPHX_THROW("Lorem ipsum");

            return input_shape;
        }
    }

    shape compute_static_shape(const std::vector<shape>& input_shapes,
                               const std::vector<int64_t>& sq_axes) const
    {
        auto input_shape = input_shapes[0];
        auto type        = input_shape.type();
        auto old_lens    = input_shape.lens();
        auto old_strides = input_shape.strides();
        if(any_of(sq_axes, [&](auto axis) { return old_lens[axis] != 1; }))
            MIGRAPHX_THROW("SQUEEZE: static axis dimension should be equal to 1");

        std::vector<std::size_t> new_lens;
        std::vector<std::size_t> new_strides;
        if(sq_axes.empty())
        {
            for(auto i : range(old_lens.size()))
            {
                if(old_lens[i] != 1)
                {
                    new_lens.push_back(old_lens[i]);
                    new_strides.push_back(old_strides[i]);
                }
            }
        }
        else
        {
            for(auto i : range(old_lens.size()))
            {
                if(not contains(sq_axes, i))
                {
                    new_lens.push_back(old_lens[i]);
                    new_strides.push_back(old_strides[i]);
                }
            }
        }

        if(new_lens.empty())
            return shape{type};
        else
            return shape{type, new_lens, new_strides};
    }

    std::string name() const { return "squeeze"; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1, 2);

        if(inputs[0].dynamic())
            return compute_shape_for_dynamic_input_shape(inputs);
        else
            return compute_static_shape(inputs, axes);
    }

    argument compute(const dyn_output& dyn_out, std::vector<argument> args) const
    {
        if(args.size() == 1)
            return args[0].reshape(dyn_out.computed_shape);
        else
        {
            std::vector<int64_t> sq_axes;
            args[1].visit([&](auto&& s) { sq_axes.assign(s.begin(), s.end()); });
            normalize_axes(sq_axes, args[0].get_shape().ndim());
            auto output_shape = compute_static_shape(to_shapes(args), sq_axes);
            return args[0].reshape(output_shape);
        }
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
