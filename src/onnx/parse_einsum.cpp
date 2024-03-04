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
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>
#include <migraphx/stringutils.hpp>

#define DEBUG_PRINT 0
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_einsum : op_parser<parse_einsum>
{
    using string_vec   = std::vector<std::string>;
    using char_int_map = std::map<char, int>;

    std::vector<op_desc> operators() const { return {{"Einsum"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        return decompose_equation(info, args);
    }

    private:
    instruction_ref decompose_equation(const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        instruction_ref op;
        std::optional<instruction_ref> last_op;

        if(not contains(info.attributes, "equation"))
        {
            MIGRAPHX_THROW("Equation attribute is required");
        }

        std::string equation = info.attributes.at("equation").s();
#if DEBUG_PRINT == 1
        std::cout << "EQUATION: " << equation << std::endl;
#endif

        auto [terms, unique_labels, ellipses_ndim] = analyze_equation(equation, args);
        auto mat        = make_mapping_matrix(terms, unique_labels, ellipses_ndim);
        auto duplicates = look_for_duplicates(terms);

#if DEBUG_PRINT == 1
        std::cout << "Mapping matrix: " << std::endl;
        print_matrix(mat);
#endif

        std::tuple<int, int> mat_shape = {mat.size(), mat[0].size()};
        int full_dim                   = std::get<1>(mat_shape);

        std::vector<std::vector<int>> rows = full(2, full_dim, -1);

        for(auto arg_idx = 0; arg_idx < args.size(); ++arg_idx)
        {
#if DEBUG_PRINT == 1
            std::cout << i << "----------------------------------" << std::endl;
#endif
            op      = args[arg_idx];
            rows[1] = mat[arg_idx]; // compute output row

            auto tr_row    = mat[arg_idx];
            auto duplicate = duplicates[arg_idx];
            if(duplicate.size())
            {
                std::vector<std::tuple<int, std::vector<int>>> diag;
                for(auto [_, v] : duplicate)
                {
                    if(v.size() == 1)
                    {
                        continue;
                    }

                    diag.push_back({v[0], v});
                }

                op     = apply_diagonal(info, rows, op, diag);
                tr_row = rows[1];
            }

#if DEBUG_PRINT == 1
            std::cout << "Rows before unsqueeze_transpose: " << std::endl;
            print_matrix(rows);
#endif
            // Transpose so the labels in the term are ordered alphabetically
            op = unsqueeze_transpose(info, rows, op, tr_row);
#if DEBUG_PRINT == 1
            std::cout << "Rows after unsqueeze_transpose: " << std::endl;
            print_matrix(rows);
#endif

            // reduction
            std::vector<int> red;
            for(int d = 0; d < full_dim; ++d)
            {
                int max = colwise_comp(mat, d, arg_idx + 1, mat.size(), std::greater<int>{});
                if(max == -1 and rows[1][d] != -1 and rows[0][d] == -1)
                    red.push_back(d);
            }

            if(red.size())
            {
#if DEBUG_PRINT == 1
                std::cout << "Pre-dot axes for reduction: " << to_string_range(red) << std::endl;
#endif
                op = info.add_instruction(make_op("reduce_sum", {{"axes", red}}), op);
                // compute output row
                for(int r : red)
                    rows[1][r] = -1;
            }

            if(last_op)
            {
                // Label is present in current two terms, but not in the remainder of the equation
                std::vector<int> common_dims;
                // Label is present in only left term or both terms and somewhere in the remainder
                // of the equation
                std::vector<int> left;
                // Label is present in only right term or both terms and somewhere in the remainder
                // of the equation
                std::vector<int> right;

                for(int d = 0; d < full_dim; ++d)
                {
                    int min = colwise_comp(rows, d, 0, rows.size(), std::less<int>{});
                    // There is no -1 in the column, for the current two rows
                    // The label is present in both rows
                    if(min >= 0)
                    {
                        // There is at least 1 element that is not -1, for the remaining rows of the
                        // matrix.
                        // The label is present in at least one of the subsequent rows
                        int max =
                            colwise_comp(mat, d, arg_idx + 1, mat.size(), std::greater<int>{});
                        if(max >= 0)
                        {
                            left.push_back(d);
                            right.push_back(d);
                        }
                        else
                            common_dims.push_back(d);
                    }
                    // The label is missing in one or both of the rows
                    else
                    {
                        if(rows[0][d] >= 0)
                            left.push_back(d);
                        if(rows[1][d] >= 0)
                            right.push_back(d);
                    }
                }

                op = matmul(info, rows, last_op.value(), op, common_dims, left, right);
            }

            last_op = op;
            rows[0] = rows[1];
        }

        // finalize output
        if(*(std::max_element(mat[args.size()].begin(), mat[args.size()].end())) >= 0)
        {
            rows[1] = mat[args.size()];

            std::vector<int> red;
            for(int d = 0; d < full_dim; ++d)
            {
                if(rows[0][d] > 0 and rows[1][d] == -1)
                {
                    red.push_back(d);
                }
                else if(rows[0][d] == -1 && rows[1][d] >= 0)
                {
                    MIGRAPHX_THROW("Issue in equation");
                }
            }

            if(red.size())
            {
                op = info.add_instruction(make_op("reduce_sum", {{"axes", red}}), op);
                // compute output row
                for(int r : red)
                {
                    rows[1][r] = -1;
                }
            }

            op = transpose_squeeze(info, rows, op, mat[args.size()]);
        }

        return op;
    }

    instruction_ref apply_diagonal(const onnx_parser::node_info& info,
                                   std::vector<std::vector<int>>& rows,
                                   instruction_ref op,
                                   std::vector<std::tuple<int, std::vector<int>>> diag) const
    {
        if(diag.size() != 1)
        {
            MIGRAPHX_THROW("Not implemented with more than one duplicated indice");
        }

        auto diag0 = diag[0];

        auto axis = std::get<0>(diag0);
        auto axes = std::get<1>(diag0);

        // if(not contains(axes, axis))
        // {
        //     MIGRAPHX_THROW("Axis must be in axes");
        // }

        // std::vector<size_t> shape, new_shape;

        // int i = 0;
        // for(auto s : op->get_shape().lens())
        // {
        //     if(contains(axes, i))
        //     {
        //         if(i == axis)
        //         {
        //             shape.push_back(s);
        //             new_shape.push_back(s);
        //         }
        //         else
        //         {
        //             shape.push_back(1);
        //         }
        //     }
        //     else
        //     {
        //         shape.push_back(s);
        //         new_shape.push_back(s);
        //     }

        //     i += 1;
        // }

        auto ndim = rows[0].size();

        std::vector<int> batch_axes;
        for(int i = 0; i < ndim; ++i)
        {
            if(not contains(axes, i))
            {
                batch_axes.push_back(i);
            }
        }

        auto min_axes = *(std::min_element(axes.begin(), axes.end()));
        if(not std::all_of(
               batch_axes.begin(), batch_axes.end(), [=](int ba) { return ba < min_axes; }))
        {
            MIGRAPHX_THROW("Currently batch axes have to be partitioned to the left");
        }

        auto op_shape = op->get_shape().lens();

        if(not std::all_of(axes.begin(), axes.end(), [op_shape, axis](int a) {
               return op_shape[axis] == op_shape[a];
           }))
        {
            MIGRAPHX_THROW("All duplicated indices have to be the same dimension");
        }

        auto batch_size = 1;
        for(auto ba : batch_axes)
        {
            batch_size *= op_shape[ba];
        }

        std::vector<size_t> indices;

        for(int batch = 0; batch < batch_size; ++batch)
        {
            for(int i = 0; i < op_shape[axis]; ++i)
            {
                std::vector<size_t> index(axes.size(), static_cast<size_t>(i));
                indices.insert(indices.end(), index.begin(), index.end());
            }
        }

        auto indices_arg = info.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int64_type,
                            {static_cast<unsigned long>(batch_size), op_shape[axis], axes.size()}},
            indices});

        std::cout << "INDICES: " << std::endl;
        for(auto ind : indices)
        {
            std::cout << ind << " ";
        }
        std::cout << std::endl;

        op = info.add_instruction(
            migraphx::make_op("gathernd", {{"batch_dims", batch_axes.size()}}), op, indices_arg);
        // compute output row
        std::vector<int> to_remove;
        for(auto [choice, choices] : diag)
        {
            for(auto ch : choices)
            {
                if(ch != choice)
                {
                    to_remove.push_back(ch);
                }
            }
            for(int i = 0; i < rows[1].size(); ++i)
            {
                if(contains(choices, rows[1][i]))
                {
                    if(rows[1][i] != choice)
                    {
                        rows[1][i] = choice;
                    }
                }
            }
        }
        std::sort(to_remove.begin(), to_remove.end());
        for(auto r : to_remove)
        {
            for(int i = 0; i < rows[1].size(); ++i)
            {
                if(rows[1][i] == r)
                {
                    MIGRAPHX_THROW("Unexpected result");
                }
                if(rows[1][i] > r)
                {
                    rows[1][i] -= 1;
                }
            }
        }

        return op;
    }

    instruction_ref unsqueeze_transpose(const onnx_parser::node_info& info,
                                        std::vector<std::vector<int>>& rows,
                                        instruction_ref op,
                                        std::vector<int> row) const
    {
        std::vector<std::tuple<int, int>> axes;
        std::vector<std::tuple<int, int>> perm;

        for(auto i = 0, p = 0; i < row.size(); ++i)
        {
            if(row[i] == -1)
                axes.push_back({p, i});
            else
            {
                ++p;
                perm.push_back({row[i], i});
            }
        }

        std::vector<int> s_axes;
        for(auto a : axes)
            s_axes.push_back(std::get<1>(a));

#if DEBUG_PRINT == 1
        std::cout << "Shape before unsqueeze: " << op->get_shape() << std::endl;
#endif
        op = info.add_instruction(make_op("unsqueeze", {{"axes", s_axes}}), op);
#if DEBUG_PRINT == 1
        std::cout << "Shape after unsqueeze: " << op->get_shape() << std::endl;
#endif
        // check output row
        for(int s_a : s_axes)
            if(rows[1][s_a] != -1)
                MIGRAPHX_THROW("Dimensions should be -1 in output row");

        std::sort(perm.begin(), perm.end(), [](auto lhs, auto rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });

        std::vector<int> new_perm(row.size());
        std::iota(new_perm.begin(), new_perm.end(), 0);

        for(auto i = 0, p = 0; i < row.size(); ++i)
        {
            if(row[i] == -1)
                continue;

            new_perm[std::get<1>(perm[p++])] = i;
        }

        op = apply_transpose_op(info, op, new_perm, rows[1]);

        return op;
    }

    instruction_ref transpose_squeeze(const onnx_parser::node_info& info,
                                      std::vector<std::vector<int>>& rows,
                                      instruction_ref op,
                                      std::vector<int> row_output) const
    {
        std::vector<std::tuple<int, int>> perm;
        std::vector<int> sq;

        for(auto i = 0; i < row_output.size(); ++i)
        {
            if(row_output[i] == -1)
                sq.push_back(i);
            else
                perm.push_back({row_output[i], i});
        }

        std::sort(perm.begin(), perm.end(), [](auto lhs, auto rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });

        std::vector<int> new_perm(rows[1].size());
        std::iota(new_perm.begin(), new_perm.end(), 0);

        for(auto i = 0, p = 0; i < row_output.size(); ++i)
        {
            if(row_output[i] == -1)
                continue;

            new_perm[i] = std::get<1>(perm[p++]);
        }

        op = apply_transpose_op(info, op, new_perm, rows[1]);

        if(sq.size())
        {
            op = info.add_instruction(make_op("squeeze", {{"axes", sq}}), op);
            // compute output row
            for(int a : sq)
                rows[1][a] = -1;
        }

        return op;
    }

    instruction_ref matmul(const onnx_parser::node_info& info,
                           std::vector<std::vector<int>>& rows,
                           instruction_ref op1,
                           instruction_ref op2,
                           std::vector<int> axes,
                           std::vector<int> left,
                           std::vector<int> right) const
    {
#if DEBUG_PRINT == 1
        std::cout << "Matmul" << std::endl;
        std::cout << "Axes: " << migraphx::to_string_range(axes) << std::endl;
        std::cout << "Left: " << migraphx::to_string_range(left) << std::endl;
        std::cout << "Right: " << migraphx::to_string_range(right) << std::endl;
        std::cout << "Matmul rows:\n";
        print_matrix(rows);
#endif

        int ndim = rows[0].size();

        if(set_intersection(axes, left).size() != 0 or set_intersection(axes, right).size() != 0)
            MIGRAPHX_THROW("axes and right or left have axes in common");

        std::vector<int> all_axes = set_union(set_union(left, right), axes);

        // Labels that are both in left and right, and not in all_axes
        // Given previous context:
        // Labels present in both terms and somewhere in the remainder of the equation
        std::vector<int> common_axes = set_intersection(left, right);
        for(int i = 0; i < ndim; ++i)
            if(not contains(all_axes, i))
                common_axes.push_back(i);
        std::sort(common_axes.begin(), common_axes.end());

#if DEBUG_PRINT == 1
        std::cout << "All axes: " << to_string_range(all_axes) << std::endl;
        std::cout << "Common axes: " << to_string_range(common_axes) << std::endl;
#endif

        // ReduceSum
        // All labels present in rows[0]
        std::vector<int> has_dim;
        for(int i = 0; i < rows[0].size(); ++i)
            if(rows[0][i] >= 0)
                has_dim.push_back(i);

        std::vector<int> right_no_left = set_difference(
            set_intersection(right, has_dim), set_intersection(right, set_union(left, axes)));

        if(right_no_left.size())
        {
            std::sort(right_no_left.begin(), right_no_left.end());
            op1 = info.add_instruction(make_op("reduce_sum", {{"axes", right_no_left}}), op1);
            // compute output row
            for(int r : right_no_left)
                rows[0][r] = -1;
        }

        has_dim.clear();
        for(int i = 0; i < rows[1].size(); ++i)
            if(rows[1][i] >= 0)
                has_dim.push_back(i);

        std::vector<int> left_no_right = set_difference(
            set_intersection(left, has_dim), set_intersection(left, set_union(right, axes)));

        if(left_no_right.size())
        {
            std::sort(left_no_right.begin(), left_no_right.end());
            op2 = info.add_instruction(make_op("reduce_sum", {{"axes", left_no_right}}), op2);
            // compute output row
            for(int r : left_no_right)
                rows[1][r] = -1;
        }

        const auto ignore_axes = set_symmetric_difference(left, right);
        auto perm              = concat_vectors(common_axes, ignore_axes, axes);

        const auto perm_for_side = [&](const auto& labels) {
            std::vector<int> ret;
            for(int i = 0; i < perm.size(); ++i)
                if(contains(labels, perm[i]))
                    ret.push_back(i);
            return ret;
        };
        const auto perm_left  = perm_for_side(left);
        const auto perm_right = perm_for_side(right);

#if DEBUG_PRINT == 1
        std::cout << "Perm: " << to_string_range(perm) << std::endl;
        std::cout << "Perm_left: " << to_string_range(perm_left) << std::endl;
        std::cout << "Perm_right: " << to_string_range(perm_right) << std::endl;
#endif

        // Transpose so labels are ordered according to category
        op1 = apply_transpose_op(info, op1, perm, rows[0]);
        op2 = apply_transpose_op(info, op2, perm, rows[1]);

#if DEBUG_PRINT == 1
        std::cout << "Rows after transpose before reshape:\n";
        print_matrix(rows);
#endif

        // Axes = common_dims -> Label present only in current two terms(label category 1)
        std::vector<int> new_axes(axes.size());
        std::iota(new_axes.begin(), new_axes.end(), ndim - axes.size());

        // common_axes -> labels present in both terms and remainder of eq(label category -1)
        std::vector<int> new_common_axes(common_axes.size());
        std::iota(new_common_axes.begin(), new_common_axes.end(), 0);

        // Labels that are not in left or right
        // common_axes is the intersection of left and right, if a label is in neither of
        // these, surely it cannot be in common_axes, making the check superfluous?
        std::vector<int> not_in_both;
        for(int i = 0; i < ndim; ++i)
        {
            if(not contains(left, i) and not contains(right, i) and not contains(common_axes, i))
                not_in_both.push_back(i);
        }

#if DEBUG_PRINT == 1
        std::cout << "new_axes: " << to_string_range(new_axes) << std::endl;
        std::cout << "new_common_axes: " << to_string_range(new_common_axes) << std::endl;
        std::cout << "not_in_both: " << to_string_range(not_in_both) << std::endl;
#endif

        instruction_ref op =
            batch_dot(info, rows, op1, op2, new_common_axes, {}, new_axes, perm_left, perm_right);

        // Transpose again
        std::vector<int> ordered_axes = common_axes;
        std::copy_if(left.begin(), left.end(), std::back_inserter(ordered_axes), [=](int el) {
            return not contains(right, el);
        });
        std::copy_if(right.begin(), right.end(), std::back_inserter(ordered_axes), [=](int el) {
            return not contains(left, el);
        });
        std::copy(not_in_both.begin(), not_in_both.end(), std::back_inserter(ordered_axes));

#if DEBUG_PRINT == 1
        std::cout << "ordered_axes: " << to_string_range(ordered_axes) << std::endl;
#endif
        std::vector<std::tuple<int, int>> rev_perm;
        for(auto i = 0; i < ordered_axes.size(); ++i)
            rev_perm.push_back({ordered_axes[i], i});

        std::sort(rev_perm.begin(), rev_perm.end(), [](auto lhs, auto rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });

        perm.clear();
        for(auto p : rev_perm)
            perm.push_back(std::get<1>(p));

        op = apply_transpose_op(info, op, perm, rows[1]);

        return op;
    }

    instruction_ref batch_dot(const onnx_parser::node_info& info,
                              std::vector<std::vector<int>>& rows,
                              instruction_ref op1,
                              instruction_ref op2,
                              std::vector<int> batch_axes,
                              std::vector<int> keep_axes,
                              std::vector<int> sum_axes,
                              std::vector<int> left,
                              std::vector<int> right) const
    {
        if(op1->get_shape().ndim() != op2->get_shape().ndim())
        {
            MIGRAPHX_THROW("batch_dot input tensors need to have the same number of dimensions");
        }

#if DEBUG_PRINT == 1
        std::cout << "BATCH-DOT" << std::endl;
        std::cout << "Batch axes: " << migraphx::to_string_range(batch_axes) << std::endl;
        std::cout << "Keep axes: " << migraphx::to_string_range(keep_axes) << std::endl;
        std::cout << "Sum axes: " << migraphx::to_string_range(sum_axes) << std::endl;
        std::cout << "Left: " << migraphx::to_string_range(left) << std::endl;
        std::cout << "Right: " << migraphx::to_string_range(right) << std::endl;
        std::cout << "Batch-dot rows:\n";
        print_matrix(rows);
#endif

        auto common_labels = set_union(batch_axes, sum_axes);
#if DEBUG_PRINT == 1
        std::cout << "Common labels: " << migraphx::to_string_range(common_labels) << std::endl;
#endif
        std::tie(op1, op2) = apply_broadcast_op(info, op1, op2, common_labels);

        auto op1_shape = op1->get_shape().lens();
        auto op2_shape = op2->get_shape().lens();

        auto calc_dim = [](const auto& axes, const auto& lens) {
            return std::accumulate(
                axes.begin(), axes.end(), 1, [&](auto acc, auto l) { return acc *= lens[l]; });
        };
        std::array<int, 3> dims1{
            calc_dim(batch_axes, op1_shape), -1, calc_dim(sum_axes, op1_shape)};
        std::array<int, 3> dims2{
            calc_dim(batch_axes, op2_shape), -1, calc_dim(sum_axes, op2_shape)};

        op1 = info.add_instruction(make_op("reshape", {{"dims", dims1}}), op1);
        op2 = info.add_instruction(make_op("reshape", {{"dims", dims2}}), op2);
        op2 = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1}}}), op2);
        instruction_ref dot = info.add_instruction(make_op("dot"), op1, op2);

        std::vector<int> new_shape;
        for(int i : batch_axes)
            new_shape.push_back(std::max(op1_shape[i], op2_shape[i]));

        for(int i : left)
            if(not contains(batch_axes, i))
                new_shape.push_back(op1_shape[i]);

        for(int i : right)
            if(not contains(batch_axes, i))
                new_shape.push_back(op2_shape[i]);

        while(new_shape.size() < op1_shape.size())
            new_shape.push_back(1);

        auto op = info.add_instruction(make_op("reshape", {{"dims", new_shape}}), dot);
        // compute output row
        std::transform(
            rows[0].begin(), rows[0].end(), rows[1].begin(), rows[1].begin(), [](int l, int r) {
                return std::max(l, r);
            });

        for(int a : sum_axes)
            if(not contains(right, a))
                rows[1][a] = -1;

        return op;
    }

    bool is_transpose_identity(std::vector<int> perm) const
    {
        for(auto i = 0u; i < perm.size(); ++i)
            if(perm[i] != i)
                return false;

        return true;
    }

    std::vector<std::vector<int>> full(int rows, int cols, int fill_value) const
    {
        std::vector<std::vector<int>> ret(rows);
        for(auto& row : ret)
            for(int i = 0; i < cols; ++i)
                row.push_back(fill_value);

        return ret;
    }

    int colwise_comp(std::vector<std::vector<int>> mat,
                     int col,
                     int begin,
                     int end,
                     std::function<bool(int, int)> pred) const
    {
        int ret = mat[begin][col];
        for(int i = begin + 1; i < end; ++i)
            if(pred(mat[i][col], ret))
                ret = mat[i][col];

        return ret;
    }

    std::vector<int> set_union(const std::vector<int>& lhs, const std::vector<int>& rhs) const
    {
        std::vector<int> ret;
        std::set_union(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(ret));

        return ret;
    }

    std::vector<int> set_intersection(const std::vector<int>& lhs,
                                      const std::vector<int>& rhs) const
    {
        std::vector<int> ret;
        std::set_intersection(
            lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(ret));

        return ret;
    }

    std::vector<int> set_difference(const std::vector<int>& lhs, const std::vector<int>& rhs) const
    {
        std::vector<int> ret;
        std::set_difference(
            lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(ret));

        return ret;
    }

    std::vector<int> set_symmetric_difference(const std::vector<int>& lhs,
                                              const std::vector<int>& rhs) const
    {
        std::vector<int> ret;
        std::set_symmetric_difference(
            lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(ret));

        return ret;
    }

    template <typename... Vecs>
    std::decay_t<std::tuple_element_t<0, std::tuple<Vecs...>>> concat_vectors(Vecs&&... vecs) const
    {
        size_t reserve_size = 0u;
        ([&](auto&& vec) { reserve_size += vec.size(); }(std::forward<Vecs>(vecs)), ...);

        std::decay_t<std::tuple_element_t<0, std::tuple<Vecs...>>> ret;
        ret.reserve(reserve_size);

        ([&](auto&& vec) { ret.insert(ret.end(), vec.begin(), vec.end()); }(
             std::forward<Vecs>(vecs)),
         ...);

        return ret;
    }

    std::tuple<string_vec, std::string, size_t>
    analyze_equation(std::string_view equation, const std::vector<instruction_ref>& args) const
    {
        std::tuple<string_vec, std::string, size_t> ret;
        auto& [terms, unique_labels, ellipses_ndim] = ret;

        auto [input_terms, output_term, label_count, explicit_form] = parse_equation(equation);

        ellipses_ndim = validate_input_terms(input_terms, args);
        if(not output_term.empty())
            validate_output_term(output_term, label_count, ellipses_ndim);
        else if(not explicit_form)
            output_term = generate_output_term(label_count, ellipses_ndim);

        terms = std::move(input_terms);
        terms.emplace_back(std::move(output_term));
        for(auto [l, _] : label_count)
            unique_labels += l;

        return ret;
    }

    std::vector<std::vector<int>> make_mapping_matrix(const string_vec& terms,
                                                      std::string_view unique_labels,
                                                      size_t ellipses_ndim) const
    {
        std::map<char, int> label_to_column;
        for(auto i = 0; i < unique_labels.size(); ++i)
            label_to_column[unique_labels[i]] = i;

        std::vector<std::vector<int>> mat =
            full(terms.size(), unique_labels.size() + ellipses_ndim, -1);

        for(auto i = 0; i < terms.size(); ++i)
        {
            const auto& term = terms[i];
            int col_id       = 0;
            for(auto j = 0; j < term.size(); ++j)
            {
                if(term[j] == '*')
                {
                    std::iota(mat[i].end() - ellipses_ndim, mat[i].end(), col_id);
                    col_id += ellipses_ndim;
                }
                else
                    mat[i][label_to_column[term[j]]] = col_id++;
            }
        }

        return mat;
    }

    std::vector<std::map<char, std::vector<int>>> look_for_duplicates(string_vec terms) const
    {
        std::vector<std::map<char, std::vector<int>>> duplicates;
        for(auto term : terms)
        {
            if(term.size() == std::set<char>(term.begin(), term.end()).size())
            {
                duplicates.push_back({});
                continue;
            }

            std::map<char, std::vector<int>> counts;
            int i = 0;
            for(char c : term)
            {
                counts[c].push_back(i++);
            }
            duplicates.push_back(counts);
        }

        return duplicates;
    }

    std::tuple<std::vector<std::string>, std::string, std::map<char, int>, bool>
    parse_equation(std::string_view equation) const
    {
        std::tuple<std::vector<std::string>, std::string, std::map<char, int>, bool> ret;
        auto& [input_terms, output_term, label_count, explicit_form] = ret;

        std::string term;
        bool has_ellipsis = false;
        explicit_form     = false;

        for(int i = 0; i < equation.size(); ++i)
        {
            const char c = equation[i];
            switch(c)
            {
            case ' ': break;
            case '-':
                if(explicit_form)
                    MIGRAPHX_THROW("Einsum equation has multiple '->' symbols");
                if(i + 1 >= equation.size() || equation[i + 1] != '>')
                    MIGRAPHX_THROW("Invalid '->' in einsum equation");

                ++i;
                explicit_form = true;
                [[fallthrough]];
            case ',':
                has_ellipsis = false;
                input_terms.emplace_back(term);
                term.clear();
                break;
            case '.':
                if(has_ellipsis)
                    MIGRAPHX_THROW("Ellipsis can only appear once per einsum equation term");

                if(i + 2 >= equation.size() || equation[i + 1] != '.' || equation[i + 2] != '.')
                    MIGRAPHX_THROW("Incomplete ellipsis in einsum equation " +
                                   std::string(equation));

                i += 2;
                has_ellipsis = true;
                term += '*';
                break;
            default:
                if(!std::isalpha(c))
                    MIGRAPHX_THROW(std::string("Invalid character '") + c +
                                   "' in einsum equation term");

                term += c;
                if(not explicit_form)
                    ++label_count[c];
            }
        }

        if(explicit_form)
            output_term = term;
        else
            input_terms.push_back(term);

        return ret;
    }

    std::string generate_output_term(const char_int_map& label_count, size_t ellipsis_ndim) const
    {
        std::string output_term = ellipsis_ndim != 0 ? "*" : "";
        for(const auto [label, count] : label_count)
            if(count == 1)
                output_term += label;

        return output_term;
    }

    void validate_output_term(std::string_view output_term,
                              const char_int_map& label_count,
                              size_t ellipses_ndim) const
    {
        for(const auto label : output_term)
            if(not contains(label_count, label) and label != '*')
                MIGRAPHX_THROW("Output term contains label " + std::to_string(label) +
                               ", which is not present in any of the input terms");

        if(ellipses_ndim != 0 and not contains(output_term, "*"))
            MIGRAPHX_THROW(
                "Output term does not contain ellipsis (...) even though an input term does");
    }

    size_t validate_input_terms(const string_vec& input_terms,
                                const std::vector<instruction_ref>& args) const
    {
        if(input_terms.size() != args.size())
            MIGRAPHX_THROW(
                "Number of terms in the input equation - " + std::to_string(input_terms.size()) +
                " does not match the number of input tensors " + std::to_string(args.size()));

        auto global_ellipses_dims = 0u;
        for(auto i = 0u; i < args.size(); ++i)
        {
            const auto& term = input_terms[i];
            const auto dims  = args[i]->get_shape().lens();
            const auto rank  = dims.size();

            auto current_dim = 0u;
            for(const auto l : term)
            {
                if(l == '*')
                {
                    auto ellipses_dims = rank - term.size() + 1;
                    if(global_ellipses_dims > 0 and ellipses_dims != global_ellipses_dims)
                        MIGRAPHX_THROW("Every occurrence of ellipsis in the equation must "
                                       "represent the same number of dimensions");
                    global_ellipses_dims = ellipses_dims;
                    current_dim += ellipses_dims;
                }
                else
                    ++current_dim;
            }

            if(current_dim != rank)
                MIGRAPHX_THROW("Number of labels in " + std::to_string(i + 1) + ". input_term (" +
                               term + ") does not match the rank (" + std::to_string(rank) +
                               ") of corresponding input");
        }

        return global_ellipses_dims;
    }

    std::pair<instruction_ref, instruction_ref>
    apply_broadcast_op(const onnx_parser::node_info& info,
                       instruction_ref opl,
                       instruction_ref opr,
                       const std::vector<int>& common_labels) const
    {
        std::pair<instruction_ref, instruction_ref> ret;

        auto llens = opl->get_shape().lens();
        auto rlens = opr->get_shape().lens();

        bool lbc = false;
        bool rbc = false;
        for(auto l : common_labels)
        {
            if(llens[l] == 1 and rlens[l] == 1)
                continue;

            if(llens[l] == 1)
            {
                lbc      = true;
                llens[l] = rlens[l];
            }

            if(rlens[l] == 1)
            {
                rbc      = true;
                rlens[l] = llens[l];
            }
        }

        if(lbc)
            opl = info.add_instruction(make_op("multibroadcast", {{"out_lens", llens}}), opl);
        if(rbc)
            opr = info.add_instruction(make_op("multibroadcast", {{"out_lens", rlens}}), opr);

        ret.first  = opl;
        ret.second = opr;
        return ret;
    }

    instruction_ref apply_transpose_op(const onnx_parser::node_info& info,
                                       instruction_ref op,
                                       const std::vector<int>& perm,
                                       std::vector<int>& row) const
    {
        if(is_transpose_identity(perm))
            return op;

        op = info.add_instruction(make_op("transpose", {{"permutation", perm}}), op);
        // compute output row
        auto cpy = row;
        for(auto i = 0u; i < perm.size(); ++i)
            row[i] = cpy[perm[i]];

        return op;
    }

    void print_matrix(std::vector<std::vector<int>>& mat) const
    {
        for(auto& row : mat)
        {
            for(auto e : row)
            {
                std::cout.width(2);
                std::cout << std::right << e << " ";
            }
            std::cout << std::endl;
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
