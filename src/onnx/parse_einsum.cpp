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

        int i = 0;
        for(instruction_ref arg : args)
        {
#if DEBUG_PRINT == 1
            std::cout << i << "----------------------------------" << std::endl;
#endif
            op      = arg;
            rows[1] = mat[i]; // compute output row

            auto tr_row    = mat[i];
            auto duplicate = duplicates[i];
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
                    // TODO
                }
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
                int max = colwise_comp(mat, d, i + 1, mat.size(), std::greater<int>{});
                if(max == -1 and rows[1][d] != -1 and rows[0][d] == -1)
                {
                    red.push_back(d);
                }
            }

            if(red.size())
            {
#if DEBUG_PRINT == 1
                std::cout << "Pre-dot axes for reduction: " << to_string_range(red) << std::endl;
#endif
                op = info.add_instruction(make_op("reduce_sum", {{"axes", red}}), op);
                // compute output row
                for(int r : red)
                {
                    rows[1][r] = -1;
                }
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
                        int max = colwise_comp(mat, d, i + 1, mat.size(), std::greater<int>{});
                        if(max >= 0)
                        {
                            left.push_back(d);
                            right.push_back(d);
                        }
                        else
                        {
                            common_dims.push_back(d);
                        }
                    }
                    // The label is missing in one or both of the rows
                    else
                    {
                        if(rows[0][d] >= 0)
                        {
                            left.push_back(d);
                        }
                        if(rows[1][d] >= 0)
                        {
                            right.push_back(d);
                        }
                    }
                }

                op = matmul(info, rows, last_op.value(), op, common_dims, left, right);
            }

            last_op = op;
            rows[0] = rows[1];

            i += 1;
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

    instruction_ref unsqueeze_transpose(const onnx_parser::node_info& info,
                                        std::vector<std::vector<int>>& rows,
                                        instruction_ref op,
                                        std::vector<int> row) const
    {
        std::vector<std::tuple<int, int>> axes;
        int p = 0;
        std::vector<std::tuple<int, int>> perm;

        int i = 0;
        for(int r : row)
        {
            if(r == -1)
            {
                axes.push_back({p, i++});
            }
            else
            {
                p += 1;
                perm.push_back({r, i++});
            }
        }

        std::vector<int> s_axes;
        for(auto a : axes)
        {
            s_axes.push_back(std::get<1>(a));
        }

#if DEBUG_PRINT == 1
        std::cout << "Shape before unsqueeze: " << op->get_shape() << std::endl;
#endif
        op = info.add_instruction(make_op("unsqueeze", {{"axes", s_axes}}), op);
#if DEBUG_PRINT == 1
        std::cout << "Shape after unsqueeze: " << op->get_shape() << std::endl;
#endif
        // check output row
        for(int s_a : s_axes)
        {
            if(rows[1][s_a] != -1)
            {
                MIGRAPHX_THROW("Dimensions should be -1 in output row");
            }
        }

        std::sort(perm.begin(), perm.end(), [](auto lhs, auto rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });
        p = 0;

        std::vector<int> new_perm(row.size());
        std::iota(new_perm.begin(), new_perm.end(), 0);

        i = 0;
        for(int r : row)
        {
            if(r == -1)
            {
                i += 1;
                continue;
            }

            new_perm[std::get<1>(perm[p])] = i++;
            p += 1;
        }

        // *ik,kjl->ij*
        //
        // term1

        if(not is_transpose_identity(new_perm))
        {
#if DEBUG_PRINT == 1
            std::cout << "Shape before transpose: " << op->get_shape() << std::endl;
#endif
            op = info.add_instruction(make_op("transpose", {{"permutation", new_perm}}), op);
#if DEBUG_PRINT == 1
            std::cout << "Shape after transpose: " << op->get_shape() << std::endl;
#endif
            // compute output row
            auto cpy = rows[1];
            i        = 0;
            for(int np : new_perm)
            {
                rows[1][i++] = cpy[np];
            }
        }

        return op;
    }

    instruction_ref transpose_squeeze(const onnx_parser::node_info& info,
                                      std::vector<std::vector<int>>& rows,
                                      instruction_ref op,
                                      std::vector<int> row_output) const
    {
        std::vector<std::tuple<int, int>> perm;
        std::vector<int> sq;

        int i = 0;
        for(int d : row_output)
        {
            if(d == -1)
            {
                sq.push_back(i++);
            }
            else
            {
                perm.push_back({d, i++});
            }
        }

        std::sort(perm.begin(), perm.end(), [](auto lhs, auto rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });

        std::vector<int> new_perm(rows[1].size());
        std::iota(new_perm.begin(), new_perm.end(), 0);

        int p = 0;

        i = 0;
        for(int d : row_output)
        {
            if(d == -1)
            {
                i += 1;
                continue;
            }

            new_perm[i++] = std::get<1>(perm[p]);
            p += 1;
        }

        if(not is_transpose_identity(new_perm))
        {
            op = info.add_instruction(make_op("transpose", {{"permutation", new_perm}}), op);
            // compute output row
            auto cpy = rows[1];
            i        = 0;
            for(int np : new_perm)
            {
                rows[1][i++] = cpy[np];
            }
        }

        if(sq.size())
        {
            op = info.add_instruction(make_op("squeeze", {{"axes", sq}}), op);
            // compute output row
            for(int a : sq)
            {
                rows[1][a] = -1;
            }
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

        if(not(set_intersection(axes, left).size() == 0 and
               set_intersection(axes, right).size() == 0))
        {
            MIGRAPHX_THROW("Not implemented");
        }

        if(set_intersection(axes, left).size() == 0 and set_intersection(axes, right).size() == 0)
        {
            std::vector<int> all_axes = set_union(set_union(left, right), axes);

            // Labels that are both in left and right, and not in all_axes
            // Given previous context:
            // Labels present in both terms and somewhere in the remainder of the equation
            std::vector<int> common_axes = set_intersection(left, right);
            for(int i = 0; i < ndim; ++i)
            {
                if(std::find(all_axes.begin(), all_axes.end(), i) == all_axes.end())
                {
                    common_axes.push_back(i);
                }
            }
            std::sort(common_axes.begin(), common_axes.end());

#if DEBUG_PRINT == 1
            std::cout << "All axes: " << to_string_range(all_axes) << std::endl;
            std::cout << "Common axes: " << to_string_range(common_axes) << std::endl;
#endif

            // ReduceSum
            // All labels present in rows[0]
            std::vector<int> has_dim;
            for(int i = 0; i < rows[0].size(); ++i)
            {
                if(rows[0][i] >= 0)
                {
                    has_dim.push_back(i);
                }
            }

            std::vector<int> right_no_left = set_difference(
                set_intersection(right, has_dim), set_intersection(right, set_union(left, axes)));

            if(right_no_left.size())
            {
                std::sort(right_no_left.begin(), right_no_left.end());
                op1 = info.add_instruction(make_op("reduce_sum", {{"axes", right_no_left}}), op1);
                // compute output row
                for(int r : right_no_left)
                {
                    rows[0][r] = -1;
                }
            }

            has_dim.clear();
            for(int i = 0; i < rows[1].size(); ++i)
            {
                if(rows[1][i] >= 0)
                {
                    has_dim.push_back(i);
                }
            }

            std::vector<int> left_no_right = set_difference(
                set_intersection(left, has_dim), set_intersection(left, set_union(right, axes)));

            if(left_no_right.size())
            {
                std::sort(left_no_right.begin(), left_no_right.end());
                op2 = info.add_instruction(make_op("reduce_sum", {{"axes", left_no_right}}), op2);
                // compute output row
                for(int r : left_no_right)
                {
                    rows[1][r] = -1;
                }
            }

            // Transpose
            std::vector<std::tuple<int, int>> i_axes;
            for(int i = 0; i < ndim; ++i)
            {
                int first;
                // Label present in both terms and somewhere in the remainder of equation
                if(std::find(common_axes.begin(), common_axes.end(), i) != common_axes.end())
                {
                    first = -1;
                }
                // Label present ONLY in current two terms
                else if(std::find(axes.begin(), axes.end(), i) != axes.end())
                {
                    first = 1;
                }
                // Label present only in remainder of equation
                else
                {
                    first = 0;
                }
                i_axes.push_back({first, i});
            }

            std::sort(i_axes.begin(), i_axes.end(), [](auto lhs, auto rhs) {
                return std::get<0>(lhs) < std::get<0>(rhs);
            });

            // Label permutation in accordance to their category{-1, 0, 1}
            std::vector<int> perm;
            for(auto _ : i_axes)
            {
                perm.push_back(std::get<1>(_));
            }

            // Label permutation in accordance to their category that are surely present in the left
            // term
            std::vector<int> perm_left;
            for(int i = 0; i < perm.size(); ++i)
            {
                if(std::find(left.begin(), left.end(), perm[i]) != left.end())
                {
                    perm_left.push_back(i);
                }
            }

            // Label permutation in accordance to their category that are surely present in the
            // right term
            std::vector<int> perm_right;
            for(int i = 0; i < perm.size(); ++i)
            {
                if(std::find(right.begin(), right.end(), perm[i]) != right.end())
                {
                    perm_right.push_back(i);
                }
            }

#if DEBUG_PRINT == 1
            std::cout << "Perm: " << to_string_range(perm) << std::endl;
            std::cout << "Perm_left: " << to_string_range(perm_left) << std::endl;
            std::cout << "Perm_right: " << to_string_range(perm_right) << std::endl;
#endif

            // Transpose so labels are ordered according to category
            if(!is_transpose_identity(perm))
            {
                op1 = info.add_instruction(make_op("transpose", {{"permutation", perm}}), op1);
                // compute output row
                auto cpy = rows[0];
                int i    = 0;
                for(int p : perm)
                {
                    rows[0][i++] = cpy[p];
                }

                op2 = info.add_instruction(make_op("transpose", {{"permutation", perm}}), op2);
                // compute output row
                cpy = rows[1];
                i   = 0;
                for(int p : perm)
                {
                    rows[1][i++] = cpy[p];
                }
            }

#if DEBUG_PRINT == 1
            std::cout << "Rows after transpose before reshape:\n";
            print_matrix(rows);
#endif

            // Reshape
            std::vector<int> all_axes2(ndim);
            std::iota(all_axes2.begin(), all_axes2.end(), 0);

            std::vector<int> new_axes;
            // Axes = common_dims -> Label present only in current two terms(label category 1)
            if(axes.size() > 0)
            {
                std::copy(
                    all_axes2.end() - axes.size(), all_axes2.end(), std::back_inserter(new_axes));
            }

            // common_axes -> labels present in both terms and remainder of eq(label category -1)
            std::vector<int> new_common_axes;
            std::copy(all_axes2.begin(),
                      all_axes2.begin() + common_axes.size(),
                      std::back_inserter(new_common_axes));

            // Labels that are not in left or right
            // common_axes is the intersection of left and right, if a label is in neither of
            // these, surely it cannot be in common_axes, making the check superfluous?
            std::vector<int> not_in_both;
            for(int i = 0; i < ndim; ++i)
            {
                if(std::find(left.begin(), left.end(), i) == left.end() and
                   std::find(right.begin(), right.end(), i) == right.end() and
                   std::find(common_axes.begin(), common_axes.end(), i) == common_axes.end())
                {
                    not_in_both.push_back(i);
                }
            }

#if DEBUG_PRINT == 1
            std::cout << "new_axes: " << to_string_range(new_axes) << std::endl;
            std::cout << "new_common_axes: " << to_string_range(new_common_axes) << std::endl;
            std::cout << "not_in_both: " << to_string_range(not_in_both) << std::endl;
#endif

            instruction_ref op = batch_dot(
                info, rows, op1, op2, new_common_axes, {}, new_axes, perm_left, perm_right);

            // Transpose again
            std::vector<int> ordered_axes = common_axes;
            std::copy_if(left.begin(), left.end(), std::back_inserter(ordered_axes), [=](int el) {
                return std::find(right.begin(), right.end(), el) == right.end();
            });
            std::copy_if(right.begin(), right.end(), std::back_inserter(ordered_axes), [=](int el) {
                return std::find(left.begin(), left.end(), el) == left.end();
            });
            std::copy(not_in_both.begin(), not_in_both.end(), std::back_inserter(ordered_axes));

#if DEBUG_PRINT == 1
            std::cout << "ordered_axes: " << to_string_range(ordered_axes) << std::endl;
#endif

            std::vector<std::tuple<int, int>> rev_perm;
            int i = 0;
            for(int a : ordered_axes)
            {
                rev_perm.push_back({a, i++});
            }

            std::sort(rev_perm.begin(), rev_perm.end(), [](auto lhs, auto rhs) {
                return std::get<0>(lhs) < std::get<0>(rhs);
            });

            perm.clear();
            for(auto p : rev_perm)
            {
                perm.push_back(std::get<1>(p));
            }

            if(not is_transpose_identity(perm))
            {
                op = info.add_instruction(make_op("transpose", {{"permutation", perm}}), op);
                // compute output row
                auto cpy = rows[1];
                int i    = 0;
                for(int p : perm)
                {
                    rows[1][i++] = cpy[p];
                }
            }

            return op;
        }

        // TODO
        MIGRAPHX_THROW("axes and right or left have axes in common");
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

        // std::vector<int> left_term, right_term;
        // for(auto i = 0; i < rows[0].size(); ++i)
        //     if(rows[0][i] != -1)
        //         left_term.push_back(i);
        // for(auto i = 0; i < rows[1].size(); ++i)
        //     if(rows[1][i] != -1)
        //         right_term.push_back(i);
        // auto common_labels = set_intersection(left_term, right_term);
        auto common_labels = set_union(batch_axes, sum_axes);

#if DEBUG_PRINT == 1
        std::cout << "Left term: " << migraphx::to_string_range(left_term) << std::endl;
        std::cout << "Right term: " << migraphx::to_string_range(right_term) << std::endl;
        std::cout << "Common labels: " << migraphx::to_string_range(common_labels) << std::endl;
#endif

        std::vector<std::size_t> op1_shape = op1->get_shape().lens();
        std::vector<std::size_t> op2_shape = op2->get_shape().lens();

        for(auto l : common_labels)
        {
#if DEBUG_PRINT == 1
            std::cout << "Common label: " << l << std::endl;
#endif
            if(op1_shape[l] == 1 and op2_shape[l] == 1)
                continue;
            if(op1_shape[l] == 1)
            {
                op1_shape[l] = op2_shape[l];
                op1 =
                    info.add_instruction(make_op("multibroadcast", {{"out_lens", op1_shape}}), op1);
            }
            if(op2_shape[l] == 1)
            {
                op2_shape[l] = op1_shape[l];
                op2 =
                    info.add_instruction(make_op("multibroadcast", {{"out_lens", op2_shape}}), op2);
            }
        }

        int dim0 = 1;
        for(int i : batch_axes)
        {
            dim0 *= op1_shape[i];
        }

        int dim0b = 1;
        for(int i : batch_axes)
        {
            dim0b *= op2_shape[i];
        }

        int dimb = 1;
        if(keep_axes.empty())
        {
            dimb = -1;
        }
        else
        {
            for(int i : keep_axes)
            {
                dimb *= op1_shape[i];
            }
        }

        int dim1 = 1;
        for(int i : sum_axes)
        {
            dim1 *= op1_shape[i];
        }

        int dim2 = 1;
        for(int i : sum_axes)
        {
            dim2 *= op2_shape[i];
        }

        instruction_ref op1sh =
            info.add_instruction(make_op("reshape", {{"dims", {dim0, dimb, dim1}}}), op1);

        instruction_ref op2sh =
            info.add_instruction(make_op("reshape", {{"dims", {dim0b, dimb, dim2}}}), op2);

        instruction_ref dot;
        op2sh = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1}}}), op2sh);
        // op1sh = info.add_instruction(make_op("contiguous"), op1sh);
        // op2sh = info.add_instruction(make_op("contiguous"), op2sh);
        dot   = info.add_instruction(make_op("dot"), op1sh, op2sh);

        std::vector<int> new_shape;
        for(int i : batch_axes)
        {
            new_shape.push_back(std::max(op1_shape[i], op2_shape[i]));
        }
        for(int i : left)
        {
            if(std::find(batch_axes.begin(), batch_axes.end(), i) == batch_axes.end())
            {
                new_shape.push_back(op1_shape[i]);
            }
        }
        for(int i : right)
        {
            if(std::find(batch_axes.begin(), batch_axes.end(), i) == batch_axes.end())
            {
                new_shape.push_back(op2_shape[i]);
            }
        }

        while(new_shape.size() < op1_shape.size())
        {
            new_shape.push_back(1);
        }

        instruction_ref op = info.add_instruction(make_op("reshape", {{"dims", new_shape}}), dot);
        // compute output row
        std::transform(
            rows[0].begin(), rows[0].end(), rows[1].begin(), rows[1].begin(), [](int l, int r) {
                return std::max(l, r);
            });
        for(int a : sum_axes)
        {
            if(std::find(right.begin(), right.end(), a) == right.end())
            {
                rows[1][a] = -1;
            }
        }

        return op;
    }

    bool is_transpose_identity(std::vector<int> perm) const
    {
        std::vector<int> range(perm.size());
        std::iota(range.begin(), range.end(), 0);
        return perm == range;
    }

    std::vector<std::vector<int>> full(int rows, int cols, int fill_value) const
    {
        std::vector<std::vector<int>> ret(rows);
        for(auto& row : ret)
        {
            for(int i = 0; i < cols; ++i)
            {
                row.push_back(fill_value);
            }
        }
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
        {
            if(pred(mat[i][col], ret))
            {
                ret = mat[i][col];
            }
        }
        return ret;
    }

    std::vector<int> set_union(std::vector<int> lhs, std::vector<int> rhs) const
    {
        std::vector<int> ret;
        std::set_union(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(ret));
        return ret;
    }

    std::vector<int> set_intersection(std::vector<int> lhs, std::vector<int> rhs) const
    {
        std::vector<int> ret;
        std::set_intersection(
            lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(ret));
        return ret;
    }

    std::vector<int> set_difference(std::vector<int> lhs, std::vector<int> rhs) const
    {
        std::vector<int> ret;
        std::set_difference(
            lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(ret));
        return ret;
    }

    // EQUATION PARSING

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
                {
                    mat[i][label_to_column[term[j]]] = col_id++;
                }
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
                {
                    MIGRAPHX_THROW("Einsum equation has multiple '->' symbols");
                }
                if(i + 1 >= equation.size() || equation[i + 1] != '>')
                {
                    MIGRAPHX_THROW("Invalid '->' in einsum equation");
                }
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
                {
                    MIGRAPHX_THROW("Ellipsis can only appear once per einsum equation term");
                }
                if(i + 2 >= equation.size() || equation[i + 1] != '.' || equation[i + 2] != '.')
                {
                    MIGRAPHX_THROW("Incomplete ellipsis in einsum equation " +
                                   std::string(equation));
                }
                i += 2;
                has_ellipsis = true;
                term += '*';
                break;
            default:
                if(!std::isalpha(c))
                {
                    MIGRAPHX_THROW(std::string("Invalid character '") + c +
                                   "' in einsum equation term");
                }
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
        {
            if(not contains(label_count, label) and label != '*')
            {
                MIGRAPHX_THROW("Output term contains label " + std::to_string(label) +
                               ", which is not present in any of the input terms");
            }
        }
        if(ellipses_ndim != 0 and not contains(output_term, "*"))
        {
            MIGRAPHX_THROW(
                "Output term does not contain ellipsis (...) even though an input term does");
        }
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

// einsum_test.cpp:einsum_3d_opposite_broadcast_test graph without eliminate_contiguous applied
//
// @0 = check_context::migraphx::gpu::context -> float_type, {}, {}, target_id=0
// @1 = hip::hip_allocate_memory[shape=int8_type, {224}, {1},id=main:scratch] -> int8_type, {224}, {1}, target_id=0
// @2 = load[offset=32,end=56](@1) -> float_type, {1, 3, 2}, {6, 2, 1}, target_id=0
// x1 = @param:x1 -> float_type, {1, 3, 2}, {6, 2, 1}, target_id=0
// @4 = hip::copy_to_gpu(x1,@2) -> float_type, {1, 3, 2}, {6, 2, 1}, target_id=0
// @5 = load[offset=64,end=96](@1) -> float_type, {2, 1, 4}, {4, 4, 1}, target_id=0
// x2 = @param:x2 -> float_type, {2, 1, 4}, {4, 4, 1}, target_id=0
// @7 = hip::copy_to_gpu(x2,@5) -> float_type, {2, 1, 4}, {4, 4, 1}, target_id=0
// @8 = load[offset=0,end=32](@1) -> float_type, {2, 1, 4, 1}, {4, 4, 1, 1}, target_id=0
// @9 = unsqueeze[axes={1},steps={}](@7) -> float_type, {2, 1, 1, 4}, {4, 4, 4, 1}, target_id=0
// @10 = transpose[permutation={0, 1, 3, 2}](@9) -> float_type, {2, 1, 4, 1}, {4, 4, 1, 4}, target_id=0
// @11 = gpu::code_object[code_object=4112,symbol_name=contiguous_kernel,global=4,local=1024,](@10,@8) -> float_type, {2, 1, 4, 1}, {4, 4, 1, 1}, target_id=0
// @12 = load[offset=96,end=144](@1) -> float_type, {2, 3, 1, 2}, {6, 2, 2, 1}, target_id=0
// @13 = unsqueeze[axes={2},steps={}](@4) -> float_type, {1, 3, 1, 2}, {6, 2, 2, 1}, target_id=0
// @14 = multibroadcast[out_lens={2, 3, 1, 2},out_dyn_dims={}](@13) -> float_type, {2, 3, 1, 2}, {0, 2, 2, 1}, target_id=0
// @15 = gpu::code_object[code_object=4176,symbol_name=contiguous_kernel,global=6,local=1024,](@14,@12) -> float_type, {2, 3, 1, 2}, {6, 2, 2, 1}, target_id=0
// @16 = load[offset=48,end=96](@1) -> float_type, {2, 3, 1, 2}, {6, 2, 2, 1}, target_id=0
// @17 = gpu::code_object[code_object=4112,symbol_name=contiguous_kernel,global=6,local=1024,](@15,@16) -> float_type, {2, 3, 1, 2}, {6, 2, 2, 1}, target_id=0
// @18 = load[offset=160,end=224](@1) -> float_type, {2, 1, 4, 2}, {8, 8, 2, 1}, target_id=0
// @19 = multibroadcast[out_lens={2, 1, 4, 2},out_dyn_dims={}](@11) -> float_type, {2, 1, 4, 2}, {4, 4, 1, 0}, target_id=0
// @20 = gpu::code_object[code_object=4112,symbol_name=contiguous_kernel,global=8,local=1024,](@19,@18) -> float_type, {2, 1, 4, 2}, {8, 8, 2, 1}, target_id=0
// @21 = load[offset=96,end=160](@1) -> float_type, {2, 1, 4, 2}, {8, 8, 2, 1}, target_id=0
// @22 = gpu::code_object[code_object=4112,symbol_name=contiguous_kernel,global=8,local=1024,](@20,@21) -> float_type, {2, 1, 4, 2}, {8, 8, 2, 1}, target_id=0
// @23 = reshape_lazy[dims={2, -1, 2}](@17) -> float_type, {2, 3, 2}, {6, 2, 1}, target_id=0
// @24 = load[offset=0,end=48](@1) -> float_type, {2, 3, 2}, {6, 2, 1}, target_id=0
// @25 = gpu::code_object[code_object=4112,symbol_name=contiguous_kernel,global=6,local=1024,](@23,@24) -> float_type, {2, 3, 2}, {6, 2, 1}, target_id=0
// @26 = load[offset=160,end=224](@1) -> float_type, {2, 4, 2}, {8, 2, 1}, target_id=0
// @27 = reshape_lazy[dims={2, -1, 2}](@22) -> float_type, {2, 4, 2}, {8, 2, 1}, target_id=0
// @28 = gpu::code_object[code_object=4112,symbol_name=contiguous_kernel,global=8,local=1024,](@27,@26) -> float_type, {2, 4, 2}, {8, 2, 1}, target_id=0
// @29 = load[offset=48,end=112](@1) -> float_type, {2, 2, 4}, {8, 4, 1}, target_id=0
// @30 = transpose[permutation={0, 2, 1}](@28) -> float_type, {2, 2, 4}, {8, 1, 2}, target_id=0
// @31 = gpu::code_object[code_object=4112,symbol_name=contiguous_kernel,global=16,local=1024,](@30,@29) -> float_type, {2, 2, 4}, {8, 4, 1}, target_id=0
// @32 = load[offset=112,end=208](@1) -> float_type, {2, 3, 4}, {12, 4, 1}, target_id=0
// @33 = gpu::gemm[alpha=1,beta=0,compute_fp32=0,trans_batch=0,solution_idx=0](@25,@31,@32) -> float_type, {2, 3, 4}, {12, 4, 1}, target_id=0
// @34 = hip::copy_from_gpu(@33) -> float_type, {2, 3, 4}, {12, 4, 1}, target_id=0
// @35 = hip::sync_stream(@34) -> float_type, {2, 3, 4}, {12, 4, 1}, target_id=0
// @36 = @return(@35), target_id=0

// The graph after eliminate_contiguous is applied
// @0 = check_context::migraphx::gpu::context -> float_type, {}, {}, target_id=0
// @1 = hip::hip_allocate_memory[shape=int8_type, {160}, {1},id=main:scratch] -> int8_type, {160}, {1}, target_id=0
// @2 = load[offset=96,end=128](@1) -> float_type, {2, 1, 4}, {4, 4, 1}, target_id=0
// x2 = @param:x2 -> float_type, {2, 1, 4}, {4, 4, 1}, target_id=0
// @4 = hip::copy_to_gpu(x2,@2) -> float_type, {2, 1, 4}, {4, 4, 1}, target_id=0
// @5 = load[offset=128,end=152](@1) -> float_type, {1, 3, 2}, {6, 2, 1}, target_id=0
// x1 = @param:x1 -> float_type, {1, 3, 2}, {6, 2, 1}, target_id=0
// @7 = hip::copy_to_gpu(x1,@5) -> float_type, {1, 3, 2}, {6, 2, 1}, target_id=0
// @8 = load[offset=0,end=96](@1) -> float_type, {2, 3, 4}, {12, 4, 1}, target_id=0
// @9 = unsqueeze[axes={2},steps={}](@7) -> float_type, {1, 3, 1, 2}, {6, 2, 2, 1}, target_id=0
// @10 = multibroadcast[out_lens={2, 3, 1, 2},out_dyn_dims={}](@9) -> float_type, {2, 3, 1, 2}, {0, 2, 2, 1}, target_id=0
// @11 = reshape_lazy[dims={2, -1, 2}](@10) -> float_type, {2, 3, 2}, {0, 2, 1}, target_id=0
// @12 = unsqueeze[axes={1},steps={}](@4) -> float_type, {2, 1, 1, 4}, {4, 4, 4, 1}, target_id=0
// @13 = transpose[permutation={0, 1, 3, 2}](@12) -> float_type, {2, 1, 4, 1}, {4, 4, 1, 4}, target_id=0
// @14 = multibroadcast[out_lens={2, 1, 4, 2},out_dyn_dims={}](@13) -> float_type, {2, 1, 4, 2}, {4, 4, 1, 0}, target_id=0
// @15 = reshape_lazy[dims={2, -1, 2}](@14) -> float_type, {2, 4, 2}, {4, 1, 0}, target_id=0
// @16 = transpose[permutation={0, 2, 1}](@15) -> float_type, {2, 2, 4}, {4, 0, 1}, target_id=0
// @17 = gpu::gemm[alpha=1,beta=0,compute_fp32=0,trans_batch=0,solution_idx=0](@11,@16,@8) -> float_type, {2, 3, 4}, {12, 4, 1}, target_id=0
// @18 = hip::copy_from_gpu(@17) -> float_type, {2, 3, 4}, {12, 4, 1}, target_id=0
// @19 = hip::sync_stream(@18) -> float_type, {2, 3, 4}, {12, 4, 1}, target_id=0
// @20 = @return(@19), target_id=0