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
#include <migraphx/stringutils.hpp>

#define DEBUG 0
#define GRAPH 1

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
        return decompose_einsum_equation(info, args);
    }

    private:
    instruction_ref decompose_einsum_equation(const onnx_parser::node_info& info,
                                              const std::vector<instruction_ref>& args) const
    {
        instruction_ref op;
        std::optional<instruction_ref> last_op;

        if(not contains(info.attributes, "equation"))
        {
            MIGRAPHX_THROW("Equation attribute is required");
        }

        std::string equation = info.attributes.at("equation").s();

#if DEBUG > 0
        std::cout << "EQUATION: " << std::endl;
        std::cout << equation << std::endl;
#endif

        // auto [letters, mat, lengths] = analyse_einsum_equation(equation);
        auto [terms, unique_labels] = analyze_equation(equation, args);
        auto mat                    = make_mapping_matrix(terms, unique_labels);
        auto duplicates             = look_for_duplicates(terms);

        std::tuple<int, int> mat_shape = {mat.size(), mat[0].size()};
        int full_dim                   = std::get<1>(mat_shape);

        // if(letters.size() != full_dim)
        // {
        //     MIGRAPHX_THROW("Unexpected number of letters");
        // }

        // basic_verification(lengths, args, equation);

        std::vector<std::vector<int>> rows = full(2, full_dim, -1);

        int i = 0;
        for(instruction_ref arg : args)
        {
#ifdef GRAPH
            std::cout << "input: " << i << std::endl;
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
                }

                std::cout << "DIAG *****************" << std::endl;
                for(auto [_, v] : diag)
                {
                    std::cout << _ << ": ";
                    for(auto el : v)
                    {
                        std::cout << el << " ";
                    }
                    std::cout << std::endl;
                }
            }

            op = apply_transpose_reshape(info, rows, op, tr_row);

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

#if DEBUG == 1
            std::cout << "REDUCTION:" << std::endl;
            for(auto r : red)
            {
                std::cout << r << std::endl;
            }
#endif

            if(red.size())
            {
#ifdef GRAPH
                std::cout << "reduce_sum" << std::endl;
                std::cout << "> input shape: " << op->get_shape() << std::endl;
                std::cout << "> axes: ";
                for(auto _ : red)
                {
                    std::cout << _ << " ";
                }
                std::cout << std::endl;
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
                std::vector<int> common_dims;
                std::vector<int> left;
                std::vector<int> right;

                for(int d = 0; d < full_dim; ++d)
                {
                    int min = colwise_comp(rows, d, 0, rows.size(), std::less<int>{});
                    if(min >= 0)
                    {
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

#if DEBUG == 1
                std::cout << "ROWS:" << std::endl;
                for(int i = 0; i < rows.size(); ++i)
                {
                    for(int j = 0; j < rows[0].size(); ++j)
                    {
                        std::cout << rows[i][j] << " ";
                    }
                    std::cout << std::endl;
                }

                std::cout << "i: " << i << std::endl;

                std::cout << "COMMON DIMS:" << std::endl;
                for(int d : common_dims)
                {
                    std::cout << d << std::endl;
                }
                std::cout << "LEFT DIMS:" << std::endl;
                for(int d : left)
                {
                    std::cout << d << std::endl;
                }
                std::cout << "RIGHT DIMS:" << std::endl;
                for(int d : right)
                {
                    std::cout << d << std::endl;
                }
#endif

                op = apply_einsum_matmul(info, rows, last_op.value(), op, common_dims, left, right);
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

#if DEBUG == 1
            std::cout << "REDUCTION 2:" << std::endl;
            for(auto r : red)
            {
                std::cout << r << std::endl;
            }
#endif

            if(red.size())
            {
#ifdef GRAPH
                std::cout << "reduce_sum" << std::endl;
                std::cout << "> input shape: " << op->get_shape() << std::endl;
                std::cout << "> axes: ";
                for(auto _ : red)
                {
                    std::cout << _ << " ";
                }
                std::cout << std::endl;
#endif
                op = info.add_instruction(make_op("reduce_sum", {{"axes", red}}), op);
                // compute output row
                for(int r : red)
                {
                    rows[1][r] = -1;
                }
            }

            op = apply_squeeze_transpose(info, rows, op, mat[args.size()]);
        }

        return op;
    }

    void basic_verification(std::vector<int> lengths,
                            const std::vector<instruction_ref>& args,
                            std::string /*equation*/) const
    {
        if(lengths.size() - 1 != args.size())
        {
            MIGRAPHX_THROW("Equation doesn't match with number of provided inputs");
        }

        int i = 0;
        for(instruction_ref arg : args)
        {
            if(lengths[i++] != arg->get_shape().ndim())
            {
                MIGRAPHX_THROW("Dimensions of provided input don't match with equation");
            }
        }
    }

    std::tuple<std::set<char>, std::vector<std::vector<int>>, std::vector<int>>
    analyse_einsum_equation(std::string equation) const
    {
        std::vector<std::string> spl = split(trim(equation), "->");
        if(spl.size() != 2 or spl[1].size() == 0 or spl[0].size() == 0)
        {
            MIGRAPHX_THROW("The equation has to have two sides"); // TODO can have only left side
        }

        std::vector<std::string> inputs;
        for(std::string s : split(spl[0], ","))
        {
            inputs.push_back(trim(s));
        }
        std::string output = trim(spl[1]);

#if DEBUG == 1
        std::cout << "INPUTS:" << std::endl;
        for(std::string inp : inputs)
        {
            std::cout << inp << std::endl;
        }
        std::cout << "OUTPUT:" << std::endl;
        std::cout << output << std::endl;
#endif

        std::set<char> letters;
        for(std::string inp : inputs)
        {
            letters.merge(std::set<char>(inp.begin(), inp.end()));
        }

#if DEBUG == 1
        std::cout << "LETTERS:" << std::endl;
        for(char c : letters)
        {
            std::cout << c << std::endl;
        }
#endif

        if(!std::all_of(letters.begin(), letters.end(), [](char c) {
               return ('a' <= c and c <= 'z') or ('A' <= c and c <= 'Z');
           }))
        {
            MIGRAPHX_THROW("Equation must only contain letters"); // TODO ellipsis
        }

        std::map<char, int> rev;

        int i = 0;
        for(char c : letters)
        {
            rev[c] = i++;
        }

        for(char c : output)
        {
            if(!letters.count(c))
            {
                MIGRAPHX_THROW("Output contains unexpected letter");
            }
        }

        std::vector<std::vector<int>> mat = full(inputs.size() + 1, letters.size(), -1);

        i = 0;
        for(std::string inp : inputs)
        {
            int k = 0;
            for(char c : inp)
            {
                mat[i][rev[c]] = k++;
            }
            i += 1;
        }

        int k = 0;
        for(char c : output)
        {
            mat[inputs.size()][rev[c]] = k++;
        }

#if DEBUG == 1
        std::cout << "MATRIX:" << std::endl;
        for(int i = 0; i < mat.size(); ++i)
        {
            for(int j = 0; j < mat[0].size(); ++j)
            {
                std::cout << mat[i][j] << " ";
            }
            std::cout << std::endl;
        }
#endif

        std::vector<int> lengths;
        for(std::string inp : inputs)
        {
            lengths.push_back(inp.size());
        }
        lengths.push_back(output.size());

#if DEBUG == 1
        std::cout << "LENGTHS:" << std::endl;
        for(int le : lengths)
        {
            std::cout << le << std::endl;
        }
#endif

        // TODO handle duplicates

        return {letters, mat, lengths};
    }

    instruction_ref apply_transpose_reshape(const onnx_parser::node_info& info,
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
#ifdef GRAPH
        std::cout << "unsqueeze" << std::endl;
        std::cout << "> input shape: " << op->get_shape() << std::endl;
        std::cout << "> axes: ";
        for(auto _ : s_axes)
        {
            std::cout << _ << " ";
        }
        std::cout << std::endl;
#endif
        op = info.add_instruction(make_op("unsqueeze", {{"axes", s_axes}}), op);
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

#if DEBUG == 1
        std::cout << "ROW:" << std::endl;
        for(auto r : row)
        {
            std::cout << r << std::endl;
        }
        std::cout << "PERMUTATION:" << std::endl;
        for(auto p : perm)
        {
            std::cout << std::get<1>(p) << std::endl;
        }
        std::cout << "NEW PERMUTATION:" << std::endl;
        for(auto np : new_perm)
        {
            std::cout << np << std::endl;
        }
#endif

        if(not is_transpose_identity(new_perm))
        {
#ifdef GRAPH
            std::cout << "transpose" << std::endl;
            std::cout << "> input shape: " << op->get_shape() << std::endl;
            std::cout << "> permutation: ";
            for(auto _ : new_perm)
            {
                std::cout << _ << " ";
            }
            std::cout << std::endl;
#endif
            op = info.add_instruction(make_op("transpose", {{"permutation", new_perm}}), op);
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

    instruction_ref apply_squeeze_transpose(const onnx_parser::node_info& info,
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

#if DEBUG == 1
        std::cout << "NEW PERMUTATION 2:" << std::endl;
        for(auto np : new_perm)
        {
            std::cout << np << std::endl;
        }
#endif

        if(not is_transpose_identity(new_perm))
        {
#ifdef GRAPH
            std::cout << "transpose" << std::endl;
            std::cout << "> input shape: " << op->get_shape() << std::endl;
            std::cout << "> permutation: ";
            for(auto _ : new_perm)
            {
                std::cout << _ << " ";
            }
            std::cout << std::endl;
#endif
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
#ifdef GRAPH
            std::cout << "squeeze" << std::endl;
            std::cout << "> input shape: " << op->get_shape() << std::endl;
            std::cout << "> axes: ";
            for(auto _ : sq)
            {
                std::cout << _ << " ";
            }
            std::cout << std::endl;
#endif
            op = info.add_instruction(make_op("squeeze", {{"axes", sq}}), op);
            // compute output row
            for(int a : sq)
            {
                rows[1][a] = -1;
            }
        }

        return op;
    }

    instruction_ref apply_einsum_matmul(const onnx_parser::node_info& info,
                                        std::vector<std::vector<int>>& rows,
                                        instruction_ref op1,
                                        instruction_ref op2,
                                        std::vector<int> axes,
                                        std::vector<int> left,
                                        std::vector<int> right) const
    {
#if DEBUG == 2
        std::cout << "ROWS:" << std::endl;
        for(int i = 0; i < rows.size(); ++i)
        {
            for(int j = 0; j < rows[0].size(); ++j)
            {
                std::cout << rows[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "AXES:" << std::endl;
        for(int a : axes)
        {
            std::cout << a << std::endl;
        }
        std::cout << "LEFT:" << std::endl;
        for(int l : left)
        {
            std::cout << l << std::endl;
        }
        std::cout << "RIGHT:" << std::endl;
        for(int r : right)
        {
            std::cout << r << std::endl;
        }
#endif

        int ndim = rows[0].size();

        //         if(axes.size() == 0 and set_intersection(left, right).size() == 0)
        //         {
        // #if DEBUG == 2
        //             std::cout << "MUL" << std::endl;
        //             std::cout << op1->get_shape() << std::endl;
        //             std::cout << op2->get_shape() << std::endl;
        // #endif
        //             instruction_ref op = info.add_instruction(make_op("mul"), op1, op2);
        //             // compute output row
        //             std::transform(rows[0].begin(),
        //                            rows[0].end(),
        //                            rows[1].begin(),
        //                            rows[1].begin(),
        //                            std::greater<int>{});
        //             return op;
        //         }

        if(not(set_intersection(axes, left).size() == 0 and
               set_intersection(axes, right).size() == 0))
        {
            MIGRAPHX_THROW("Not implemented");
        }

        if(set_intersection(axes, left).size() == 0 and set_intersection(axes, right).size() == 0)
        {
            std::vector<int> all_axes = set_union(set_union(left, right), axes);

            std::vector<int> common_axes = set_intersection(left, right);
            for(int i = 0; i < ndim; ++i)
            {
                if(std::find(all_axes.begin(), all_axes.end(), i) == all_axes.end())
                {
                    common_axes.push_back(i);
                }
            }
            std::sort(common_axes.begin(), common_axes.end());

#if DEBUG == 2
            std::cout << "ALL AXES:" << std::endl;
            for(int a : all_axes)
            {
                std::cout << a << std::endl;
            }
            std::cout << "COMMON AXES:" << std::endl;
            for(int a : common_axes)
            {
                std::cout << a << std::endl;
            }
#endif

            // ReduceSum
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

#if DEBUG == 2
            std::cout << "HAS DIM:" << std::endl;
            for(int a : has_dim)
            {
                std::cout << a << std::endl;
            }
            std::cout << "RIGHT NO LEFT:" << std::endl;
            for(int a : right_no_left)
            {
                std::cout << a << std::endl;
            }
#endif

            if(right_no_left.size())
            {
                std::sort(right_no_left.begin(), right_no_left.end());
#ifdef GRAPH
                std::cout << "reduce_sum" << std::endl;
                std::cout << "> input1 shape: " << op1->get_shape() << std::endl;
                std::cout << "> axes: ";
                for(auto _ : right_no_left)
                {
                    std::cout << _ << " ";
                }
                std::cout << std::endl;
#endif
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

#if DEBUG == 2
            std::cout << "HAS DIM:" << std::endl;
            for(int a : has_dim)
            {
                std::cout << a << std::endl;
            }
            std::cout << "LEFT NO RIGHT:" << std::endl;
            for(int a : left_no_right)
            {
                std::cout << a << std::endl;
            }
#endif

            if(left_no_right.size())
            {
                std::sort(left_no_right.begin(), left_no_right.end());
#ifdef GRAPH
                std::cout << "reduce_sum" << std::endl;
                std::cout << "> input2 shape: " << op2->get_shape() << std::endl;
                std::cout << "> axes: ";
                for(auto _ : left_no_right)
                {
                    std::cout << _ << " ";
                }
                std::cout << std::endl;
#endif
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
                if(std::find(common_axes.begin(), common_axes.end(), i) != common_axes.end())
                {
                    first = -1;
                }
                else if(std::find(axes.begin(), axes.end(), i) != axes.end())
                {
                    first = 1;
                }
                else
                {
                    first = 0;
                }
                i_axes.push_back({first, i});
            }

            std::sort(i_axes.begin(), i_axes.end(), [](auto lhs, auto rhs) {
                return std::get<0>(lhs) < std::get<0>(rhs);
            });

            std::vector<int> perm;
            for(auto _ : i_axes)
            {
                perm.push_back(std::get<1>(_));
            }

            std::vector<int> perm_left;
            for(int i = 0; i < perm.size(); ++i)
            {
                if(std::find(left.begin(), left.end(), perm[i]) != left.end())
                {
                    perm_left.push_back(i);
                }
            }

            std::vector<int> perm_right;
            for(int i = 0; i < perm.size(); ++i)
            {
                if(std::find(right.begin(), right.end(), perm[i]) != right.end())
                {
                    perm_right.push_back(i);
                }
            }

#if DEBUG == 2
            std::cout << "I_AXES:" << std::endl;
            for(auto _ : i_axes)
            {
                std::cout << std::get<0>(_) << " " << std::get<1>(_) << std::endl;
            }
            std::cout << "PERM:" << std::endl;
            for(int p : perm)
            {
                std::cout << p << std::endl;
            }
            std::cout << "PERM LEFT:" << std::endl;
            for(int p : perm_left)
            {
                std::cout << p << std::endl;
            }
            std::cout << "PERM RIGHT:" << std::endl;
            for(int p : perm_right)
            {
                std::cout << p << std::endl;
            }
#endif

            if(!is_transpose_identity(perm))
            {
#ifdef GRAPH
                std::cout << "transpose" << std::endl;
                std::cout << "> input1 shape: " << op1->get_shape() << std::endl;
                std::cout << "> permutation: ";
                for(auto _ : perm)
                {
                    std::cout << _ << " ";
                }
                std::cout << std::endl;
#endif
                op1 = info.add_instruction(make_op("transpose", {{"permutation", perm}}), op1);
                // compute output row
                auto cpy = rows[0];
                int i    = 0;
                for(int p : perm)
                {
                    rows[0][i++] = cpy[p];
                }
#ifdef GRAPH
                std::cout << "transpose" << std::endl;
                std::cout << "> input2 shape: " << op2->get_shape() << std::endl;
                std::cout << "> permutation: ";
                for(auto _ : perm)
                {
                    std::cout << _ << " ";
                }
                std::cout << std::endl;
#endif
                op2 = info.add_instruction(make_op("transpose", {{"permutation", perm}}), op2);
                // compute output row
                cpy = rows[1];
                i   = 0;
                for(int p : perm)
                {
                    rows[1][i++] = cpy[p];
                }
            }

            // Reshape
            std::vector<int> all_axes2(ndim);
            std::iota(all_axes2.begin(), all_axes2.end(), 0);

            std::vector<int> new_axes;
            if(axes.size() > 0)
            {
                std::copy(
                    all_axes2.end() - axes.size(), all_axes2.end(), std::back_inserter(new_axes));
            }

            std::vector<int> new_common_axes;
            std::copy(all_axes2.begin(),
                      all_axes2.begin() + common_axes.size(),
                      std::back_inserter(new_common_axes));

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

#if DEBUG == 2
            std::cout << "ALL AXES 2:" << std::endl;
            for(int a : all_axes2)
            {
                std::cout << a << std::endl;
            }
            std::cout << "NEW AXES:" << std::endl;
            for(int a : new_axes)
            {
                std::cout << a << std::endl;
            }
            std::cout << "NEW COMMON AXES:" << std::endl;
            for(int a : new_common_axes)
            {
                std::cout << a << std::endl;
            }
            std::cout << "NOT IN BOTH:" << std::endl;
            for(int a : not_in_both)
            {
                std::cout << a << std::endl;
            }
#endif

            instruction_ref op = apply_batch_dot(
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
#ifdef GRAPH
                std::cout << "transpose" << std::endl;
                std::cout << "> input1 shape: " << op1->get_shape() << std::endl;
                std::cout << "> permutation: ";
                for(auto _ : perm)
                {
                    std::cout << _ << " ";
                }
                std::cout << std::endl;
#endif
                op1 = info.add_instruction(make_op("transpose", {{"permutation", perm}}), op1);
                // compute output row
                auto cpy = rows[0];
                int i    = 0;
                for(int p : perm)
                {
                    rows[0][i++] = cpy[p];
                }
#ifdef GRAPH
                std::cout << "transpose" << std::endl;
                std::cout << "> input shape: " << op->get_shape() << std::endl;
                std::cout << "> permutation: ";
                for(auto _ : perm)
                {
                    std::cout << _ << " ";
                }
                std::cout << std::endl;
#endif
                op = info.add_instruction(make_op("transpose", {{"permutation", perm}}), op);
                // compute output row
                cpy = rows[1];
                i   = 0;
                for(int p : perm)
                {
                    rows[1][i++] = cpy[p];
                }
            }

            return op;
        }

        MIGRAPHX_THROW("axes and right or left have axes in common");
    }

    instruction_ref apply_batch_dot(const onnx_parser::node_info& info,
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

        std::vector<std::size_t> op1_shape = op1->get_shape().lens();
        std::vector<std::size_t> op2_shape = op2->get_shape().lens();

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

#if DEBUG == 3
        std::cout << "dim0: " << dim0 << std::endl;
        std::cout << "dim0b: " << dim0b << std::endl;
        std::cout << "dimb: " << dimb << std::endl;
        std::cout << "dim1: " << dim1 << std::endl;
        std::cout << "dim2: " << dim2 << std::endl;
#endif

        std::string dot_kind = get_dot_kind(rows, batch_axes);
#if DEBUG == 3
        std::cout << "DOT KIND: " << dot_kind << std::endl;
        std::cout << "op1_shape" << op1->get_shape() << std::endl;
        std::cout << "op2_shape" << op2->get_shape() << std::endl;
#endif

#ifdef GRAPH
        std::cout << "reshape" << std::endl;
        std::cout << "> input1 shape: " << op1->get_shape() << std::endl;
        std::cout << "> dims: ";
        for(auto _ : std::vector<int>{dim0, dimb, dim1})
        {
            std::cout << _ << " ";
        }
        std::cout << std::endl;
#endif
        instruction_ref op1sh =
            info.add_instruction(make_op("reshape", {{"dims", {dim0, dimb, dim1}}}), op1);
#ifdef GRAPH
        std::cout << "reshape" << std::endl;
        std::cout << "> input2 shape: " << op2->get_shape() << std::endl;
        std::cout << "> dims: ";
        for(auto _ : std::vector<int>{dim0b, dimb, dim2})
        {
            std::cout << _ << " ";
        }
        std::cout << std::endl;
#endif
        instruction_ref op2sh =
            info.add_instruction(make_op("reshape", {{"dims", {dim0b, dimb, dim2}}}), op2);

        instruction_ref dot;
        // if(dot_kind == "11" or dot_kind == "N1" or dot_kind == "1N")
        // {
        //     op1sh = info.add_instruction(
        //         make_op("reshape", {{"dims", {-1, op1sh->get_shape().lens().back()}}}), op1sh);
        //     op2sh = info.add_instruction(
        //         make_op("reshape", {{"dims", {-1, op2sh->get_shape().lens().back()}}}), op2sh);

        //     std::vector<std::size_t> all_dims = op1sh->get_shape().lens();
        //     std::copy(op2sh->get_shape().lens().begin(),
        //               op2sh->get_shape().lens().end(),
        //               std::back_inserter(all_dims));
        //     bool square = std::min_element(all_dims.begin(), all_dims.end()) ==
        //                   std::max_element(all_dims.begin(), all_dims.end());

        //     if(not square)
        //     {
        //         dot = info.add_instruction(
        //             make_op("dot"),
        //             op1sh,
        //             info.add_instruction(make_op("transpose", {{"permutation", {1, 0}}}),
        //             op2sh));
        //     }
        //     else
        //     {
        //         dot = info.add_instruction(
        //             make_op("dot"),
        //             info.add_instruction(make_op("transpose", {{"permutation", {1, 0}}}), op2sh),
        //             op1sh);
        //     }
        // }
        // else
        {
#ifdef GRAPH
            std::cout << "transpose" << std::endl;
            std::cout << "> input2 shape: " << op2sh->get_shape() << std::endl;
            std::cout << "> permutation: 0, 2, 1 " << std::endl;
#endif
            op2sh = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1}}}), op2sh);
#ifdef GRAPH
            std::cout << "dot" << std::endl;
            std::cout << "> input1 shape: " << op1sh->get_shape() << std::endl;
            std::cout << "> input2 shape: " << op2sh->get_shape() << std::endl;
#endif
            dot = info.add_instruction(make_op("dot"), op1sh, op2sh);
        }

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
#ifdef GRAPH
        std::cout << "reshape" << std::endl;
        std::cout << "> input shape: " << dot->get_shape() << std::endl;
        std::cout << "> dims: ";
        for(auto _ : new_shape)
        {
            std::cout << _ << " ";
        }
        std::cout << std::endl;
#endif
        instruction_ref op = info.add_instruction(make_op("reshape", {{"dims", new_shape}}), dot);
        // compute output row
        std::transform(
            rows[0].begin(), rows[0].end(), rows[1].begin(), rows[1].begin(), std::greater<int>{});
        for(int a : sum_axes)
        {
            if(std::find(right.begin(), right.end(), a) == right.end())
            {
                rows[1][a] = -1;
            }
        }

        return op;
    }

    std::string get_dot_kind(const std::vector<std::vector<int>>& rows,
                             std::vector<int> batch_axes) const
    {
#if DEBUG == 3
        std::cout << "GET DOT KIND ROWS:" << std::endl;
        for(int i = 0; i < rows.size(); ++i)
        {
            for(int j = 0; j < rows[0].size(); ++j)
            {
                std::cout << rows[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "BATCH AXES:" << std::endl;
        for(int k : batch_axes)
        {
            std::cout << k << std::endl;
        }
#endif

        std::vector<int> batch_left, batch_right;
        for(int k : batch_axes)
        {
            batch_left.push_back(rows[0][k]);
            batch_right.push_back(rows[1][k]);
        }

        bool n_left = batch_left.size() >
                      0 /*and *std::max_element(batch_left.begin(), batch_left.end()) == 2*/;
        bool n_right =
            batch_right.size() > 0 /*and
                                    *std::max_element(batch_right.begin(), batch_right.end()) == 2*/
            ;

        std::string ret;
        return ret + (n_left ? "N" : "1") + (n_right ? "N" : "1");
    }

    bool is_transpose_identity(std::vector<int> perm) const
    {
        std::vector<int> range(perm.size());
        std::iota(range.begin(), range.end(), 0);
        return perm == range;
    }

    std::string ltrim(std::string s) const
    {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }));
        return s;
    }

    std::string rtrim(std::string s) const
    {
        s.erase(
            std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); })
                .base(),
            s.end());
        return s;
    }

    std::string trim(std::string s) const { return ltrim(rtrim(s)); }

    std::vector<std::string> split(std::string str, std::string delim) const
    {
        std::vector<std::string> ret;
        std::size_t prev = 0u, cur = 0u;
        while((cur = str.find(delim, prev)) != std::string::npos)
        {
            ret.emplace_back(str.substr(prev, cur - prev));
            prev = cur + delim.size();
        }
        ret.emplace_back(str.substr(prev, std::string::npos));

        return ret;
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

    // NEW PARSING

    std::tuple<string_vec, std::string>
    analyze_equation(std::string_view equation, const std::vector<instruction_ref>& args) const
    {
        std::tuple<string_vec, std::string> ret;
        auto& [terms, unique_labels] = ret;

        auto [input_terms, output_term, label_count, explicit_form] = parse_equation(equation);

        validate_input_terms(input_terms, args);
        if(not output_term.empty())
            validate_output_term(output_term, label_count);
        else if(not explicit_form)
            output_term = generate_output_term(label_count);

#if DEBUG
        std::cout << migraphx::to_string_range(input_terms) << std::endl;
        std::cout << output_term << std::endl;
        std::cout << "{" << std::endl;
        for(auto [k, v] : label_count)
            std::cout << "  " << k << ": " << v << std::endl;
        std::cout << "}" << std::endl;

#endif

        terms = std::move(input_terms);
        terms.emplace_back(std::move(output_term));
        for(auto [l, _] : label_count)
            unique_labels += l;

        return ret;
    }

    std::vector<std::vector<int>> make_mapping_matrix(const string_vec& terms,
                                                      std::string_view unique_labels) const
    {
        std::map<char, int> label_to_column;
        for(auto i = 0; i < unique_labels.size(); ++i)
            label_to_column[unique_labels[i]] = i;

        std::vector<std::vector<int>> mat = full(terms.size(), unique_labels.size(), -1);

        for(auto i = 0; i < terms.size(); ++i)
        {
            const auto& term = terms[i];
            for(auto j = 0; j < term.size(); ++j)
                mat[i][label_to_column[term[j]]] = j;
        }

#if DEBUG == 1
        std::cout << "MATRIX:" << std::endl;
        for(int i = 0; i < mat.size(); ++i)
        {
            for(int j = 0; j < mat[0].size(); ++j)
            {
                std::cout << mat[i][j] << " ";
            }
            std::cout << std::endl;
        }
#endif

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

    std::string generate_output_term(const char_int_map& label_count) const
    {
        std::string output_term;
        for(const auto [label, count] : label_count)
            if(count == 1)
                output_term += label;

        return output_term;
    }

    void validate_output_term(std::string_view output_term, const char_int_map& label_count) const
    {
        for(const auto label : output_term)
            if(not contains(label_count, label))
                MIGRAPHX_THROW("Output term contains label " + std::to_string(label) +
                               ", which is not present in any of the input terms");
    }

    void validate_input_terms(const string_vec& input_terms,
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
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
