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

#define DEBUG 2

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_einsum : op_parser<parse_einsum>
{
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

        auto [letters, mat, lengths] = analyse_einsum_equation(equation);

        std::tuple<int, int> mat_shape = {mat.size(), mat[0].size()};
        int full_dim                   = std::get<1>(mat_shape);

        if(letters.size() != full_dim)
        {
            MIGRAPHX_THROW("Unexpected number of letters");
        }

        basic_verification(lengths, args, equation);

        std::vector<std::vector<int>> rows = full(2, full_dim, -1);

        int i = 0;
        for(const auto& arg : args)
        {
            op      = info.add_instruction(make_op("identity"), arg);
            rows[1] = mat[i]; // compute output row

            op = apply_transpose_reshape(info, rows, op, mat[i]);

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
        for(const auto& arg : args)
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

        if(axes.size() == 0 and set_intersection(left, right).size() == 0)
        {
            instruction_ref op = info.add_instruction(make_op("mul"), op1, op2);
            // compute output row
            std::transform(rows[0].begin(),
                           rows[0].end(),
                           rows[1].begin(),
                           rows[1].begin(),
                           std::greater<int>{});
            return op;
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
                op1 = info.add_instruction(make_op("reduce_sum", {{"axes", right_no_left}}), op1);
                // compute output row
                // TODO
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
                op2 = info.add_instruction(make_op("reduce_sum", {{"axes", left_no_right}}), op2);
                // compute output row
                // TODO
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
                op1 = info.add_instruction(make_op("transpose", {{"permutation", perm}}), op1);
                // compute output row
                // TODO

                op2 = info.add_instruction(make_op("transpose", {{"permutation", perm}}), op2);
                // compute output row
                // TODO
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

            instruction_ref op;
            op = info.add_instruction(make_op("dot"), op1, op2);
            // compute output row


            return op1;
        }

        MIGRAPHX_THROW("axes and right or left have axes in common");
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
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
