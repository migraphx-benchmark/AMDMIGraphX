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

#define DEBUG 0

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_einsum : op_parser<parse_einsum>
{
    std::vector<op_desc> operators() const { return {{"Einsum"}}; }

    instruction_ref parse(const op_desc& /* opd */,
                          const onnx_parser& /* parser */,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        return decompose_equation(info, args);
    }

    instruction_ref decompose_equation(const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        instruction_ref op;

        const auto [letters, mat, lengths] =
            analyse_einsum_equation(info.attributes.at("equation").s());

        // TODO verify against input shapes

        int fd                             = mat.at(0).size();
        std::vector<std::vector<int>> rows = full(2, fd, -1);

        int i = 0;
        for(const auto& arg : args)
        {
            op = info.add_instruction(make_op("identity"), arg);

            // compute output row
            for(int j = 0; j < fd; ++j)
            {
                rows[1][j] = mat[i][j];
            }

            op = apply_transpose_reshape(info, op, rows, mat[i]);

            // Reduction
            std::vector<int> red;
            for(int d = 0; d < fd; ++d)
            {
                bool found = false;
                for(int l = i + 1; l < mat.size(); ++l)
                {
                    if(mat[l][d] > -1)
                    {
                        found = true;
                        break;
                    }
                }

                if(!found && rows[1][d] != -1 && rows[0][d] == -1)
                {
                    red.push_back(d);
                }
            }

            if(red.size())
            {
                op = info.add_instruction(make_op("reduce_sum", {{"axes", red}}), op);

                // compute output row
                for(auto r : red)
                {
                    rows[1][r] = -1;
                }
            }

            // TODO matmul

            rows[0] = rows[1];

            i += 1;
        }

        // Final
        

        return op;
    }

    instruction_ref apply_transpose_reshape(const onnx_parser::node_info& info,
                                            instruction_ref& op,
                                            std::vector<std::vector<int>>& rows,
                                            std::vector<int> row) const
    {
        std::vector<std::tuple<int, int>> axes;
        int p = 0;
        std::vector<std::tuple<int, int>> perm;

        int i = 0;
        for(auto r : row)
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

        std::sort(perm.begin(), perm.end(), [](auto lhs, auto rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });
        p = 0;

        std::vector<int> new_perm(row.size());
        std::iota(new_perm.begin(), new_perm.end(), 0);

        i = 0;
        for(auto r : row)
        {
            if(r != -1)
            {
                new_perm[std::get<1>(perm[p])] = i;
                p += 1;
            }
            i += 1;
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

        // TODO check for identity
        op = info.add_instruction(make_op("transpose", {{"permutation", new_perm}}), op);

        // compute output row
        auto cpy = rows[1];
        i        = 0;
        for(auto np : new_perm)
        {
            rows[1][i++] = cpy[np];
        }

        return op;
    }

    std::tuple<std::set<char>, std::vector<std::vector<int>>, std::vector<int>>
    analyse_einsum_equation(std::string equation) const
    {
#if DEBUG == 1
        std::cout << "EQUATION: " << equation << std::endl;
#endif

        auto spl = split(trim(equation), "->");
        if(spl.size() != 2 || spl[1].size() == 0 || spl[0].size() == 0)
        {
            MIGRAPHX_THROW("The equation has to have two sides"); // TODO can have only left side
        }

        std::vector<std::string> inputs;
        for(auto s : split(spl[0], ","))
        {
            inputs.push_back(trim(s));
        }

        std::string output = trim(spl[1]);

#if DEBUG == 1
        std::cout << "INPUTS:" << std::endl;
        for(auto input : inputs)
        {
            std::cout << input << std::endl;
        }

        std::cout << "OUTPUT:" << std::endl;
        std::cout << output << std::endl;
#endif

        std::set<char> letters;
        for(auto input : inputs)
        {
            for(auto c : input)
            {
                letters.insert(c);
            }
        }

#if DEBUG == 1
        std::cout << "LETTERS:" << std::endl;
        for(auto c : letters)
        {
            std::cout << c << std::endl;
        }
#endif

        for(auto c : letters)
        {
            if(!('a' <= c && c <= 'z') && !('A' <= c && c <= 'Z'))
            {
                MIGRAPHX_THROW("Equation must only contain letters"); // TODO ellipsis
            }
        }

        std::map<char, int> rev;

        int i = 0;
        for(auto c : letters)
        {
            rev[c] = i++;
        }

        for(auto c : output)
        {
            if(!letters.count(c))
            {
                MIGRAPHX_THROW("Output contains unexpected letter");
            }
        }

        size_t mat_rows = inputs.size() + 1;
        size_t mat_cols = letters.size();

#if DEBUG == 1
        std::cout << "MAT DIMS: " << mat_rows << " " << mat_cols << std::endl;
#endif

        std::vector<std::vector<int>> mat = full(mat_rows, mat_cols, -1);

        i = 0;
        for(auto input : inputs)
        {
            int k = 0;
            for(auto c : input)
            {
                mat[i][rev[c]] = k++;
            }
            i += 1;
        }

        int k = 0;
        for(auto c : output)
        {
            mat[inputs.size()][rev[c]] = k++;
        }

#if DEBUG == 1
        std::cout << "MATRIX:" << std::endl;
        for(int i = 0; i < mat_rows; ++i)
        {
            for(int j = 0; j < mat_cols; ++j)
            {
                std::cout << mat[i][j] << " ";
            }
            std::cout << std::endl;
        }
#endif

        std::vector<int> lengths;
        for(auto input : inputs)
        {
            lengths.push_back(input.size());
        }
        lengths.push_back(output.size());

#if DEBUG == 1
        std::cout << "LENGTHS:" << std::endl;
        for(auto l : lengths)
        {
            std::cout << l << std::endl;
        }
#endif

        // TODO handle duplicates

        return std::make_tuple(letters, mat, lengths);
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

    std::vector<std::string> split(const std::string& str, const std::string& delim) const
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

    std::vector<std::vector<int>> full(int rows, int cols, int fill) const
    {
        std::vector<std::vector<int>> ret(rows);
        for(auto& row : ret)
        {
            for(int i = 0; i < cols; ++i)
            {
                row.push_back(fill);
            }
        }

        return ret;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
