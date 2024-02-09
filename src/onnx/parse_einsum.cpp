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

#if DEBUG == 1
        std::cout << "EQUATION: " std::endl;
        std::cout << equation << std::endl;
#endif

        auto [letters, mat, lengths] = analyse_einsum_equation(equation);

        std::tuple<int, int> mat_shape = {mat.size(), mat[0].size()};

        if(letters.size() != std::get<1>(mat_shape))
        {
            MIGRAPHX_THROW("Unexpected number of letters");
        }

        basic_verification(lengths, args, equation);

        std::vector<std::vector<int>> rows = full(2, std::get<1>(mat_shape), -1);
        int fd                             = std::get<1>(mat_shape);

        int i = 0;
        for(const auto& arg : args)
        {
            op      = info.add_instruction(make_op("identity"), arg);
            rows[1] = mat[i]; // compute output row

            op = apply_transpose_reshape(info, rows, op, mat[i]);

            // reduction
            std::vector<int> red;
            for(int d = 0; d < std::get<1>(mat_shape); ++d)
            {
                bool used_later = false;
                for(int l = i + 1; l < mat.size(); ++l)
                {
                    if(mat[l][d] != -1)
                    {
                        used_later = true;
                        break;
                    }
                }

                if(not used_later and rows[1][d] != -1 and rows[0][d] == -1)
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
                // TODO op = matmul(last_op, op)
            }

            last_op = op;
            rows[0] = rows[1];

            i += 1;
        }

        // final
        if(*(std::max_element(mat[args.size()].begin(), mat[args.size()].end())) >= 0)
        {
            rows[1] = mat[args.size()];

            std::vector<int> red;

            for(int d = 0; d < fd; ++d)
            {
                if(rows[0][d] > 0 && rows[1][d] == -1)
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
                for(auto r : red)
                {
                    rows[1][r] = -1;
                }
            }

            op = apply_squeeze_transpose(info, op, rows, mat[args.size()]);
        }

        return op;
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

    instruction_ref apply_transpose_reshape(const onnx_parser::node_info& info,
                                            std::vector<std::vector<int>>& rows,
                                            instruction_ref& op,
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

    bool is_transpose_identity(std::vector<int> perm) const
    {
        std::vector<int> range(perm.size());
        std::iota(range.begin(), range.end(), 0);
        return perm == range;
    }

    instruction_ref apply_squeeze_transpose(const onnx_parser::node_info& info,
                                            instruction_ref& op,
                                            std::vector<std::vector<int>>& rows,
                                            std::vector<int> row_output) const
    {
        std::vector<std::tuple<int, int>> perm;
        std::vector<int> sq;

        int i = 0;
        for(auto d : row_output)
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
        for(auto d : row_output)
        {
            if(d != -1)
            {
                new_perm[i] = std::get<1>(perm[p]);
                p += 1;
            }
            i += 1;
        }

#if DEBUG == 1
        std::cout << "NEW PERMUTATION 2:" << std::endl;
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

        if(sq.size())
        {
            op = info.add_instruction(make_op("squeeze", {{"axes", sq}}), op);

            // compute output row
            for(auto a : sq)
            {
                rows[1][a] = -1;
            }
        }

        return op;
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
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
