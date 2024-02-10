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
        instruction_ref op;
        std::optional<instruction_ref> last_op;

        if(not contains(info.attributes, "equation"))
            MIGRAPHX_THROW("Equation attribute is required");

        std::string equation = info.attributes.at("equation").s();
        std::cout << "EQUATION: " << equation << std::endl;

        auto [terms, unique_labels]    = analyze_equation(equation, args);
        auto mat                       = make_mapping_matrix(terms, unique_labels);
        std::tuple<int, int> mat_shape = {mat.size(), mat[0].size()};

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

    std::tuple<string_vec, std::string>
    analyze_equation(std::string_view equation, const std::vector<instruction_ref>& args) const
    {
        std::tuple<string_vec, std::string> ret;
        auto& [terms, unique_labels] = ret;

        auto [input_terms, output_term, label_count] = parse_equation(equation);

        validate_input_terms(input_terms, args);
        if(not output_term.empty())
            validate_output_term(output_term, label_count);
        else
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

    std::tuple<std::vector<std::string>, std::string, std::map<char, int>>
    parse_equation(std::string_view equation) const
    {
        std::tuple<std::vector<std::string>, std::string, std::map<char, int>> ret;
        auto& [input_terms, output_term, label_count] = ret;

        std::string term;
        bool has_ellipsis  = false;
        bool explicit_form = false;

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
