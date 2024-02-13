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

#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>
#include <onnx_verify_utils.hpp>

TEST_CASE(einsum_permute_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_permute_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x_data = {
        0.06727745, 0.21160052, 0.1340474, 0.74153227, 0.40337096, 0.81284493};

    migraphx::parameter_map pm;
    pm["x"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        0.06727745, 0.74153227, 0.21160052, 0.40337096, 0.1340474, 0.81284493};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

// TODO equation has to have two sides
// TEST_CASE(einsum_summation_test)
// {
//     migraphx::program p = migraphx::parse_onnx("einsum_summation_test.onnx");
//     p.compile(migraphx::make_target("ref"));

//     migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
//     std::vector<float> x_data = {
//         0.79413969, 0.45169144, 0.06846618, 0.67973967, 0.83375529, 0.44838823};

//     migraphx::parameter_map pm;
//     pm["x"] = migraphx::argument{x_shape, x_data.data()};

//     auto result = p.eval(pm).back();
//     std::vector<float> result_vector;
//     result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

//     std::vector<float> gold = {3.2761804984270566};
//     EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
// }

TEST_CASE(einsum_column_sum_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_column_sum_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x_data = {
        0.22235926, 0.83263138, 0.04747776, 0.96030827, 0.18947713, 0.48815767};

    migraphx::parameter_map pm;
    pm["x"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.18266753, 1.0221085, 0.53563543};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_row_sum_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_row_sum_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x_data = {
        0.17123185, 0.59008514, 0.37948294, 0.73022965, 0.22919172, 0.27532941};

    migraphx::parameter_map pm;
    pm["x"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.14079993, 1.23475077};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_matrix_vector_multiplication_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_matrix_vector_multiplication_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x_data = {
        0.4834133, 0.14106742, 0.50055824, 0.91764271, 0.95528452, 0.98199955};

    migraphx::shape v_shape{migraphx::shape::float_type, {1, 3}};
    std::vector<float> v_data = {0.73961958, 0.53071864, 0.34152803};

    migraphx::parameter_map pm;
    pm["x"] = migraphx::argument{x_shape, x_data.data()};
    pm["v"] = migraphx::argument{v_shape, v_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.60336371, 1.52107419};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_matrix_matrix_multiplication_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_matrix_matrix_multiplication_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x_data = {
        0.45176257, 0.84846429, 0.4374105, 0.25132236, 0.70519571, 0.4902031};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x_shape, x_data.data()};
    pm["x2"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.11530901, 0.92629139, 0.92629139, 0.80076299};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

// TODO equation has to have two sides
// TEST_CASE(einsum_vector_dot_product_test)
// {
//     migraphx::program p = migraphx::parse_onnx("einsum_vector_dot_product_test.onnx");
//     p.compile(migraphx::make_target("ref"));

//     migraphx::shape x_shape{migraphx::shape::float_type, {3}};
//     std::vector<float> x_data = {0.45263196, 0.90876706, 0.9584567};

//     migraphx::parameter_map pm;
//     pm["x1"] = migraphx::argument{x_shape, x_data.data()};
//     pm["x2"] = migraphx::argument{x_shape, x_data.data()};

//     auto result = p.eval(pm).back();
//     std::vector<float> result_vector;
//     result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

//     std::vector<float> gold = {1.94937252};
//     EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
// }

// TODO equation has to have two sides
// TEST_CASE(einsum_matrix_dot_product_test)
// {
//     migraphx::program p = migraphx::parse_onnx("einsum_matrix_dot_product_test.onnx");
//     p.compile(migraphx::make_target("ref"));

//     migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
//     std::vector<float> x_data = {
//         0.50001808, 0.12468059, 0.85439214, 0.00773521, 0.84764693, 0.87185525};

//     migraphx::parameter_map pm;
//     pm["x1"] = migraphx::argument{x_shape, x_data.data()};
//     pm["x2"] = migraphx::argument{x_shape, x_data.data()};

//     auto result = p.eval(pm).back();
//     std::vector<float> result_vector;
//     result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

//     std::vector<float> gold = {2.47424599};
//     EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
// }

TEST_CASE(einsum_hadamard_product_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_hadamard_product_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x_data = {
        0.86162928, 0.76609605, 0.03362172, 0.21778614, 0.27204858, 0.83778314};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x_shape, x_data.data()};
    pm["x2"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        0.74240502, 0.58690315, 0.00113042, 0.0474308, 0.07401043, 0.70188058};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_vector_outer_product_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_vector_outer_product_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {3}};
    std::vector<float> x1_data = {0.35935151, 0.51298139, 0.46076789};

    migraphx::shape x2_shape{migraphx::shape::float_type, {5}};
    std::vector<float> x2_data = {0.82417482, 0.17984153, 0.17680769, 0.55499376, 0.74447638};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.29616847,
                               0.06462632,
                               0.06353611,
                               0.19943785,
                               0.26752871,
                               0.42278634,
                               0.09225536,
                               0.09069905,
                               0.28470147,
                               0.38190252,
                               0.37975329,
                               0.0828652,
                               0.08146731,
                               0.2557233,
                               0.34303081};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_matrix_outer_product_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_matrix_outer_product_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {2, 3}};
    std::vector<float> x1_data = {
        0.25870501, 0.06755926, 0.18247427, 0.19436556, 0.61580192, 0.20010939};

    migraphx::shape x2_shape{migraphx::shape::float_type, {2, 5}};
    std::vector<float> x2_data = {0.30771264,
                                  0.86270274,
                                  0.55251869,
                                  0.35880608,
                                  0.3234085,
                                  0.24642323,
                                  0.82411907,
                                  0.33488431,
                                  0.69288027,
                                  0.21717812};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        0.0796068,  0.22318552, 0.14293935, 0.09282493, 0.0836674,  0.06375092, 0.21320373,
        0.08663625, 0.17925159, 0.05618507, 0.02078884, 0.05828356, 0.03732775, 0.02424067,
        0.02184924, 0.01664817, 0.05567687, 0.02262453, 0.04681048, 0.01467239, 0.05614964,
        0.15742105, 0.10082044, 0.06547288, 0.05901373, 0.0449659,  0.15038052, 0.06110777,
        0.12643282, 0.03962942, 0.05980874, 0.1676797,  0.1073906,  0.06973954, 0.06285947,
        0.04789619, 0.16018036, 0.06508997, 0.13467206, 0.04221195, 0.18949004, 0.53125401,
        0.34024207, 0.22095347, 0.19915557, 0.1517479,  0.50749411, 0.2062224,  0.426677,
        0.1337387,  0.06157619, 0.17263492, 0.11056418, 0.07180047, 0.06471708, 0.0493116,
        0.16491396, 0.06701349, 0.13865185, 0.04345938};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_batch_matrix_multiplication_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_batch_matrix_multiplication_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {3, 2, 5}};
    std::vector<float> x1_data = {0.99236023, 0.6848901,  0.37916487, 0.35448254, 0.06103943,
                                  0.88991707, 0.20816843, 0.12124124, 0.90632983, 0.88490338,
                                  0.93530363, 0.41393917, 0.95269137, 0.95556378, 0.63113954,
                                  0.87936215, 0.66831395, 0.38079353, 0.74128241, 0.05493966,
                                  0.12545692, 0.77418839, 0.17562823, 0.5558762,  0.95698858,
                                  0.49207445, 0.81934147, 0.50168285, 0.13782384, 0.71351839};

    migraphx::shape x2_shape{migraphx::shape::float_type, {3, 5, 3}};
    std::vector<float> x2_data = {
        0.72870257, 0.44635711, 0.05938103, 0.7031737,  0.52116502, 0.01719079, 0.99837568,
        0.29989025, 0.63673246, 0.39255282, 0.39796917, 0.03082538, 0.20994321, 0.11431396,
        0.06561894, 0.99749458, 0.45970296, 0.76957234, 0.98073012, 0.63154904, 0.22862209,
        0.71098086, 0.68895963, 0.92763041, 0.61730666, 0.54453456, 0.99719059, 0.05984043,
        0.64232788, 0.9754334,  0.39450223, 0.1005812,  0.11753032, 0.59885466, 0.75932222,
        0.45269589, 0.26201765, 0.39022748, 0.96507247, 0.55260731, 0.42233854, 0.50671452,
        0.60313192, 0.32628192, 0.40066181};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.73524908,
                               1.06164644,
                               0.32706016,
                               1.45746952,
                               1.00391812,
                               0.21962538,
                               2.64391179,
                               2.27348666,
                               3.26667873,
                               2.26421769,
                               1.52761296,
                               1.97554961,
                               1.44350867,
                               1.21602803,
                               1.19981019,
                               1.32274886,
                               1.15842452,
                               1.2686234};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_matrix_diagonal_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_matrix_diagonal_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {3, 3}};
    std::vector<float> x_data = {0.47776573,
                                 0.63448645,
                                 0.89651875,
                                 0.23679368,
                                 0.99918665,
                                 0.27613904,
                                 0.57251725,
                                 0.30676534,
                                 0.01097199};

    migraphx::parameter_map pm;
    pm["x"] = migraphx::argument{x_shape, x_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.47776573, 0.99918665, 0.01097199};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

// TODO equation has to have two sides
// TEST_CASE(einsum_matrix_trace_test)
// {
//     migraphx::program p = migraphx::parse_onnx("einsum_matrix_trace_test.onnx");
//     p.compile(migraphx::make_target("ref"));

//     migraphx::shape x_shape{migraphx::shape::float_type, {3, 3}};
//     std::vector<float> x_data = {0.90812557,
//                                  0.40719192,
//                                  0.71678312,
//                                  0.78176503,
//                                  0.57731702,
//                                  0.23585615,
//                                  0.06292936,
//                                  0.46016886,
//                                  0.37753559};

//     migraphx::parameter_map pm;
//     pm["x"] = migraphx::argument{x_shape, x_data.data()};

//     auto result = p.eval(pm).back();
//     std::vector<float> result_vector;
//     result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

//     std::vector<float> gold = {1.86297818};
//     EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
// }

// TODO equation has to have two sides
// TEST_CASE(einsum_2d_3d_multiplication_test)
// {
//     migraphx::program p = migraphx::parse_onnx("einsum_2d_3d_multiplication_test.onnx");
//     p.compile(migraphx::make_target("ref"));

//     migraphx::shape x1_shape{migraphx::shape::float_type, {3, 3}};
//     std::vector<float> x1_data = {0.77117604,
//                                   0.10042859,
//                                   0.68555583,
//                                   0.93192629,
//                                   0.39255794,
//                                   0.99285767,
//                                   0.88129697,
//                                   0.56599014,
//                                   0.03828527};

//     migraphx::shape x2_shape{migraphx::shape::float_type, {3, 4, 5}};
//     std::vector<float> x2_data = {
//         0.19665868, 0.49490562, 0.73175228, 0.89251999, 0.08735652, 0.25944536, 0.37003717,
//         0.09387889, 0.75490936, 0.81022481, 0.9987667,  0.04082882, 0.26160334, 0.85590193,
//         0.80221833, 0.11203218, 0.31701572, 0.45973754, 0.3452479,  0.85151585, 0.86455042,
//         0.19206577, 0.09922319, 0.58911914, 0.15871974, 0.61540675, 0.21682354, 0.69036427,
//         0.77451157, 0.91950467, 0.52659111, 0.80857867, 0.63179264, 0.10085509, 0.96412482,
//         0.42412458, 0.0330562,  0.13279482, 0.39372801, 0.80698385, 0.1182876,  0.75943908,
//         0.59421519, 0.66827559, 0.09009574, 0.66649037, 0.43015355, 0.37795428, 0.11304274,
//         0.37406792, 0.33043231, 0.32357327, 0.38079892, 0.42659918, 0.55308245, 0.49437723,
//         0.95926415, 0.99762983, 0.70624046, 0.24298556};

//     migraphx::parameter_map pm;
//     pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
//     pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

//     auto result = p.eval(pm).back();
//     std::vector<float> result_vector;
//     result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

//     std::vector<float> gold = {
//         0.3195768,  0.92158614, 0.98164236, 1.20559466, 0.14507291, 0.71879884, 0.60203336,
//         0.40083822, 0.73744823, 0.97361497, 1.04963956, 0.33451816, 0.5262512,  0.96263736,
//         1.09464615, 0.46791396, 0.90542384, 1.05180592, 0.78995572, 0.90429304, 0.64010028,
//         1.29062741, 1.31086115, 1.72652878, 0.23316878, 1.14509684, 0.85704442, 0.73375098,
//         1.1197959,  1.48742487, 1.46556673, 0.67672563, 0.86988939, 1.26078125, 1.67521536,
//         0.76174542, 1.26082452, 1.47107559, 1.17750291, 1.351588,   0.66717038, 0.57394148,
//         0.72380011, 1.1455959,  0.17027018, 0.60247933, 0.46530117, 0.48794463, 1.10799312,
//         1.24880054, 1.19090614, 0.50601796, 0.60271763, 0.82771923, 1.27385264, 0.35771131,
//         0.33482015, 0.51852039, 0.5541507,  1.21648601};
//     EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
// }

TEST_CASE(einsum_element_wise_multiplication_and_row_sum_test)
{
    migraphx::program p =
        migraphx::parse_onnx("einsum_element_wise_multiplication_and_row_sum_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {3}};
    std::vector<float> x1_data = {0.66866322, 0.01371844, 0.85036724};

    migraphx::shape x2_shape{migraphx::shape::float_type, {3, 4}};
    std::vector<float> x2_data = {0.72487469,
                                  0.24707426,
                                  0.8735483,
                                  0.04525622,
                                  0.52379655,
                                  0.32056461,
                                  0.51596208,
                                  0.10696902,
                                  0.08682559,
                                  0.95054461,
                                  0.16377484,
                                  0.61029108};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.2642773, 0.02012896, 1.54038595};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

// TODO /code/AMDMIGraphX/src/include/migraphx/op/dot.hpp:84: compute_shape: DOT: static outer
// dimensions of A and B mismatch: {1, 4, 2} x {2, 4, 1} TEST_CASE(einsum_3_inputs_test)
// {
//     migraphx::program p = migraphx::parse_onnx("einsum_3_inputs_test.onnx");
//     p.compile(migraphx::make_target("ref"));

//     migraphx::shape x1_shape{migraphx::shape::float_type, {2, 2, 2}};
//     std::vector<float> x1_data = {0.78808491,
//                                   0.6661874,
//                                   0.4170594,
//                                   0.80972418,
//                                   0.22687053,
//                                   0.52144567,
//                                   0.70463225,
//                                   0.8934412};

//     migraphx::shape x2_shape{migraphx::shape::float_type, {2, 2}};
//     std::vector<float> x2_data = {0.98518483, 0.61526655, 0.89011461, 0.02600793};

//     migraphx::shape x3_shape{migraphx::shape::float_type, {2, 2, 2}};
//     std::vector<float> x3_data = {0.04135729,
//                                   0.36723732,
//                                   0.82196749,
//                                   0.35332048,
//                                   0.92673273,
//                                   0.50014512,
//                                   0.91129541,
//                                   0.97557965};

//     migraphx::parameter_map pm;
//     pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
//     pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};
//     pm["x3"] = migraphx::argument{x3_shape, x3_data.data()};

//     auto result = p.eval(pm).back();
//     std::vector<float> result_vector;
//     result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

//     std::vector<float> gold = {1.54312876,
//                                0.59155446,
//                                1.19274407,
//                                0.56709538,
//                                2.79449706,
//                                1.61644006,
//                                2.15997517,
//                                1.5496049};
//     EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
// }

TEST_CASE(einsum_common_1_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_common_1_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {2, 2, 2, 2}};
    std::vector<float> x1_data = {0.35498396,
                                  0.92145607,
                                  0.81807284,
                                  0.37990484,
                                  0.22314499,
                                  0.90337144,
                                  0.02492543,
                                  0.36666091,
                                  0.33262049,
                                  0.37052745,
                                  0.01950226,
                                  0.83690205,
                                  0.61551503,
                                  0.55244304,
                                  0.62696715,
                                  0.74933671};

    migraphx::shape x2_shape{migraphx::shape::float_type, {2, 2, 2, 2}};
    std::vector<float> x2_data = {0.44903857,
                                  0.47304138,
                                  0.63679145,
                                  0.78101353,
                                  0.41525864,
                                  0.57356733,
                                  0.83636479,
                                  0.01236986,
                                  0.10068789,
                                  0.46623025,
                                  0.29825429,
                                  0.56816588,
                                  0.00558546,
                                  0.91900877,
                                  0.74972012,
                                  0.4509882};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.59528833,
                               0.52753278,
                               0.67592725,
                               0.61080723,
                               0.81765261,
                               0.30223943,
                               0.68890669,
                               0.0253823,
                               0.20624196,
                               0.31954056,
                               0.34237582,
                               0.51113793,
                               0.48131582,
                               0.6127432,
                               0.39205418,
                               0.8079919};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_common_2_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_common_2_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {2, 2, 2, 2}};
    std::vector<float> x1_data = {0.77858647,
                                  0.8659616,
                                  0.89981848,
                                  0.45454779,
                                  0.27364842,
                                  0.69225887,
                                  0.01304595,
                                  0.14404551,
                                  0.47394644,
                                  0.39058325,
                                  0.977306,
                                  0.90298946,
                                  0.01456065,
                                  0.70478062,
                                  0.92796867,
                                  0.00407166};

    migraphx::shape x2_shape{migraphx::shape::float_type, {2, 2, 2, 2}};
    std::vector<float> x2_data = {0.12299003,
                                  0.42677007,
                                  0.84213152,
                                  0.26884624,
                                  0.85685616,
                                  0.53033816,
                                  0.61543941,
                                  0.00586418,
                                  0.79310638,
                                  0.66468861,
                                  0.22797244,
                                  0.32789713,
                                  0.01537162,
                                  0.28328088,
                                  0.39257709,
                                  0.83954883};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {2.51890769,
                               1.78883817,
                               2.11484282,
                               1.38804189,
                               2.81881969,
                               1.09537142,
                               3.0398521,
                               1.07377846};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_common_3_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_common_3_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {2, 2, 2, 2}};
    std::vector<float> x1_data = {0.22151958,
                                  0.19284961,
                                  0.8126814,
                                  0.02360209,
                                  0.99137254,
                                  0.0550951,
                                  0.34794661,
                                  0.03083101,
                                  0.03127261,
                                  0.04609321,
                                  0.02422953,
                                  0.30878066,
                                  0.42532866,
                                  0.02191982,
                                  0.34276933,
                                  0.66997637};

    migraphx::shape x2_shape{migraphx::shape::float_type, {2, 2, 2, 2}};
    std::vector<float> x2_data = {0.76051399,
                                  0.92365044,
                                  0.14703117,
                                  0.07201171,
                                  0.81879942,
                                  0.91050362,
                                  0.90936259,
                                  0.94197062,
                                  0.73971579,
                                  0.08809791,
                                  0.17392649,
                                  0.36623704,
                                  0.23731799,
                                  0.67476051,
                                  0.97480632,
                                  0.35175013};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.62099637,
                               2.20329706,
                               0.6457657,
                               1.61829179,
                               0.4142793,
                               0.52881853,
                               2.00689201,
                               2.20807455};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(einsum_common_4_test)
{
    migraphx::program p = migraphx::parse_onnx("einsum_common_4_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x1_shape{migraphx::shape::float_type, {2, 2, 3, 2}};
    std::vector<float> x1_data = {0.56144416, 0.70795103, 0.10800643, 0.85461707, 0.53053745,
                                  0.42957473, 0.2801385,  0.91878799, 0.51160639, 0.90354742,
                                  0.83131358, 0.84237736, 0.01078178, 0.75952001, 0.74426499,
                                  0.70506648, 0.65528756, 0.54674358, 0.3923791,  0.33558121,
                                  0.18089114, 0.41982192, 0.50568299, 0.83929267};

    migraphx::shape x2_shape{migraphx::shape::float_type, {2, 2, 4, 2}};
    std::vector<float> x2_data = {
        0.71114916, 0.10373848, 0.85011488, 0.08836512, 0.01426097, 0.63389153, 0.3714056,
        0.42466907, 0.5412509,  0.12682203, 0.88595126, 0.09839624, 0.10689487, 0.1196194,
        0.5887543,  0.51683836, 0.50278953, 0.94187525, 0.98227159, 0.57961915, 0.12739494,
        0.59140361, 0.34997506, 0.43158845, 0.60170823, 0.06098434, 0.24573198, 0.15357368,
        0.99864135, 0.92721276, 0.81457582, 0.49836327};

    migraphx::parameter_map pm;
    pm["x1"] = migraphx::argument{x1_shape, x1_data.data()};
    pm["x2"] = migraphx::argument{x2_shape, x2_data.data()};

    auto result = p.eval(pm).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        0.4727123,  0.53985021, 0.4567709,  0.50916841, 0.16546536, 0.16733621, 0.5432748,
        0.40304363, 0.42185469, 0.48897721, 0.27986976, 0.37947168, 0.26814778, 0.33859434,
        0.13985024, 0.63979763, 0.39149714, 0.54216399, 0.1627699,  0.76819843, 0.55678123,
        0.81939007, 0.18962783, 0.92481237, 0.72079407, 0.45082298, 0.45055642, 0.33157342,
        1.03829331, 1.13974038, 0.51179445, 0.56477273, 0.84443597, 0.9605734,  0.40682645,
        0.46530252, 0.25656293, 0.14795654, 0.70300118, 0.48686388, 0.13444625, 0.10892434,
        0.56990961, 0.35657337, 0.35545733, 0.25315575, 1.28319881, 0.83018978};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
