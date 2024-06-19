#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################

file(GLOB OP_BUILDER_SRCS CONFIGURE_DEPENDS *.cpp)
add_library(migraphx_op_builders ${OP_BUILDER_SRCS})
target_include_directories(migraphx_op_builders PRIVATE include)
set_target_properties(migraphx_op_builders PROPERTIES EXPORT_NAME op_builders)
migraphx_generate_export_header(migraphx_op_builders)
rocm_set_soversion(migraphx_op_builders ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(migraphx_op_builders)
if(NOT WIN32)
    target_link_libraries(migraphx_op_builders PRIVATE "-Wl,--exclude-libs,ALL")
endif()
target_link_libraries(migraphx_op_builders PUBLIC migraphx)

rocm_install_targets(
  TARGETS migraphx_op_builders
)