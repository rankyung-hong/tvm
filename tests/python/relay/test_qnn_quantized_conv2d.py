# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
import numpy as np
from tvm import relay
from tvm.contrib import graph_runtime


def test_qnn_conv2d():
    def verify(func, goldens):
        with relay.build_config(opt_level=0):
            golden_data, golden_weight, golden_output = goldens
            parameters = {"weight": golden_weight}
            graph, lib, params = relay.build(func, "llvm", params=parameters)
            mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
            mod.set_input("quantized_data", golden_data)
            mod.set_input(**params)
            mod.run()
            res = mod.get_output(0).asnumpy()
            np.testing.assert_equal(res, golden_output)

    def get_func(data_shape, data_dtype, weight_shape, weight_dtype,
                 input_zero_point, kernel_zero_point, strides, padding, dilation,
                 data_layout, kernel_layout, out_layout, out_dtype):
        quantized_data = relay.var("quantized_data", shape=data_shape, dtype=data_dtype)
        weight = relay.var("weight", shape=weight_shape, dtype=weight_dtype)

        func = relay.qnn.op.conv2d(data=quantized_data,
                                   weight=weight,
                                   input_zero_point=input_zero_point,
                                   kernel_zero_point=kernel_zero_point,
                                   strides=strides,
                                   padding=padding,
                                   dilation=dilation,
                                   channels=weight_shape[0],
                                   kernel_size=[weight_shape[2], weight_shape[3]],
                                   data_layout=data_layout,
                                   kernel_layout=kernel_layout,
                                   out_layout=out_layout,
                                   out_dtype=out_dtype)

        func = relay.Function(relay.analysis.free_vars(func), func)
        func = relay.Module.from_expr(func)
        func = relay.qnn.transform.QnnLower()(func)
        print(func)
        return func

    def run_tests():
        # Get the same input data as TF testcases in a NCHW layout
        def get_golden_input(data_dtype, shape_NCHW, zero_point):
            data_shape_NHWC = [shape_NCHW[0], shape_NCHW[2], shape_NCHW[3], shape_NCHW[1]]
            total_size = np.prod(data_shape_NHWC)
            golden_data = np.arange(1 + zero_point, total_size + 1 + zero_point, dtype=data_dtype)
            golden_data = golden_data.reshape(data_shape_NHWC)
            return np.transpose(golden_data, [0, 3, 1, 2])

        # Get the same weight data as TF testcases in a OIHW layout
        def get_golden_weight(weight_dtype, shape_OIHW, zero_point):
            weight_shape_HWIO = [shape_OIHW[2], shape_OIHW[3], shape_OIHW[1], shape_OIHW[0]]
            total_size = np.prod(weight_shape_HWIO)
            golden_weight = np.arange(1 + zero_point, total_size + 1 + zero_point, dtype=weight_dtype)
            golden_weight = golden_weight.reshape(weight_shape_HWIO)
            return np.transpose(golden_weight, [3, 2, 0, 1])

        # Get the same output data as TF testcases in a NCHW layout
        def get_golden_output(output_data, output_dtype, shape_NCHW):
            output_shape_NHWC = [shape_NCHW[0], shape_NCHW[2], shape_NCHW[3], shape_NCHW[1]]
            golden_output = np.array(output_data, dtype=output_dtype).reshape(output_shape_NHWC)
            return np.transpose(golden_output, [0, 3, 1, 2])

        def test_Conv2D_1x1_Filter():
            data_dtype = 'int32'
            weight_dtype = 'int32'
            output_dtype = 'int32'
            input_zero_point = 1
            kernel_zero_point = 1
            data_shape_NCHW = [1, 3, 2, 3]
            weight_shape_OIHW = [3, 3, 1, 1]
            output_shape_NCHW = [1, 3, 2, 3]
            output_data = [
                30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0,
                204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0
            ]
            golden_data = get_golden_input(data_dtype, data_shape_NCHW, input_zero_point)
            golden_weight = get_golden_weight(weight_dtype, weight_shape_OIHW, kernel_zero_point)
            golden_output = get_golden_output(output_data, data_dtype, output_shape_NCHW)

            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            weight_shape=weight_shape_OIHW,
                            weight_dtype=weight_dtype,
                            input_zero_point=input_zero_point,
                            kernel_zero_point=kernel_zero_point,
                            strides=[1, 1],
                            padding=[0, 0],
                            dilation=[1, 1],
                            data_layout="NCHW",
                            kernel_layout="OIHW",
                            out_layout="NCHW",
                            out_dtype=output_dtype)
            verify(func, (golden_data, golden_weight, golden_output))

        def test_Conv2D_2x2_Filter():
            data_dtype = 'int32'
            weight_dtype = 'int32'
            output_dtype = 'int32'
            input_zero_point = 1
            kernel_zero_point = 1
            data_shape_NCHW = [1, 3, 2, 3]
            weight_shape_OIHW = [3, 3, 2, 2]
            output_shape_NCHW = [1, 3, 1, 2]
            output_data = [2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0]
            golden_data = get_golden_input(data_dtype, data_shape_NCHW, input_zero_point)
            golden_weight = get_golden_weight(weight_dtype, weight_shape_OIHW, kernel_zero_point)
            golden_output = get_golden_output(output_data, data_dtype, output_shape_NCHW)

            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            weight_shape=weight_shape_OIHW,
                            weight_dtype=weight_dtype,
                            input_zero_point=input_zero_point,
                            kernel_zero_point=kernel_zero_point,
                            strides=[1, 1],
                            padding=[0, 0],
                            dilation=[1, 1],
                            data_layout="NCHW",
                            kernel_layout="OIHW",
                            out_layout="NCHW",
                            out_dtype=output_dtype)
            verify(func, (golden_data, golden_weight, golden_output))

        def test_Conv2D_2x2_Filter_2x1_Dilation():
            data_dtype = 'int32'
            weight_dtype = 'int32'
            output_dtype = 'int32'
            input_zero_point = 1
            kernel_zero_point = 1
            data_shape_NCHW = [1, 1, 4, 4]
            weight_shape_OIHW = [1, 1, 2, 2]
            output_shape_NCHW = [1, 1, 2, 3]
            output_data = [72, 82, 92, 112, 122, 132]
            golden_data = get_golden_input(data_dtype, data_shape_NCHW, input_zero_point)
            golden_weight = get_golden_weight(weight_dtype, weight_shape_OIHW, kernel_zero_point)
            golden_output = get_golden_output(output_data, data_dtype, output_shape_NCHW)

            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            weight_shape=weight_shape_OIHW,
                            weight_dtype=weight_dtype,
                            input_zero_point=input_zero_point,
                            kernel_zero_point=kernel_zero_point,
                            strides=[1, 1],
                            padding=[0, 0],
                            dilation=[2, 1],
                            data_layout="NCHW",
                            kernel_layout="OIHW",
                            out_layout="NCHW",
                            out_dtype=output_dtype)
            verify(func, (golden_data, golden_weight, golden_output))

        def test_Conv2D_1x2_Filter():
            data_dtype = 'int32'
            weight_dtype = 'int32'
            output_dtype = 'int32'
            input_zero_point = 1
            kernel_zero_point = 1
            data_shape_NCHW = [1, 3, 2, 3]
            weight_shape_OIHW = [3, 3, 1, 2]
            output_shape_NCHW = [1, 3, 2, 2]
            output_data = [
                231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0,
                936.0, 1029.0
            ]
            golden_data = get_golden_input(data_dtype, data_shape_NCHW, input_zero_point)
            golden_weight = get_golden_weight(weight_dtype, weight_shape_OIHW, kernel_zero_point)
            golden_output = get_golden_output(output_data, data_dtype, output_shape_NCHW)

            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            weight_shape=weight_shape_OIHW,
                            weight_dtype=weight_dtype,
                            input_zero_point=input_zero_point,
                            kernel_zero_point=kernel_zero_point,
                            strides=[1, 1],
                            padding=[0, 0],
                            dilation=[1, 1],
                            data_layout="NCHW",
                            kernel_layout="OIHW",
                            out_layout="NCHW",
                            out_dtype=output_dtype)
            verify(func, (golden_data, golden_weight, golden_output))

        def test_Conv2D_2x2_Filter_Stride2():
            data_dtype = 'int32'
            weight_dtype = 'int32'
            output_dtype = 'int32'
            input_zero_point = 1
            kernel_zero_point = 1
            data_shape_NCHW = [1, 3, 2, 3]
            weight_shape_OIHW = [3, 3, 2, 2]
            output_shape_NCHW = [1, 3, 1, 1]
            output_data = [2271.0, 2367.0, 2463.0]
            golden_data = get_golden_input(data_dtype, data_shape_NCHW, input_zero_point)
            golden_weight = get_golden_weight(weight_dtype, weight_shape_OIHW, kernel_zero_point)
            golden_output = get_golden_output(output_data, data_dtype, output_shape_NCHW)

            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            weight_shape=weight_shape_OIHW,
                            weight_dtype=weight_dtype,
                            input_zero_point=input_zero_point,
                            kernel_zero_point=kernel_zero_point,
                            strides=[2, 2],
                            padding=[0, 0],
                            dilation=[1, 1],
                            data_layout="NCHW",
                            kernel_layout="OIHW",
                            out_layout="NCHW",
                            out_dtype=output_dtype)
            verify(func, (golden_data, golden_weight, golden_output))

        def test_Conv2D_2x2_Filter_Stride2_Same():
            data_dtype = 'int32'
            weight_dtype = 'int32'
            output_dtype = 'int32'
            input_zero_point = 0
            kernel_zero_point = 0
            data_shape_NCHW = [1, 3, 2, 3]
            weight_shape_OIHW = [3, 3, 2, 2]
            output_shape_NCHW = [1, 3, 1, 2]
            output_data = [2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]
            golden_data = get_golden_input(data_dtype, data_shape_NCHW, input_zero_point)
            golden_weight = get_golden_weight(weight_dtype, weight_shape_OIHW, kernel_zero_point)
            golden_output = get_golden_output(output_data, data_dtype, output_shape_NCHW)

            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            weight_shape=weight_shape_OIHW,
                            weight_dtype=weight_dtype,
                            input_zero_point=input_zero_point,
                            kernel_zero_point=kernel_zero_point,
                            strides=[2, 2],
                            padding=[0, 1],
                            dilation=[1, 1],
                            data_layout="NCHW",
                            kernel_layout="OIHW",
                            out_layout="NCHW",
                            out_dtype=output_dtype)
            verify(func, (golden_data, golden_weight, golden_output))

        def test_Conv2D_Empty_Dilation():
            data_dtype = 'int32'
            weight_dtype = 'int32'
            output_dtype = 'int32'
            input_zero_point = 1
            kernel_zero_point = 1
            data_shape_NCHW = [0, 3, 2, 3]
            weight_shape_OIHW = [3, 3, 1, 1]
            output_shape_NCHW = [0, 3, 2, 3]
            output_data = np.zeros(output_shape_NCHW)
            golden_data = get_golden_input(data_dtype, data_shape_NCHW, input_zero_point)
            golden_weight = get_golden_weight(weight_dtype, weight_shape_OIHW, kernel_zero_point)
            golden_output = get_golden_output(output_data, data_dtype, output_shape_NCHW)

            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            weight_shape=weight_shape_OIHW,
                            weight_dtype=weight_dtype,
                            input_zero_point=input_zero_point,
                            kernel_zero_point=kernel_zero_point,
                            strides=[1, 1],
                            padding=[0, 0],
                            dilation=[2, 1],
                            data_layout="NCHW",
                            kernel_layout="OIHW",
                            out_layout="NCHW",
                            out_dtype=output_dtype)
            verify(func, (golden_data, golden_weight, golden_output))

        def test_Conv2D_2x2_Filter_Dilation():
            data_dtype = 'int32'
            weight_dtype = 'int32'
            output_dtype = 'int32'
            input_zero_point = 1
            kernel_zero_point = 1
            data_shape_NCHW = [1, 3, 2, 3]
            weight_shape_OIHW = [3, 3, 2, 2]
            output_shape_NCHW = [1, 3, 1, 1]
            output_data = [2667, 2781, 2895]
            golden_data = get_golden_input(data_dtype, data_shape_NCHW, input_zero_point)
            golden_weight = get_golden_weight(weight_dtype, weight_shape_OIHW, kernel_zero_point)
            golden_output = get_golden_output(output_data, data_dtype, output_shape_NCHW)

            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            weight_shape=weight_shape_OIHW,
                            weight_dtype=weight_dtype,
                            input_zero_point=input_zero_point,
                            kernel_zero_point=kernel_zero_point,
                            strides=[1, 1],
                            padding=[0, 0],
                            dilation=[1, 2],
                            data_layout="NCHW",
                            kernel_layout="OIHW",
                            out_layout="NCHW",
                            out_dtype=output_dtype)
            verify(func, (golden_data, golden_weight, golden_output))

        def test_Conv2D_1x2_Filter_Dilation():
            data_dtype = 'int32'
            weight_dtype = 'int32'
            output_dtype = 'int32'
            input_zero_point = 1
            kernel_zero_point = 1
            data_shape_NCHW = [1, 3, 2, 3]
            weight_shape_OIHW = [3, 3, 1, 2]
            output_shape_NCHW = [1, 3, 2, 2]
            output_data = [231,  252,  273,  384,  423,  462,
                           690,  765,  840,  843,  936, 1029]
            golden_data = get_golden_input(data_dtype, data_shape_NCHW, input_zero_point)
            golden_weight = get_golden_weight(weight_dtype, weight_shape_OIHW, kernel_zero_point)
            golden_output = get_golden_output(output_data, data_dtype, output_shape_NCHW)

            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            weight_shape=weight_shape_OIHW,
                            weight_dtype=weight_dtype,
                            input_zero_point=input_zero_point,
                            kernel_zero_point=kernel_zero_point,
                            strides=[1, 1],
                            padding=[0, 0],
                            dilation=[2, 1],
                            data_layout="NCHW",
                            kernel_layout="OIHW",
                            out_layout="NCHW",
                            out_dtype=output_dtype)
            verify(func, (golden_data, golden_weight, golden_output))

        def test_Conv2D_KernelSize_Matches_InputSize_Dilation():
            data_dtype = 'int32'
            weight_dtype = 'int32'
            output_dtype = 'int32'
            input_zero_point = 1
            kernel_zero_point = 1
            data_shape_NCHW = [1, 1, 3, 3]
            weight_shape_OIHW = [2, 1, 2, 2]
            output_shape_NCHW = [1, 2, 1, 1]
            output_data = [108, 128]
            golden_data = get_golden_input(data_dtype, data_shape_NCHW, input_zero_point)
            golden_weight = get_golden_weight(weight_dtype, weight_shape_OIHW, kernel_zero_point)
            golden_output = get_golden_output(output_data, data_dtype, output_shape_NCHW)

            func = get_func(data_shape=data_shape_NCHW,
                            data_dtype=data_dtype,
                            weight_shape=weight_shape_OIHW,
                            weight_dtype=weight_dtype,
                            input_zero_point=input_zero_point,
                            kernel_zero_point=kernel_zero_point,
                            strides=[1, 1],
                            padding=[0, 0],
                            dilation=[2, 2],
                            data_layout="NCHW",
                            kernel_layout="OIHW",
                            out_layout="NCHW",
                            out_dtype=output_dtype)
            verify(func, (golden_data, golden_weight, golden_output))

        test_Conv2D_1x1_Filter()
        test_Conv2D_2x2_Filter()
        test_Conv2D_2x2_Filter_2x1_Dilation()
        test_Conv2D_1x2_Filter()
        test_Conv2D_2x2_Filter_Stride2()
        # test_Conv2D_2x2_Filter_Stride2_Same()
        test_Conv2D_Empty_Dilation()
        test_Conv2D_2x2_Filter_Dilation()
        test_Conv2D_1x2_Filter_Dilation()
        test_Conv2D_KernelSize_Matches_InputSize_Dilation()

    run_tests()


if __name__ == "__main__":
    test_qnn_conv2d()
