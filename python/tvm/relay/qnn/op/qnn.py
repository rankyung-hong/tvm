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
# pylint: disable=invalid-name
"""QNN dialect operators."""

from __future__ import absolute_import as _abs
from . import _make
from tvm import relay


def requantize(data,
               input_scale,
               input_zero_point,
               output_scale,
               output_zero_point,
               rounding="TONEAREST",
               out_dtype="int8"):
    r"""Requantized operator.

    The requantize operator converts one quantized tensor representation to
    another quantized tensor representation. For the output tensor, we are
    provided with output scale and zero point. The computation is as follows

    Q_output = zp_output +  (scale_input)/(scale_output) * (Q_input - zp_input)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    input_scale: float
        The quantization scale for the input tensor.

    input_zero_point: int
        The zero point of the input tensor.

    output_scale: float
        The quantization scale for the output tensor.

    output_zero_point: int
        The zero point of the output tensor.

    rounding : string, optional
        Defines the rounding direction when the value is midway between two
        representable values.

    out_dtype : str, optional
        Specifies the output data type.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.requantize(data,
                            input_scale,
                            input_zero_point,
                            output_scale,
                            output_zero_point,
                            rounding,
                            out_dtype)


def quantized_dense(data, weight, input_zero_point, kernel_zero_point, units=None, out_dtype="int32"):
    """Dense operator.
    Applies a linear transformation

    .. math::

    `Y = X * W`

    Parameters
    ----------
    data : tvm.relay.Expr
        The quantied input data to the operator.

    weight : tvm.relay.Expr
        The quantized weight expressions.

    units : int, optional
        Number of hidden units of the dense transformation.

    out_dtype : str, optional
        Specifies the output data type for mixed precision dense can be int32 or int16.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.dense(data, weight, units, input_zero_point, kernel_zero_point, out_dtype)


def quantize(input_data, output_zero_point, output_scale, out_dtype='int8'):
    r""" Quantize op
     This operator takes float32 as input and produces quantized int8 or unit8 as output.
     The input tensor can be of any shape. The output shape is the same as input shape.
     ..math::
            \mbox{out}[x] =
                \mbox{clamp(round(input_tensor/output_scale) + output_zero_point);
                 out_dtype::min, out_dtype::max}
     Parameters
    ----------
    input_data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    output_zero_point :
        The output zero_point.
    output_scale:
        The output scale.
    input_dtype:
        The data type of the input tensor. Can be [int8, uint8]
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.quantize(input_data, output_zero_point, output_scale, out_dtype)


def dequantize(input_data, input_zero_point, input_scale):
    r""" Dequantize op
     This operator takes quantized int8 and unit8 as input and produces
    dequantized float32 as output. The output shape is the same as input shape. The input
    tensor can be of any shape.
     Parameters
    ----------
    input_data : tvm.relay.Expr
        The input tensor to be dequantized. Can be of type [int8, uint8].
    input_zero_point :
        The output zero_point.
    input_scale:
        The output scale.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.dequantize(input_data, input_zero_point, input_scale)


def max_pool2d(data,
               input_zero_point,
               pool_size=(1, 1),
               strides=(1, 1),
               padding=(0, 0),
               layout="NCHW",
               ceil_mode=False):
    r"""Quantized 2D maximum pooling operator.

    This operator takes quantized data as input and does 2D max value calculation
    with in pool_size sized window by striding defined by stride


    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with the following rule:

    with data of shape (b, c, h, w) and pool_size (kh, kw)

    .. math::

        \mbox{out}(b, c, y, x)  = \max_{m=0, \ldots, kh-1} \max_{n=0, \ldots, kw-1}
             \mbox{data}(b, c, \mbox{stride}[0] * y + m, \mbox{stride}[1] * x + n)

    Padding is applied to quantized_data before the computation.
    ceil_mode is used to take ceil or floor while computing out shape.
    This operator accepts data layout specification.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input quantized_data to the operator.

    input_zero_point: int
       The zero point of the data distribution.

    pool_size : tuple of int, optional
        The size of pooling window.

    strides : tuple of int, optional
        The strides of pooling.

    padding : tuple of int, optional
        The padding for pooling.

    layout : str, optional
        Layout of the input.

    ceil_mode : bool, optional
        To enable or disable ceil while pooling.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    casted_data = relay.cast(data, dtype="int32")
    shifted_data = relay.subtract(casted_data, relay.const(input_zero_point, "int32"))
    return relay.nn.max_pool2d(shifted_data,
                               pool_size=pool_size,
                               strides=strides,
                               padding=padding,
                               layout=layout,
                               ceil_mode=ceil_mode)


def avg_pool2d(data,
               input_zero_point,
               pool_size=(1, 1),
               strides=(1, 1),
               padding=(0, 0),
               layout="NCHW",
               ceil_mode=False,
               count_include_pad=False):
    r"""Quantized 2D average pooling operator.

    This operator takes quantized data as input and does 2D average value calculation
    with in pool_size sized window by striding defined by stride


    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with the following rule:

    with data of shape (b, c, h, w), pool_size (kh, kw)

    .. math::

        \mbox{out}(b, c, y, x)  = \frac{1}{kh * kw} \sum_{m=0}^{kh-1} \sum_{n=0}^{kw-1}
             \mbox{data}(b, c, \mbox{stride}[0] * y + m, \mbox{stride}[1] * x + n)

    Padding is applied to quantized_data before the computation.
    ceil_mode is used to take ceil or floor while computing out shape.
    count_include_pad indicates including or excluding padded input values in computation.
    This operator accepts data layout specification.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input quantized_data to the operator.

    input_zero_point: int
       The size of pooling window.

    pool_size : tuple of int, optional
        The zero point of the data distribution.

    strides : tuple of int, optional
        The strides of pooling.

    padding : tuple of int, optional
        The padding for pooling.

    layout : str, optional
        Layout of the input.

    ceil_mode : bool, optional
        To enable or disable ceil while pooling.

    count_include_pad : bool, optional
        To include padding to compute the average.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    casted_data = relay.cast(data, dtype="int32")
    shifted_data = relay.subtract(casted_data, relay.const(input_zero_point, "int32"))
    return relay.nn.avg_pool2d(shifted_data,
                               pool_size=pool_size,
                               strides=strides,
                               padding=padding,
                               layout=layout,
                               ceil_mode=ceil_mode,
                               count_include_pad=count_include_pad)


def softmax(data,
            input_zero_point,
            axis=-1):
    r"""Computes softmax with quantized_data.

    .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

    .. note::
        This operator can be optimized away for inference.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input quantized_data to the operator.

    input_zero_point: int
       The zero point of the data distribution.

    axis: int, optional
        The axis to sum over when computing softmax

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    casted_data = relay.cast(data, dtype="float32")
    shifted_data = relay.subtract(casted_data, relay.const(float(input_zero_point), "float32"))
    return relay.nn.softmax(shifted_data, axis)


def reshape(data, newshape):
    """Reshapes the quantized input array.

    Example::

    To give user more convenience in without doing manual shape inference,
    some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}.
    The significance of each is explained below:

    - ``0``  copy this dimension from the input to the output shape.

    Example::

    - data.shape = (2,3,4), newshape = (4,0,2), result.shape = (4,3,2)
    - data.shape = (2,3,4), newshape = (2,0,0), result.shape = (2,3,4)

    - ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions
    keeping the size of the new array same as that of the input array.
    At most one dimension of shape can be -1.

    Example::

    - data.shape = (2,3,4), newshape = (6,1,-1), result.shape = (6,1,4)
    - data.shape = (2,3,4), newshape = (3,-1,8), result.shape = (3,1,8)
    - data.shape = (2,3,4), newshape = (-1,), result.shape = (24,)

    - ``-2`` copy all/remainder of the input dimensions to the output shape.

    Example::

    - data.shape = (2,3,4), newshape = (-2,), result.shape = (2,3,4)
    - data.shape = (2,3,4), newshape = (2,-2), result.shape = (2,3,4)
    - data.shape = (2,3,4), newshape = (-2,1,1), result.shape = (2,3,4,1,1)

    - ``-3`` use the product of two consecutive dimensions of the input shape
    as the output dimension.

    Example::

    - data.shape = (2,3,4), newshape = (-3,4), result.shape = (6,4)
    - data.shape = (2,3,4,5), newshape = (-3,-3), result.shape = (6,20)
    - data.shape = (2,3,4), newshape = (0,-3), result.shape = (2,12)
    - data.shape = (2,3,4), newshape = (-3,-2), result.shape = (6,4)

    - ``-4`` split one dimension of the input into two dimensions passed subsequent
    to -4 in shape (can contain -1).

    Example::

    - data.shape = (2,3,4), newshape = (-4,1,2,-2), result.shape = (1,2,3,4)
    - data.shape = (2,3,4), newshape = (2,-4,-1,3,-2), result.shape = (2,1,3,4)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input quantized_data to the operator.

    newshape : Union[int, Tuple[int], List[int]]
        The new shape. Should be compatible with the original shape.

    Returns
    -------
    result : relay.Expr
        The reshaped result.
    """
    return relay.reshape(data, newshape)


def concatenate(data,
                input_scales,
                input_zero_points,
                output_scale,
                output_zero_point,
                output_dtype,
                axis):
    """Concatenate the quantized input tensors along the given axis.

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple[relay.Expr])
        A list of tensors.

    input_scales : List[float32]
        A list of scales of quantized_data

    input_zero_points : List[int32]
        A list of zero points of quantized_data

    output_scale : float32
        A scales of output

    output_zero_point : int32
        A zero points of output

    axis : int
        The axis along which the tensors are concatenated.

    Returns
    -------
    result: relay.Expr
        The concatenated tensor.
    """
    quantized_data = list(data)
    tuned_quantized_data = list()
    for idx, qdata in enumerate(quantized_data):
        qdata_scale = input_scales[idx]
        qdata_zero_point = input_zero_points[idx]
        if qdata_scale != output_scale or qdata_zero_point != output_zero_point:
            tuned_quantized_data.append(requantize(quantized_data[idx],
                                                   input_scale=qdata_scale,
                                                   input_zero_point=qdata_zero_point,
                                                   output_scale=output_scale,
                                                   output_zero_point=output_zero_point,
                                                   out_dtype=output_dtype,
                                                   rounding="UPWARD"))
        else:
            tuned_quantized_data.append(quantized_data[idx])
    return relay.concatenate(tuple(tuned_quantized_data), axis)


def relu(data, input_zero_point):
    """Rectified linear unit.

    .. math::
       out = max(x, 0)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data

    input_zero_point: int
       The zero point of the data distribution.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    casted_data = relay.cast(data, dtype="int32")
    shifted_data = relay.subtract(casted_data, relay.const(input_zero_point, "int32"))
    return relay.nn.relu(shifted_data)


def conv2d(data,
           weight,
           input_zero_point,
           kernel_zero_point,
           strides=(1, 1),
           padding=(0, 0),
           dilation=(1, 1),
           groups=1,
           channels=None,
           kernel_size=None,
           data_layout="NCHW",
           kernel_layout="OIHW",
           out_layout="",
           out_dtype="int32"):
    r"""Quantized 2D convolution.

    This operator convolves quantized weight with quantized data. The scale of
    the output quantized tensor is the product of the weight_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    ouptut to (u)int8.


    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    input_zero_point: int
           The zero point of the data distribution.

    kernel_zero_point: int
           The zero point of the quantized_kernel distribution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution weight.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the weight.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.


    Returns
    -------
    result: relay.Expr
        The convoluted tensor.
    """

    return _make.conv2d(data, weight,
                        input_zero_point, kernel_zero_point,
                        strides, padding, dilation,
                        groups, channels, kernel_size,
                        data_layout, kernel_layout, out_layout, out_dtype)
