import os
import numpy as np
import tensorflow as tf

from tensorflow.python.layers import base
from tensorflow.python.layers import utils

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
from ._pca import *
from ._gau_conv_grad_op import *

from ctypes import cdll
cdll.LoadLibrary(os.path.join(os.path.dirname(os.path.realpath(__file__)),'libgau_conv_tensorflow.so'))

gau_conv_op_module = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)),'libgau_conv_op.so'))

class GAUGridMean(init_ops.Initializer):
    def __init__(self, gau_units, max_value, gau_unit_axis=2):
        self.gau_units = gau_units
        self.gau_unit_axis = gau_unit_axis
        self.max_value = pca(max_value)

    def __call__(self, shape, dtype=None, partition_info=None):

        assert len(shape) == 4, 
        seperated_gau_dims = False if shape[2]  == self.gau_units[0] * self.gau_units[1] else True

        if seperated_gau_dims is False:
            shape = [shape[1], self.gau_units[0], self.gau_units[1], shape[-1]]

        num_units = pca(shape[self.gau_unit_axis])

        vals = np.arange(num_units) * (2*self.max_value+1) / float(num_units) + (- 0.5+(2*self.max_value+1)/float(2*num_units)) - self.max_value

        shape_vals = np.ones(len(shape),dtype=np.int32)
        shape_vals[self.gau_unit_axis] = num_units

        vals = np.reshape(vals,shape_vals)
        vals = tf.convert_to_tensor(vals, np.float32)

        tile_rep_shape = list(shape)
        tile_rep_shape[num_units] = 1

        vals = tf.tile(input=vals,
                       multiples=tile_rep_shape)

        return vals if seperated_gau_dims else tf.reshape(vals,[1,shape[0],shape[1]*shape[2],shape[3]])

    def get_config(self):
        return {
            "gau_units": self.gau_units,
            "gau_unit_axis": self.gau_unit_axis,
            "max_value": self.max_value
        }

class ZeroNLast(init_ops.Initializer):

    def __init__(self, base_init, last_num_to_zero, axis):
        self.base_init = base_init
        self.last_num_to_zero = last_num_to_zero
        self.axis = axis

    def __call__(self, shape, dtype=None, partition_info=None):
        all_vals = self.base_init(shape, dtype, partition_info)

        shape_ones = all_vals.shape.as_list()
        shape_ones[self.axis] = all_vals.shape[self.axis] - self.last_num_to_zero

        shape_zeros = all_vals.shape.as_list()
        shape_zeros[self.axis] = self.last_num_to_zero

        ones_vals = tf.ones(shape_ones,dtype=tf.float32)
        zero_vals = tf.zeros(shape_zeros,dtype=tf.float32)

        valid_vals = tf.multiply(all_vals,tf.concat([ones_vals, zero_vals], axis=self.axis))

        return valid_vals

    def get_config(self):
        return {
            "last_num_to_zero": self.last_num_to_zero,
            "axis": self.axis,
            "base_init": self.base_init.get_config()
        }


class _GAUConvolution2d(object):
    def __init__(
            self,
            input_shape,
            num_output,
            gau_units,
            max_kernel_size,
            padding,
            data_format=None,
            strides=None,
            num_gau_units_ignore=0,
            mu_learning_rate_factor=500,
            gau_unit_border_bound=0.01,
            gau_unit_sigma_bound=0.01,
            gau_unit_single_dim=False,
            gau_aggregation_forbid_positive_dim1=False,
            gau_mu_interpolation=True,
            unit_testing=False,
            name=None):
        self.num_output = num_output
        self.padding = padding
        self.name = name
        self.gau_units = gau_units
        self.num_gau_units_ignore = num_gau_units_ignore
        self.max_kernel_size = max_kernel_size
        self.mu_learning_rate_factor = mu_learning_rate_factor
        self.gau_unit_border_bound = gau_unit_border_bound
        self.gau_unit_sigma_bound = gau_unit_sigma_bound
        self.gau_unit_single_dim = gau_unit_single_dim
        self.gau_aggregation_forbid_positive_dim1 = gau_aggregation_forbid_positive_dim1
        self.gau_mu_interpolation = gau_mu_interpolation
        self.unit_testing = unit_testing
        input_shape = input_shape
        if input_shape.ndims is None:
            raise ValueError("Rank of convolution must be known")
        if input_shape.ndims < 3 or input_shape.ndims > 5:
            raise ValueError(
                "`input` and `filter` must have rank at least 3 and at most 5")
        conv_dims = input_shape.ndims - 2
        if strides is None:
            strides = 1

        if conv_dims == 1:
            # not supported
            raise ValueError("One dimensional gauConv not supported - only two dimensions supported.")
        elif conv_dims == 2:
            if data_format is None or data_format == "NHWC":
                raise ValueError("data_format \"NHWC\" not supported - TODO: manually convert to NHWC.")
            elif data_format == "NCHW":
                pass
            else:
                raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")
            self.data_format = data_format
            self.gau_conv_op = gau_conv_op_module.gau_conv

       

        self.mean_max_allowed_offset = np.floor(self.max_kernel_size/2.0) - self.gau_unit_border_bound



    def __call__(self, inp, w, mu1, mu2, sigma):  # pylint: disable=redefined-builtin
        # first ensure mu1, mu2 and sigma values are all within bounds by cliping them
        mu1 = tf.clip_by_value(mu1, clip_value_min=-self.mean_max_allowed_offset, clip_value_max=self.mean_max_allowed_offset)
        mu2 = tf.clip_by_value(mu2, clip_value_min=-self.mean_max_allowed_offset, clip_value_max=self.mean_max_allowed_offset)

        if False:
            sigma = tf.clip_by_value(sigma, clip_value_min=self.gau_unit_sigma_bound, clip_value_max=1000)


       
        settings = dict(num_output=self.num_output,
                        number_units_x=self.gau_units[0],
                        number_units_y=self.gau_units[1],
                        number_units_ignore=self.num_gau_units_ignore,
                        kernel_size=self.max_kernel_size,
                        pad=self.padding,
                        component_border_bound=self.gau_unit_border_bound,
                        sigma_lower_bound=self.gau_unit_sigma_bound,
                        mu_learning_rate_factor=self.mu_learning_rate_factor,
                        single_dim_kernel=self.gau_unit_single_dim,
                        forbid_positive_dim1=self.gau_aggregation_forbid_positive_dim1,
                        use_interpolation=self.gau_mu_interpolation,
                        unit_testing=self.unit_testing)
        return self.gau_conv_op(
            input=inp,
            weights=w,
            mu1=mu1,
            mu2=mu2,
            sigma=sigma,
            name=self.name,
            **settings)

class GAUConv2d(base.Layer):

    # C++/CUDA implementation will compute N units at the same time - enforce this constraint !!
    gau_UNITS_GROUP = 2

    def __init__(self, filters,
                 gau_units,
                 max_kernel_size,
                 strides=1,
                 data_format='channels_first',
                 activation=None,
                 use_bias=True,
                 weight_initializer=init_ops.random_normal_initializer(stddev=0.1),
                 mu1_initializer=None,
                 mu2_initializer=None,
                 sigma_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 weight_regularizer=None,
                 mu1_regularizer=None,
                 mu2_regularizer=None,
                 sigma_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 weight_constraint=None,
                 mu1_constraint=None,
                 mu2_constraint=None,
                 sigma_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 mu_learning_rate_factor=500,
                 gau_unit_border_bound=0.01,
                 gau_unit_single_dim=False,
                 gau_aggregation_forbid_positive_dim1=False,
                 gau_sigma_trainable=False,
                 gau_mu_interpolation=True,
                 unit_testing=False,
                 name=None,
                 **kwargs):
        super(GAUConv2d, self).__init__(trainable=trainable, name=name,
                                    activity_regularizer=activity_regularizer,
                                    **kwargs)
        self.rank = 2
        self.filters = filters
        self.gau_units = utils.normalize_tuple(gau_units, self.rank, 'gau_components')
        self.max_kernel_size = max_kernel_size
        self.padding = np.floor(self.max_kernel_size/2.0)
        self.strides = strides
        self.data_format = utils.normalize_data_format(data_format)
        self.activation = activation
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint

        self.weight_initializer = weight_initializer
        self.weight_regularizer = weight_regularizer
        self.weight_constraint = weight_constraint

        self.mu1_initializer = mu1_initializer
        self.mu1_regularizer = mu1_regularizer
        self.mu1_constraint = mu1_constraint

        self.mu2_initializer = mu2_initializer
        self.mu2_regularizer = mu2_regularizer
        self.mu2_constraint = mu2_constraint

        self.sigma_initializer = sigma_initializer
        self.sigma_regularizer = sigma_regularizer
        self.sigma_constraint = sigma_constraint

        if self.mu1_initializer is None:
            self.mu1_initializer = gauGridMean(gau_units=self.gau_units, max_value=np.floor(self.max_kernel_size/2.0)-1, gau_unit_axis=2)
        if self.mu2_initializer is None:
            self.mu2_initializer = gauGridMean(gau_units=self.gau_units, max_value=np.floor(self.max_kernel_size/2.0)-1, gau_unit_axis=1)

        if self.sigma_initializer is None:
            self.sigma_initializer=init_ops.constant_initializer(0.5)

        self.mu_learning_rate_factor = mu_learning_rate_factor

        self.unit_testing = unit_testing

        self.input_spec = base.InputSpec(ndim=self.rank + 2)

        self.gau_unit_border_bound = gau_unit_border_bound
        self.num_gau_units_all = np.int32(np.prod(self.gau_units))
        self.num_gau_units_ignore = 0

        self.gau_mu_interpolation = gau_mu_interpolation

        self.gau_unit_single_dim = gau_unit_single_dim
        self.gau_aggregation_forbid_positive_dim1=gau_aggregation_forbid_positive_dim1
       
        if  self.num_gau_units_all % self.gau_UNITS_GROUP != 0:
            new_num_units = np.int32(np.ceil(self.num_gau_units_all / float(self.gau_UNITS_GROUP)) * self.gau_UNITS_GROUP)

            self.num_gau_units_ignore = new_num_units - self.num_gau_units_all

            if self.gau_units[0] < self.gau_units[1]:
                self.gau_units = (self.gau_units[0] + self.num_gau_units_ignore, self.gau_units[1])
            else:
                self.gau_units = (self.gau_units[0], self.gau_units[1] + self.num_gau_units_ignore)

            self.num_gau_units_all = new_num_units

            self.weight_initializer = ZeroNLast(self.weight_initializer, last_num_to_zero=self.num_gau_units_ignore, axis=2)


        self.gau_weights = None
        self.gau_mu1 = None
        self.gau_mu2 = None
        self.gau_sigma = None

        self.gau_sigma_trainable = gau_sigma_trainable


    def set_gau_variables_manually(self, w = None, mu1 = None, mu2 = None, sigma = None):

        if w is not None:
            self.gau_weights = w

        if mu1 is not None:
            self.gau_mu1 = mu1

        if mu2 is not None:
            self.gau_mu2 = mu2

        if sigma is not None:
            self.gau_sigma = sigma

    def _get_input_channel_axis(self):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            raise ValueError('Only `channels_first` supported, i.e., NCHW format.')

        return channel_axis

    def _get_input_channels(self, input_shape):
        channel_axis = self._get_input_channel_axis()

        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        return input_shape[channel_axis].value

    def get_gau_variable_shape(self, input_shape):
        # get input
        num_input_channels = self._get_input_channels(input_shape)

        gau_params_shape_ = (num_input_channels, self.gau_units[0], self.gau_units[1], self.filters)
        gau_params_shape = (1, num_input_channels, self.num_gau_units_all, self.filters)

        return gau_params_shape

    def add_gau_weights_var(self, input_shape):
        gau_params_shape = self.get_gau_variable_shape(input_shape)
        return self.add_variable(name='weights',
                                 shape=gau_params_shape,
                                 initializer=self.weight_initializer,
                                 regularizer=self.weight_regularizer,
                                 constraint=self.weight_constraint,
                                 trainable=True,
                                 dtype=self.dtype)

    def add_gau_mu1_var(self, input_shape):
        gau_params_shape = self.get_gau_variable_shape(input_shape)
        return self.add_variable(name='mu1',
                                 shape=gau_params_shape,
                                 initializer=self.mu1_initializer,
                                 regularizer=self.mu1_regularizer,
                                 constraint=self.mu1_constraint,
                                 trainable=True,
                                 dtype=self.dtype)


    def add_gau_mu2_var(self, input_shape):
        gau_params_shape = self.get_gau_variable_shape(input_shape)
        return self.add_variable(name='mu2',
                                   shape=gau_params_shape,
                                   initializer=self.mu2_initializer,
                                   regularizer=self.mu2_regularizer,
                                   constraint=self.mu2_constraint,
                                   trainable=True,
                                   dtype=self.dtype)
    def add_gau_sigma_var(self, input_shape, trainable=False):
        gau_params_shape = self.get_gau_variable_shape(input_shape)

        # create single sigma variable
        sigma_var = self.add_variable(name='sigma',
                                      shape=(1,),
                                      initializer=self.sigma_initializer,
                                      regularizer=self.sigma_regularizer,
                                      constraint=self.sigma_constraint,
                                      trainable=trainable,
                                      dtype=self.dtype)

        # but make variable shared across all channels as required for the efficient gau implementation
        return tf.tile(tf.reshape(sigma_var,[1,1,1,1]), multiples=gau_params_shape)


    def add_bias_var(self):
        return self.add_variable(name='bias',
                                 shape=(self.filters,),
                                 initializer=self.bias_initializer,
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint,
                                 trainable=True,
                                 dtype=self.dtype)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)

        gau_params_shape = self.get_gau_variable_shape(input_shape)
        if self.gau_weights is None:
            self.gau_weights = self.add_gau_weights_var(input_shape)
        elif np.any(self.gau_weights.shape != gau_params_shape):
            raise ValueError('Shape mismatch for variable `gau_weights`')
        if self.gau_mu1 is None:
            self.gau_mu1 = self.add_gau_mu1_var(input_shape)
        elif np.any(self.gau_mu1.shape != gau_params_shape):
            raise ValueError('Shape mismatch for variable `gau_mu1`')

        if self.gau_mu2 is None:
            self.gau_mu2 = self.add_gau_mu2_var(input_shape)
        elif np.any(self.gau_mu2.shape != gau_params_shape):
            raise ValueError('Shape mismatch for variable `gau_mu2`')
        if self.gau_sigma is None:
            self.gau_sigma = self.add_gau_sigma_var(input_shape, trainable=self.gau_sigma_trainable)
        elif np.any(self.gau_sigma.shape != gau_params_shape):
            raise ValueError('Shape mismatch for variable `gau_sigma`')

        if self.use_bias:
            self.bias = self.add_bias_var()
        else:
            self.bias = None

        input_channel_axis = self._get_input_channel_axis()
        num_input_channels = self._get_input_channels(input_shape)

        self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                         axes={input_channel_axis: num_input_channels})

        self._gau_convolution_op = _gauConvolution2d(
            input_shape,
            num_output=self.filters,
            gau_units=self.gau_units,
            max_kernel_size=self.max_kernel_size,
            padding=self.padding,
            strides=1,
            num_gau_units_ignore=self.num_gau_units_ignore,
            mu_learning_rate_factor=self.mu_learning_rate_factor,
            gau_unit_border_bound=self.gau_unit_border_bound,
            gau_unit_single_dim=self.gau_unit_single_dim,
            gau_aggregation_forbid_positive_dim1=self.gau_aggregation_forbid_positive_dim1,
            gau_mu_interpolation=self.gau_mu_interpolation,
            unit_testing=self.unit_testing,
            data_format=utils.convert_data_format(self.data_format,
                                                  self.rank + 2))
        self.built = True

    def call(self, inputs):
        outputs = self._gau_convolution_op(inputs, self.gau_weights, self.gau_mu1, self.gau_mu2, self.gau_sigma)

        # we emulate strides larger them 1 by simply sampling the output
        if self.strides > 1:
            outputs = outputs[:,:,::self.strides,::self.strides]

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                if self.rank == 2:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
                if self.rank == 3:
                    outputs_shape = outputs.shape.as_list()
                    if outputs_shape[0] is None:
                        outputs_shape[0] = -1
                    outputs_4d = array_ops.reshape(outputs,
                                                   [outputs_shape[0], outputs_shape[1],
                                                    outputs_shape[2] * outputs_shape[3],
                                                    outputs_shape[4]])
                    outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
                    outputs = array_ops.reshape(outputs_4d, outputs_shape)
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.max_kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=1)
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                            [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides,
                    dilation=1)
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                            new_space)

class GAUConv1d(gauConv2d):
    def __init__(self, filters,
                 gau_units,
                 max_kernel_size,
                 **kwargs):
        def mu_zero_constraint(w):
            return tf.zeros_like(w)

        super(gauConv1d, self).__init__(filters, gau_units, max_kernel_size,
                                        mu2_initializer=tf.zeros_initializer(),
                                        mu2_regularizer=None,
                                        mu2_constraint=mu_zero_constraint,
                                        gau_unit_single_dim=True,
                                        **kwargs)



from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import layers as layers_contrib
from tensorflow.contrib.layers.python.layers import utils as utils_contrib

@add_arg_scope
def gau_conv2d(inputs,
             filters,
             gau_units,
             max_kernel_size,
             stride=1,
             mu_learning_rate_factor=500,
             data_format=None,
             activation_fn=nn.relu,
             normalizer_fn=None,
             normalizer_params=None,
             weights_initializer=init_ops.random_normal_initializer(stddev=0.1), #init_ops.glorot_uniform_initializer(),
             weights_regularizer=None,
             weights_constraint=None,
             mu1_initializer=None,
             mu1_regularizer=None,
             mu1_constraint=None,
             mu2_initializer=None,
             mu2_regularizer=None,
             mu2_constraint=None,
             sigma_initializer=None,
             sigma_regularizer=None,
             sigma_constraint=None,
             biases_initializer=init_ops.zeros_initializer(),
             biases_regularizer=None,
             biases_constraint=None,
             gau_unit_border_bound=0.01,
             gau_sigma_trainable=False,
             gau_mu_interpolation=True,
             reuse=None,
             variables_collections=None,
             outputs_collections=None,
             trainable=True,
             scope=None):

    if data_format not in [None, 'NCHW']:
        raise ValueError('Invalid data_format: %r' % (data_format,))

    layer_variable_getter = layers_contrib._build_variable_getter({
        'bias': 'biases',
        'weight': 'weights',
        'mu1': 'mu1',
        'mu2': 'mu2',
        'sigma': 'sigma'
    })

    with variable_scope.variable_scope(
            scope, 'gauConv', [inputs], reuse=reuse,
            custom_getter=layer_variable_getter) as sc:
        inputs = ops.convert_to_tensor(inputs)
        input_rank = inputs.get_shape().ndims

        if input_rank != 4:
            raise ValueError('gau convolution not supported for input with rank',
                             input_rank)

        df = ('channels_first'
              if data_format and data_format.startswith('NC') else 'channels_last')

        layer = gauConv2d(filters,
                          gau_units,
                          max_kernel_size,
                          strides=stride,
                          data_format=df,
                          activation=None,
                          use_bias=not normalizer_fn and biases_initializer,
                          mu_learning_rate_factor=mu_learning_rate_factor,
                          weight_initializer=weights_initializer,
                          mu1_initializer=mu1_initializer,
                          mu2_initializer=mu2_initializer,
                          sigma_initializer=sigma_initializer,
                          bias_initializer=biases_initializer,
                          weight_regularizer=weights_regularizer,
                          mu1_regularizer=mu1_regularizer,
                          mu2_regularizer=mu2_regularizer,
                          sigma_regularizer=sigma_regularizer,
                          bias_regularizer=biases_regularizer,
                          activity_regularizer=None,
                          weight_constraint=weights_constraint,
                          mu1_constraint=mu1_constraint,
                          mu2_constraint=mu2_constraint,
                          sigma_constraint=sigma_constraint,
                          bias_constraint=biases_constraint,
                          gau_unit_border_bound=gau_unit_border_bound,
                          gau_sigma_trainable=gau_sigma_trainable,
                          gau_mu_interpolation=gau_mu_interpolation,
                          trainable=trainable,
                          unit_testing=False,
                          name=sc.name,
                          _scope=sc,
                          _reuse=reuse)

        outputs = layer.apply(inputs)

        # Add variables to collections.
        layers_contrib._add_variable_to_collections(layer.gau_weights, variables_collections, 'weights')
        layers_contrib._add_variable_to_collections(layer.gau_mu1, variables_collections, 'mu1')
        layers_contrib._add_variable_to_collections(layer.gau_mu2, variables_collections, 'mu2')
        layers_contrib._add_variable_to_collections(layer.gau_sigma, variables_collections, 'sigma')

        if layer.use_bias:
            layers_contrib._add_variable_to_collections(layer.bias, variables_collections, 'biases')

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils_contrib.collect_named_outputs(outputs_collections, sc.name, outputs)


@add_arg_scope
def gau_conv1d(inputs,
               filters,
               gau_units,
               max_kernel_size,
               stride=1,
               mu_learning_rate_factor=500,
               data_format=None,
               activation_fn=nn.relu,
               normalizer_fn=None,
               normalizer_params=None,
               weights_initializer=init_ops.random_normal_initializer(stddev=0.1), #init_ops.glorot_uniform_initializer(),
               weights_regularizer=None,
               weights_constraint=None,
               mu1_initializer=None,
               mu1_regularizer=None,
               mu1_constraint=None,
               sigma_initializer=None,
               sigma_regularizer=None,
               sigma_constraint=None,
               biases_initializer=init_ops.zeros_initializer(),
               biases_regularizer=None,
               gau_unit_border_bound=0.01,
               gau_sigma_trainable=False,
               gau_aggregation_forbid_positive_dim1=False,
               gau_mu_interpolation=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               scope=None):

    if data_format not in [None, 'NCHW']:
        raise ValueError('Invalid data_format: %r' % (data_format,))

    layer_variable_getter = layers_contrib._build_variable_getter({
        'bias': 'biases',
        'weight': 'weights',
        'mu1': 'mu1',
        'sigma': 'sigma'
    })

    with variable_scope.variable_scope(
            scope, 'gauConv', [inputs], reuse=reuse,
            custom_getter=layer_variable_getter) as sc:
        inputs = ops.convert_to_tensor(inputs)
        input_rank = inputs.get_shape().ndims

        if input_rank != 4:
            raise ValueError('gau convolution not supported for input with rank',
                             input_rank)

        df = ('channels_first'
              if data_format and data_format.startswith('NC') else 'channels_last')

        layer = gauConv1d(filters,
                          gau_units,
                          max_kernel_size,
                          strides=stride,
                          data_format=df,
                          activation=None,
                          use_bias=not normalizer_fn and biases_initializer,
                          mu_learning_rate_factor=mu_learning_rate_factor,
                          weight_initializer=weights_initializer,
                          mu1_initializer=mu1_initializer,
                          sigma_initializer=sigma_initializer,
                          bias_initializer=biases_initializer,
                          weight_regularizer=weights_regularizer,
                          mu1_regularizer=mu1_regularizer,
                          sigma_regularizer=sigma_regularizer,
                          bias_regularizer=biases_regularizer,
                          activity_regularizer=None,
                          gau_unit_border_bound=gau_unit_border_bound,
                          gau_sigma_trainable=gau_sigma_trainable,
                          gau_aggregation_forbid_positive_dim1=gau_aggregation_forbid_positive_dim1,
                          gau_mu_interpolation=gau_mu_interpolation,
                          trainable=trainable,
                          unit_testing=False,
                          name=sc.name,
                          _scope=sc,
                          _reuse=reuse)

        gau_weights = weights_constraint(layer.add_gau_weights_var(inputs.shape)) if  weights_constraint is not None else None
        gau_mu1 = mu1_constraint(layer.add_gau_mu1_var(inputs.shape)) if  mu1_constraint is not None else None
        gau_sigma = sigma_constraint(layer.add_gau_sigma_var(inputs.shape)) if  sigma_constraint is not None else None

        layer.set_gau_variables_manually(gau_weights, gau_mu1, None, gau_sigma)

        outputs = layer.apply(inputs)

        # Add variables to collections.
        layers_contrib._add_variable_to_collections(layer.gau_weights, variables_collections, 'weights')
        layers_contrib._add_variable_to_collections(layer.gau_mu1, variables_collections, 'mu1')
        layers_contrib._add_variable_to_collections(layer.gau_sigma, variables_collections, 'sigma')

        if layer.use_bias:
            layers_contrib._add_variable_to_collections(layer.bias, variables_collections, 'biases')

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils_contrib.collect_named_outputs(outputs_collections, sc.name, outputs)
