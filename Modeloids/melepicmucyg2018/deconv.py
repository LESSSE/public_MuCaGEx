##LESSSE
##04 December 2018
##MuCaGEx
##____________
##Deconvolutional Network
##____________

from collections import OrderedDict
import tensorflow as tf
from math import ceil
from component import Component
from neuralnet import NeuralNet

class DeconvNet(Component):
    """Class that defines the generator."""
    def __init__(self, tensor_in, config, condition=None, name='deconv',
                 reuse=None):
        super().__init__(tensor_in, condition)
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.tensor_out, self.nets = self.build(config)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self.scope.name)

    def build(self, config):
        """Build the recurrent convolutional net."""
        nets = OrderedDict()

        nets['t_input'] = self.tensor_in #(12,-1,512)

        nets['reshape_t_input'] = tf.reshape(nets['t_input'],
                                    (-1,
                                     1,
                                     1,
                                     1,
                                     nets['t_input'].shape[-1]
                                    )

            )
 
        nets['bar_main'] = NeuralNet(
            nets['reshape_t_input'], config['net_g']['bar_main'],
            name='bar_main'
        )

        nets['bar_pitch_time'] = NeuralNet(
            nets['bar_main'].tensor_out, config['net_g']['bar_pitch_time'],
            name='bar_pitch_time'
        )

        nets['bar_time_pitch'] = NeuralNet(
            nets['bar_main'].tensor_out, config['net_g']['bar_time_pitch'],
            name='bar_time_pitch'
        )

        config_bar_merged = config['net_g']['bar_merged'].copy()

        if config_bar_merged[-1][1][0] is None:
            l = list(config_bar_merged[-1])
            l[1] = list(l[1])
            l[1][0] = config['deconv_ds']['num_track']
            l[1] = tuple(l[1])
            config_bar_merged[-1] = tuple(l)

        nets['bar_merged'] = NeuralNet(
            tf.concat([nets['bar_pitch_time'].tensor_out,
                       nets['bar_time_pitch'].tensor_out], -1),
            config_bar_merged, name='bar_merged'
        )

        nets['t_output'] = nets['bar_merged'].tensor_out[...,:config['deconv_ds']['num_pitch'],:]

        nets['reshape_t_output'] = tf.reshape(nets['t_output'],
                                    (
                                     config['deconv_ds']["batch_size"],
                                     -1,
                                     nets['t_output'].shape[-3]*nets['t_output'].shape[-4],
                                     nets['t_output'].shape[-2],
                                     nets['t_output'].shape[-1]
                                    )
        )

        tensor_out = nets['reshape_t_output']



        return tensor_out, nets
