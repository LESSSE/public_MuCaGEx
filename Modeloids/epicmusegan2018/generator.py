##LESSSE
##29 October 2018
##MuCaGEx
##____________
##Generator adapted from https://github.com/salu133445/musegan/blob/master/v2/musegan/musegan/components.py
##____________

from collections import OrderedDict
import tensorflow as tf
from component import Component
from neuralnet import NeuralNet

class Generator(Component):
    """Class that defines the generator."""
    def __init__(self, tensor_in, config, condition=None, name='Generator',
                 reuse=None):
        super().__init__(tensor_in, condition)
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.tensor_out, self.nets = self.build(config)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self.scope.name)

    def build(self, config):
        """Build the generator."""
        nets = OrderedDict()

        # Tile shared latent vector along time axis
        if 'shared' in self.tensor_in:
            tiled_shared = tf.reshape(
                tf.tile(self.tensor_in['shared'], (1, 4)),
                (-1, 4, self.tensor_in['shared'].get_shape()[1])
            )

        # Define shared temporal generator
        if 'temporal_shared' in self.tensor_in:
            nets['temporal_shared'] = NeuralNet(
                self.tensor_in['temporal_shared'],
                config['net_g']['temporal_shared'], name='temporal_shared'
            )

        # Shared bar generator mode
        if config['net_g']['bar_generator_type'] == 'shared':
            if ('private' in self.tensor_in
                    or 'temporal_private' in self.tensor_in):
                raise ValueError("Private latent vectors received for a shared"
                                 "bar generator")

            # Get the final input for the bar generator
            z_input = tf.concat([tiled_shared,
                                 nets['temporal_shared'].tensor_out], -1)

            nets['bar_main'] = NeuralNet(z_input, config['net_g']['bar_main'],
                                         name='bar_main')

            nets['bar_pitch_time'] = NeuralNet(
                nets['bar_main'].tensor_out, config['net_g']['bar_pitch_time'],
                name='bar_pitch_time'
            )

            nets['bar_time_pitch'] = NeuralNet(
                nets['bar_main'].tensor_out, config['net_g']['bar_time_pitch'],
                name='bar_time_pitch'
            )

            if config['net_g']['bar_merged'][-1][1][0] is None:
                config['net_g']['bar_merged'][-1][1][0] = config['num_track']

            nets['bar_merged'] = NeuralNet(
                tf.concat([nets['bar_pitch_time'].tensor_out,
                           nets['bar_time_pitch'].tensor_out], -1),
                config['net_g']['bar_merged'], name='bar_merged'
            )

            tensor_out = nets['bar_merged'].tensor_out

        # Private bar generator mode
        elif config['net_g']['bar_generator_type'] == 'private':
            # Tile private latent vector along time axis
            if 'private' in self.tensor_in:
                tiled_private = [
                    tf.reshape(
                        tf.tile(self.tensor_in['private'][..., idx], (1, 4)),
                        (-1, 4, self.tensor_in['private'].get_shape()[1])
                    )
                    for idx in range(config['num_track'])
                ]

            # Define private temporal generator
            if 'temporal_private' in self.tensor_in:
                nets['temporal_private'] = [
                    NeuralNet(self.tensor_in['temporal_private'][..., idx],
                              config['net_g']['temporal_private'],
                              name='temporal_private_'+str(idx))
                    for idx in range(config['num_track'])
                ]

            # Get the final input for each bar generator
            z_input = []
            for idx in range(config['num_track']):
                to_concat = []
                if config['net_g']['z_dim_shared'] > 0:
                    to_concat.append(tiled_shared)
                if config['net_g']['z_dim_private'] > 0:
                    to_concat.append(tiled_private[idx])
                if config['net_g']['z_dim_temporal_shared'] > 0:
                    to_concat.append(nets['temporal_shared'].tensor_out)
                if config['net_g']['z_dim_temporal_private'] > 0:
                    to_concat.append(nets['temporal_private'][idx].tensor_out)
                z_input.append(tf.concat(to_concat, -1))

            # Bar generators
            nets['bar_main'] = [
                NeuralNet(z_input[idx], config['net_g']['bar_main'],
                          name='bar_main_'+str(idx))
                for idx in range(config['num_track'])
            ]

            nets['bar_pitch_time'] = [
                NeuralNet(nets['bar_main'][idx].tensor_out,
                          config['net_g']['bar_pitch_time'],
                          name='bar_pitch_time_'+str(idx))
                for idx in range(config['num_track'])
            ]

            nets['bar_time_pitch'] = [
                NeuralNet(nets['bar_main'][idx].tensor_out,
                          config['net_g']['bar_time_pitch'],
                          name='bar_time_pitch_'+str(idx))
                for idx in range(config['num_track'])
            ]

            nets['bar_merged'] = [
                NeuralNet(
                    tf.concat([nets['bar_pitch_time'][idx].tensor_out,
                               nets['bar_time_pitch'][idx].tensor_out], -1),
                    config['net_g']['bar_merged'], name='bar_merged_'+str(idx)
                )
                for idx in range(config['num_track'])
            ]

            tensor_out = tf.concat(
                [l.tensor_out for l in nets['bar_merged']], -1)

            tensor_out = tensor_out[...,:config['num_pitch'],:]

        return tensor_out, nets