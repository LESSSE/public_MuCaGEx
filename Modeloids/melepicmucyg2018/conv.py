##LESSSE
##04 December 2018
##MuCaGEx
##____________
##Convolutional Network
##____________

from collections import OrderedDict
import tensorflow as tf
from math import ceil
from component import Component
from neuralnet import NeuralNet

class ConvNet(Component):
    def __init__(self, tensor_in, config, condition=None, name='conv',
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
        nets['t_input'] = self.tensor_in

        octaves = ceil(int(self.tensor_in.shape[-2])/12)
        nets['tensor_in_padded'] = tf.pad(self.tensor_in,((0,0),
                            (0,0),
                            (0,0),
                            (0,octaves*12-int(self.tensor_in.shape[-2])),
                            (0,0)))

        nets['tensor_in_reshape'] = tf.reshape(nets['tensor_in_padded'],
                               (-1,
                                config['conv_ds']['num_bar'],
                                config['conv_ds']['num_beat']*config['conv_ds']['beat_resolution'],
                                octaves*12,
                                config['conv_ds']['num_track'])
        )

        nets['pitch_time_private'] = [
            NeuralNet(tf.expand_dims(nets['tensor_in_reshape'][..., idx], -1), #tf.expand_dims(self.tensor_in[..., idx], -1) -> selects track *idx* but keeps the same number of dimensions
                      config['net_conv']['pitch_time_private'],
                      name='pt_' + str(idx))
            for idx in range(config['conv_ds']['num_track'])
        ]
       
        nets['time_pitch_private'] = [
            NeuralNet(tf.expand_dims(nets['tensor_in_reshape'][..., idx], -1),
                      config['net_conv']['time_pitch_private'],
                      name='tp_' + str(idx))
            for idx in range(config['conv_ds']['num_track'])
        ]

        nets['merged_private'] = [
            NeuralNet(
                tf.concat([x.tensor_out,
                           nets['time_pitch_private'][idx].tensor_out], -1),
                config['net_conv']['merged_private'], name='merged_' + str(idx))
            for idx, x in enumerate(nets['pitch_time_private'])
        ] # final shape: (?,4,4*4,11,64)
        #print("merged_out:",nets['merged_private'][0].tensor_out.shape)

        nets['shared'] = NeuralNet(
            tf.concat([l.tensor_out for l in nets['merged_private']], -1),
            config['net_conv']['shared'], name='shared'
        )

        # Chroma stream
        reshaped = tf.reshape(
            nets['tensor_in_reshape'], (-1, config['conv_ds']['num_bar'], config['conv_ds']['num_beat'],
                             config['conv_ds']['beat_resolution'], octaves,
                             12, config['conv_ds']['num_track'])
        )

        self.chroma = tf.reduce_sum(reshaped, axis=(3, 4))
        nets['chroma'] = NeuralNet(self.chroma, config['net_conv']['chroma'],
                                   name='chroma')

        # Onset stream
        padded = tf.pad(nets['tensor_in_reshape'][:, :, :-1, :, :],
                        [[0, 0], [0, 0], [1, 0], [0, 0], [0, 0]])
        self.onset = tf.concat([nets['tensor_in_reshape'][..., :] - padded], -1)
        nets['onset'] = NeuralNet(self.onset, config['net_conv']['onset'],
                                  name='onset')

        if (config['net_conv']['chroma'] is not None
                or config['net_conv']['onset'] is not None):
            to_concat = [nets['shared'].tensor_out]
            if config['net_conv']['chroma'] is not None:
                to_concat.append(nets['chroma'].tensor_out)
            if config['net_conv']['onset'] is not None:
                to_concat.append(nets['onset'].tensor_out)
            concated = tf.concat(to_concat, -1)
        else:
            concated = nets['shared'].tensor_out

        # Merge streams
        nets['merged'] = NeuralNet(concated, config['net_conv']['merged'],
                                   name='merged')

        nets['merged_reshaped'] = tf.reshape(nets['merged'].tensor_out,
                               (config['conv_ds']['batch_size'],
                                -1,
                                nets['merged'].tensor_out.shape[-1])
        )

        return nets['merged_reshaped'], nets
