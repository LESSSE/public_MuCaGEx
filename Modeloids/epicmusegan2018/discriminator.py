##LESSSE
##30 October 2018
##MuCaGEx
##____________
##Discriminator adapted from https://github.com/salu133445/musegan/blob/master/v2/musegan/musegan/components.py
##____________

from collections import OrderedDict
import tensorflow as tf
from math import ceil
from component import Component
from neuralnet import NeuralNet

class Discriminator(Component):
    """Class that defines the discriminator."""
    def __init__(self, tensor_in, config, condition=None, name='Discriminator',
                 reuse=None):
        super().__init__(tensor_in, condition)
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.tensor_out, self.nets = self.build(config)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self.scope.name)

    def build(self, config):
        """Build the discriminator."""
        nets = OrderedDict()

        #Adjust the size of pitch to a multiple of 12 - (?,4,4*24,132,8)
        octaves = ceil(int(self.tensor_in.shape[-2])/12) #ceil(config['num_pitch']/12)
        self.tensor_in = tf.pad(self.tensor_in,((0,0),
                                    (0,0),
                                    (0,0),
                                    (0,octaves*12-int(self.tensor_in.shape[-2])),
                                    (0,0)))
        #print("Tensor_in:",self.tensor_in.shape)

        # Main stream
        nets['pitch_time_private'] = [
            NeuralNet(tf.expand_dims(self.tensor_in[..., idx], -1), #tf.expand_dims(self.tensor_in[..., idx], -1) -> selects track *idx* but keeps the same number of dimensions
                      config['net_d']['pitch_time_private'],
                      name='pt_' + str(idx))
            for idx in range(config['num_track'])
        ] # final shape: (?,4,4*4,11,64)
        #print("Pitch_time_out:",nets['pitch_time_private'][0].tensor_out.shape)

        nets['time_pitch_private'] = [
            NeuralNet(tf.expand_dims(self.tensor_in[..., idx], -1),
                      config['net_d']['time_pitch_private'],
                      name='tp_' + str(idx))
            for idx in range(config['num_track'])
        ] # final shape: (?,4,4*4,11,64)
        #print("time_pitch_out:",nets['time_pitch_private'][0].tensor_out.shape)

        nets['merged_private'] = [
            NeuralNet(
                tf.concat([x.tensor_out,
                           nets['time_pitch_private'][idx].tensor_out], -1),
                config['net_d']['merged_private'], name='merged_' + str(idx))
            for idx, x in enumerate(nets['pitch_time_private'])
        ] # final shape: (?,4,4*4,11,64)
        #print("merged_out:",nets['merged_private'][0].tensor_out.shape)

        nets['shared'] = NeuralNet(
            tf.concat([l.tensor_out for l in nets['merged_private']], -1),
            config['net_d']['shared'], name='shared'
        )

        # Chroma stream
        reshaped = tf.reshape(
            self.tensor_in, (-1, config['num_bar'], config['num_beat'],
                             config['beat_resolution'], octaves,
                             12, config['num_track'])
        )

        self.chroma = tf.reduce_sum(reshaped, axis=(3, 4))
        nets['chroma'] = NeuralNet(self.chroma, config['net_d']['chroma'],
                                   name='chroma')

        # Onset stream
        padded = tf.pad(self.tensor_in[:, :, :-1, :, :],
                        [[0, 0], [0, 0], [1, 0], [0, 0], [0, 0]])
        self.onset = tf.concat([self.tensor_in[..., :] - padded], -1)
        nets['onset'] = NeuralNet(self.onset, config['net_d']['onset'],
                                  name='onset')

        """
        padded = tf.pad(self.tensor_in[:, :, :-1, :, 1:],
                        [[0, 0], [0, 0], [1, 0], [0, 0], [0, 0]])
        self.onset = tf.concat([tf.expand_dims(self.tensor_in[..., 0], -1),
                                self.tensor_in[..., 1:] - padded], -1)
        nets['onset'] = NeuralNet(self.onset, config['net_d']['onset'],
                                  name='onset')
        """

        if (config['net_d']['chroma'] is not None
                or config['net_d']['onset'] is not None):
            to_concat = [nets['shared'].tensor_out]
            if config['net_d']['chroma'] is not None:
                to_concat.append(nets['chroma'].tensor_out)
            if config['net_d']['onset'] is not None:
                to_concat.append(nets['onset'].tensor_out)
            concated = tf.concat(to_concat, -1)
        else:
            concated = nets['shared'].tensor_out

        # Merge streams
        nets['merged'] = NeuralNet(concated, config['net_d']['merged'],
                                   name='merged')

        return nets['merged'].tensor_out, nets
