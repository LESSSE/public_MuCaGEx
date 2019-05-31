##LESSSE
##05 December 2018
##MuCaGEx
##____________
##MuCyG Discriminador 
##____________

from collections import OrderedDict
import tensorflow as tf
from copy import deepcopy
from conv import ConvNet
from component import Component
from neuralnet import NeuralNet

class Discriminator(Component):
    """Class that defines the discriminator."""
    def __init__(self, tensor_in, tensor_len, config, condition=None, name='Discriminator',
                 reuse=None):
        super().__init__(tensor_in, condition)
        self.tensor_len = tensor_len
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.tensor_out, self.nets = self.build(config)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self.scope.name)

    def build(self, config):
        """Build the discriminator. 
        Dataset in dis_ds do config"""
        nets = OrderedDict()
        config = deepcopy(config)

        nets['t_input'] = self.tensor_in
        nets['t_seqlen'] = self.tensor_len

        config['conv_ds'] = config['dis_ds']

        nets['conv'] = ConvNet(self.tensor_in,
                               config)

        lstm_cell = tf.nn.rnn_cell.LSTMCell(config['net_d']['rnn_features'], state_is_tuple=True, name="lstm")
        cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell],state_is_tuple=True)
        init_state = cells.zero_state(config['dis_ds']['batch_size'],tf.float32)
        nets['rnn_outputs'], nets['final_state'] = tf.nn.dynamic_rnn(cells,nets['conv'].tensor_out,initial_state=init_state, sequence_length=nets['t_seqlen'])

        nets['full_connected'] = NeuralNet(nets['rnn_outputs'][:, -1,:],
                      config['net_d']['full_connected'],
                      name='full_connected')

        nets['t_output'] = nets['full_connected'].tensor_out

        return nets['t_output'], nets
