##LESSSE
##30 October 2018
##MuCaGEx
##____________
##Simple RBM model class
##____________

from collections import OrderedDict
import tensorflow as tf
import sampling as samples

def rbm(name,n_visible, n_hidden, t_input=None,t_seqlen=None):

    with tf.variable_scope(name+'_rbm'):
        if t_input is None:
             t_input = tf.placeholder(tf.float32,
                       [n_visible], name='in_ph')
        if t_seqlen is None:
             t_seqlen = tf.placeholder(tf.float32,[1],name='len_ph')
        nets = OrderedDict()
        nets['t_input'] = t_input
        nets['-v0'] = t_input
        nets['t_seqlen'] = t_seqlen

        nets["W"] = tf.Variable(tf.random_normal([n_visible,n_hidden], 0.01), name="W")
        nets["bv"] = tf.Variable(tf.zeros([1,n_visible], tf.float32, name="bv"))
        nets["bh"] = tf.Variable(tf.zeros([1,n_hidden], tf.float32, name="bh"))
        return nets

def up_nodes(name,nets,level,visible_node=None):
    if visible_node is None:
        visible_node = nets["-v"+str(level-1)]
    with tf.variable_scope(name+'_rbm'):
        nets["v"+str(level)] = tf.identity(visible_node)
        nets["h"+str(level)] = samples.sample_f(tf.sigmoid(tf.matmul(nets["v"+str(level)], nets["W"]) + nets["bh"]))
    return nets["h"+str(level)]

def down_nodes(name,nets,level,hidden_node=None):
    if hidden_node is None:
        hidden_node = nets["h"+str(level)]
    with tf.variable_scope(name+'_rbm'):
        nets["-h"+str(level)] = hidden_node
        nets["-v"+str(level)] = samples.sample_f(tf.sigmoid(tf.matmul(nets["-h"+str(level)], tf.transpose(nets["W"])) + nets["bv"]))
    return nets["-v"+str(level)]
