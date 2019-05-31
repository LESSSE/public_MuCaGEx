##LESSSE
##29 October 2018
##MuCaGEx
##____________
##Model adapted from https://github.com/salu133445/musegan/blob/master/v2/musegan/model.py
##____________

import os.path
import numpy as np
import tensorflow as tf
from collections import OrderedDict

class Model(object):
    """Base class for models."""
    def __init__(self, sess, config, name='model'):
        self.sess = sess
        self.name = name
        self.config = config

        self.scope = None
        self.global_step = None
        self.x_ = None
        self.G = None
        self.D_real = None
        self.D_fake = None
        self.components = []
        self.metrics = None
        self.saver = None
        self.loaded = False 
        self.lr = self.config["init_lr"]

        '''input = {*dataset_name*: (data,size_of_each_seq)}'''
        self.input = OrderedDict()
        '''losses = {*dataset_name*_*track_number+1*_[W|bv|bh]: losses for weights and bias for each track of each dataset}'''
        self.losses = OrderedDict()
        '''output = {*dataset_name*: {"input": t_input,"sample_round": sample with above .5, "sample_bernoulli": with bernoulli}}'''
        self.output = OrderedDict()
        ''' optimizeters = {*dataset_name*_*track_number+1*: assign nodes to all weights}'''
        self.optimizers = {}
        '''savers = {RBM_*dataset_name*: saver for all variables of hrbm '''
        self.savers = {}

    def init_all(self):
        """Initialize all variables in the scope."""
        print('[*] Initializing variables...')
        #self.sess.run(tf.global_variables_initializer())
        tf.variables_initializer(tf.global_variables(self.scope.name)).run()

    def get_adversarial_loss(self, discriminator, scope_to_reuse=None):
        """Return the adversarial losses for the generator and the
        discriminator."""
        if self.config['gan']['type'] == 'gan':
            d_loss_real = tf.losses.sigmoid_cross_entropy(
                tf.ones_like(self.D_real.tensor_out), self.D_real.tensor_out)
            d_loss_fake = tf.losses.sigmoid_cross_entropy(
                tf.zeros_like(self.D_fake.tensor_out), self.D_fake.tensor_out)

            adv_loss_d = d_loss_real + d_loss_fake
            adv_loss_g = tf.losses.sigmoid_cross_entropy(
                tf.ones_like(self.D_fake.tensor_out), self.D_fake.tensor_out)

        if (self.config['gan']['type'] == 'wgan'
                or self.config['gan']['type'] == 'wgan-gp'):
            adv_loss_d = (tf.reduce_mean(self.D_fake.tensor_out)
                          - tf.reduce_mean(self.D_real.tensor_out))
            adv_loss_g = -tf.reduce_mean(self.D_fake.tensor_out)

            if self.config['gan']['type'] == 'wgan-gp':
                eps = tf.random_uniform(
                    [tf.shape(self.x_)[0], 1, 1, 1, 1], 0.0, 1.0)
                inter = eps * self.x_ + (1. - eps) * self.G.tensor_out
                if scope_to_reuse is None:
                    D_inter = discriminator(inter, self.config, name='D',
                                            reuse=True)
                else:
                    with tf.variable_scope(scope_to_reuse, reuse=True):
                        D_inter = discriminator(inter, self.config, name='D',
                                                reuse=True)
                gradient = tf.gradients(D_inter.tensor_out, inter)[0]
                slopes = tf.sqrt(1e-8 + tf.reduce_sum(
                    tf.square(gradient),
                    tf.range(1, len(gradient.get_shape()))))
                gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
                adv_loss_d += (self.config['gan']['gp_coefficient']
                               * gradient_penalty)

        return adv_loss_g, adv_loss_d

    def get_optimizer(self):
        """Return a Adam optimizer."""
        return tf.train.AdamOptimizer(
            self.lr,
            self.config['optimizer']['beta1'],
            self.config['optimizer']['beta2'],
            self.config['optimizer']['epsilon'])

    def stats(self):
        """Return model statistics (number of paramaters for each component)."""
        def get_num_parameter(var_list):
            """Given the variable list, return the total number of parameters.
            """
            return int(np.sum([np.product([x.value for x in var.get_shape()])
                               for var in var_list]))
        num_par = get_num_parameter(tf.trainable_variables(
            self.scope.name))
        num_par_g = get_num_parameter(self.G.vars)
        num_par_d = get_num_parameter(self.D_fake.vars)
        return {"global": num_par, "gen":  num_par_g, "dis": num_par_d, "loaded": self.loaded }

    def get_global_step(self):
        global_step = tf.train.get_or_create_global_step(self.sess.graph)
        if tf.train.global_step(self.sess,global_step) != self.iters:
            self.sess.run(global_step.assign(self.iters))
        return global_step

    def add_step(self):
        """Add one to global step and other important vars"""
        global_step = self.get_global_step()
        self.iters+=1
        self.sess.run(global_step.assign(global_step+1))

    def get_summary(self):
        """Return model summary."""
        return '\n'.join(
            ["{:-^80}".format(' < ' + self.scope.name + ' > ')]
            + [(x.get_summary() + '\n' + '-' * 80) for x in self.components])

    def get_global_step_str(self):
        """Return the global step as a string."""
        return str(tf.train.global_step(self.sess, self.global_step))

    def print_statistics(self):
        """Print model statistics (number of paramaters for each component)."""
        print("{:=^80}".format(' Model Statistics '))
        print(self.get_statistics())

    def print_summary(self):
        """Print model summary."""
        print("{:=^80}".format(' Model Summary '))
        print(self.get_summary())

    def save_statistics(self, filepath=None):
        """Save model statistics to file. Default to save to the log directory
        given as a global variable."""
        if filepath is None:
            filepath = os.path.join(self.config['log_dir'],
                                    'model_statistics.txt')
        with open(filepath, 'w') as f:
            f.write(self.get_statistics())

    def save_summary(self, filepath=None):
        """Save model summary to file. Default to save to the log directory
        given as a global variable."""
        if filepath is None:
            filepath = os.path.join(self.config['log_dir'], 'model_summary.txt')
        with open(filepath, 'w') as f:
            f.write(self.get_summary())

    def reset(self):
        self.sess.run(self.reset_weights)

    def save(self, filepath=None):
        """Save the model to a checkpoint file. Default to save to the log
        directory given as a global variable."""
        exceptions = []

        path = self.config["state"]["dir"]
        for name in self.savers:
            os.makedirs(os.path.join(path,name),exist_ok=True)
            self.savers[name].save(self.sess,os.path.join(path,name,"saver"),global_step=self.iters)
            
        return exceptions

    def load(self, filepath):
        """Load the model from the latest checkpoint in a directory."""
        print('[*] Loading checkpoint...')
        self.saver.restore(self.sess, filepath)

    def load_latest(self):
        """Load the model from the latest checkpoint in a directory."""
        path = self.config["state"]["dir"]
        for name in self.savers:
            states = tf.train.get_checkpoint_state(os.path.join(path,name))
            if states is not None:
                self.savers[name].restore(self.sess, states.model_checkpoint_path)
                self.iters=int(states.model_checkpoint_path.split('-')[-1])
                self.loaded=self.iters

    def save_samples(self, filename, samples, save_midi=False, shape=None,
                     postfix=None):
        """Save samples to an image file (and a MIDI file)."""
        if shape is None:
            shape = self.config['sample_grid']
        if len(samples) > self.config['num_sample']:
            samples = samples[:self.config['num_sample']]
        if postfix is None:
            imagepath = os.path.join(self.config['sample_dir'],
                                     '{}.png'.format(filename))
        else:
            imagepath = os.path.join(self.config['sample_dir'],
                                     '{}_{}.png'.format(filename, postfix))
        #image_io.save_image(imagepath, samples, shape)
        if save_midi:
            binarized = (samples > 0)
            midipath = os.path.join(self.config['sample_dir'],
                                    '{}.mid'.format(filename))
            #midi_io.save_midi(midipath, binarized, self.config)

    def run_sampler(self, targets, feed_dict, save_midi=False, postfix=None):
        """Run the target operation with feed_dict and save the samples."""
        if not isinstance(targets, list):
            targets = [targets]
        results = self.sess.run(targets, feed_dict)
        results = [result[:self.config['num_sample']] for result in results]
        samples = np.stack(results, 1).reshape((-1,) + results[0].shape[1:])
        shape = [self.config['sample_grid'][0],
                 self.config['sample_grid'][1] * len(results)]
        if postfix is None:
            filename = self.get_global_step_str()
        else:
            filename = self.get_global_step_str() + '_' + postfix
        self.save_samples(filename, samples, save_midi, shape)

    def run_eval(self, target, feed_dict, postfix=None):
        """Run evaluation."""
        result = self.sess.run(target, feed_dict)
        binarized = (result > 0)
        if postfix is None:
            filename = self.get_global_step_str()
        else:
            filename = self.get_global_step_str() + '_' + postfix
        reshaped = binarized.reshape((-1,) + binarized.shape[2:])
        mat_path = os.path.join(self.config['eval_dir'], filename+'.npy')
        _ = self.metrics.eval(reshaped, mat_path=mat_path)
