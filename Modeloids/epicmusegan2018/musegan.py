##LESSSE
##29 October 2018
##MuCaGEx
##____________
##Model adapted from https://github.com/salu133445/musegan/blob/master/v2/musegan/musegan/models.py
##____________

import numpy as np
import tensorflow as tf
from model import Model
from generator import Generator
from discriminator import Discriminator
from collections import OrderedDict
from random import random

class Musegan(Model):
    """Class that defines the first-stage (without refiner) model."""
    def __init__(self, sess, config, name='musegan', reuse=None):
        super().__init__(sess, config, name)

        self.epochs = 0  #training epochs correspond to the number of times all trainning set has been used 
        self.d_iters = 0 #training iterations correspond to the number of weigths updates
        self.g_iters = 0

        self.iters = 0

        #print('[*] Building GAN...')
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.build()

    def build(self):
        """Build the model."""
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        nets = OrderedDict()
        datasets = self.config["datasets"]

        self.global_step = tf.train.get_or_create_global_step()
        """lr = tf.train.noisy_linear_cosine_decay(self.config['init_lr'], 
                            self.global_step,
                            self.config['decay_steps_lr'], 
                            num_periods=self.config['cycles_lr'],
                            initial_variance=0.001,
                            alpha=0.0, 
                            beta=0.1,
                                name="learning_rate")"""

        lr = tf.constant(self.config['init_lr'])
        """lr = tf.multiply(
                tf.multiply(
                    1e-10,
                    tf.pow(
                        10e0,
                        tf.multiply(
                            tf.cast(
                                tf.div(
                                    self.global_step,
                                    10
                                ),
                                tf.float32
                            ),
                            1e-1
                        )
                    )
                ),
                tf.cast(
                    tf.mod(
                        tf.add(
                            self.global_step,
                            1
                        ),
                        2
                    ),
                    tf.float32
                )
            )"""

        nets["learning_rate"] = lr
        self.lr = lr
        self.losses[()] = [("learning_rate",self.lr)]

        for d in datasets:
            nets[datasets[d]['id']] = {}
            # Create placeholders
            self.z = {}
            with tf.variable_scope("t_input_"+datasets[d]['id']):
                if self.config['net_g']['z_dim_shared'] > 0:
                    self.z['shared'] = tf.placeholder(
                        tf.float32, (datasets[d]['sizes']['batch_size'], 
                                     self.config['net_g']['z_dim_shared']), 'z_shared'
                    ) #(12,)
                if self.config['net_g']['z_dim_private'] > 0:
                    self.z['private'] = tf.placeholder(
                        tf.float32, (datasets[d]['sizes']['batch_size'],
                                     self.config['net_g']['z_dim_private'],
                                     datasets[d]['sizes']['num_track']), 'z_private'
                    )
                if self.config['net_g']['z_dim_temporal_shared'] > 0:
                    self.z['temporal_shared'] = tf.placeholder(
                        tf.float32, (datasets[d]['sizes']['batch_size'],
                                     self.config['net_g']['z_dim_temporal_shared']),
                        'z_temporal_shared'
                    )
                if self.config['net_g']['z_dim_temporal_private'] > 0:
                    self.z['temporal_private'] = tf.placeholder(
                        tf.float32, (datasets[d]['sizes']['batch_size'],
                                     self.config['net_g']['z_dim_temporal_private'],
                                     datasets[d]['sizes']['num_track']), 'z_temporal_private'
                    )

                nets[datasets[d]['id']]["z"]=self.z

                data_shape = (datasets[d]['sizes']['batch_size'],
                              None,
                              datasets[d]['sizes']['num_timesteps'], 
                              datasets[d]['sizes']['num_pitch'],
                              datasets[d]['sizes']['num_track'])
                self.x = tf.placeholder(tf.bool, data_shape, name='in_ph_'+datasets[d]['id']) #input bool
                self.x1 = tf.cast(self.x, 
                                                tf.float32, 
                                                name='in_ph_'+datasets[d]['id']+'_'
                                 )
                
                t_seqlen = tf.placeholder(tf.float32,[datasets[d]['sizes']['batch_size']],name='len_ph_'+datasets[d]['id'])

                batch_range = tf.reshape(tf.range(datasets[d]['sizes']['batch_size'], dtype=tf.int32), shape=[datasets[d]['sizes']['batch_size'], 1, 1])
                random = tf.cast(tf.random_uniform([datasets[d]['sizes']['batch_size'], 1, 1])*tf.reshape(t_seqlen,(datasets[d]['sizes']['batch_size'],1,1)),tf.int32)
                #indices = tf.zeros_like(tf.concat([batch_range, random], axis = -1))+5
                #now  batch_size is 1
                indices = tf.zeros_like(tf.concat([batch_range, random], axis = -1))
                self.x_ = tf.reshape(tf.gather_nd(self.x1, indices),(
                                                                       datasets[d]['sizes']['batch_size'],
                                                                       datasets[d]['sizes']['num_bar'],
                                                                       datasets[d]['sizes']['num_timesteps']//datasets[d]['sizes']['num_bar'],
                                                                       datasets[d]['sizes']['num_pitch'],
                                                                       datasets[d]['sizes']['num_track']
                                                                     ))
                                             
                t_input = self.x

                self.input[datasets[d]['id']]=(t_input,t_seqlen)

            # Components
            for i in datasets[d]['sizes']:
                self.config[i] = datasets[d]['sizes'][i]
            self.G = Generator(self.z, self.config, name='G')
            self.test_round50 = self.G.tensor_out > 0.5
            self.test_round10 = self.G.tensor_out > 0.1
            self.test_bernoulli = self.G.tensor_out > tf.random_uniform(self.G.tensor_out.shape)
            nets[datasets[d]['id']]["G"]=self.G
            nets[datasets[d]['id']]["test_round50"]=self.test_round50
            nets[datasets[d]['id']]["test_round10"]=self.test_round10
            nets[datasets[d]['id']]["test_bernoulli"]=self.test_bernoulli
            self.output[()]=[
                                   
                                   (datasets[d]["id"],"sample_raw", self.G.tensor_out),
                                   (datasets[d]["id"],"sample_round50", self.test_round50),
                                   (datasets[d]["id"],"sample_round10", self.test_round10),
                                   (datasets[d]["id"],"sample_bernoulli", self.test_bernoulli),
                            ]
            self.output[(datasets[d]["id"],)] = [(datasets[d]["id"],"input", t_input)]
            self.block_sample = self.G.tensor_out

            self.D_fake = Discriminator(self.G.tensor_out, self.config, name='D')
            self.D_real = Discriminator(self.x_, self.config, name='D', reuse=True)
            self.components = (self.G, self.D_fake)
            nets[datasets[d]['id']]["D_fake"]=self.D_fake
            nets[datasets[d]['id']]["D_real"]=self.D_real

            # Losses
            self.g_loss, self.d_loss = self.get_adversarial_loss(Discriminator)
            self.losses[()] += [(datasets[d]['id']+"_"+"G",self.g_loss)]
            self.losses[(datasets[d]['id'],)] = [("diff",tf.reduce_mean(self.x_) - tf.reduce_mean(self.G.tensor_out))]
            self.losses[(datasets[d]['id'],)] += [(datasets[d]['id']+"_"+"D",self.d_loss)]
            self.losses[(datasets[d]['id'],)] += [("density",tf.log(tf.reduce_sum(self.G.tensor_out)+1))]
        
            # Optimizers
            with tf.variable_scope('Optimizer'):
                self.g_optimizer = self.get_optimizer()
                self.g_step = self.g_optimizer.minimize(
                    self.g_loss, self.global_step, self.G.vars)

                self.d_optimizer = self.get_optimizer()
                self.d_step = self.d_optimizer.minimize(
                    self.d_loss, self.global_step, self.D_fake.vars)

                self.optimizers[(datasets[d]['id'],)] = [(0,self.d_step)]
                self.optimizers[(datasets[d]['id'],)] += [(1,self.g_step)]

                # Apply weight clipping
                if self.config['gan']['type'] == 'wgan':
                    with tf.control_dependencies([self.d_step]):
                        self.d_step = tf.group(
                            *(tf.assign(var, tf.clip_by_value(
                                var, -self.config['gan']['clip_value'],
                                self.config['gan']['clip_value']))
                              for var in self.D_fake.vars))

            # Saver
            self.saver = tf.train.Saver()
            self.savers[datasets[d]['id']] = self.saver
        self.reset_weights = tf.variables_initializer(var_list=tf.trainable_variables())

    def train(self, instance):
        """Train the model."""
        # Prepare instance
        feed_dict = {}
        optimizers = {}
        ds = []
        exceptions = []

        instance_too_big = True

        while instance_too_big:
            try:
                for d in instance:
                    ds.append(d)
                    i = int(random() * (instance[d].shape[0] - 1))
                    instance[d] = instance[d][i:i+1]
                    feed_dict[self.input[d][0]] = instance[d]
                    feed_dict[self.input[d][1]] = np.tile(instance[d].shape[1],(instance[d].shape[0]))
                z_sample = {}
                for key in self.z:
                    z_sample[key] = np.random.normal(size=self.z[key].get_shape())
                    feed_dict[self.z[key]] = z_sample[key]

                if ((self.iters//200) < 25) or ((self.epochs//200) % 100 == 0):
                    num_critics = self.config['gan']['many_critics']
                else:
                    num_critics = self.config['gan']['few_critics']

                allnodes = self.optimizers.copy()
                filnodes = []

                #all the optimizers we want to run in step 0 will run num_critics times
                for e in allnodes:
                    list_ = allnodes[e].copy()
                    for i in allnodes[e]:
                        if i[0] == 0:
                            for j in range(num_critics-1):
                                list_.append(i)
                    allnodes[e] = list_

                #given the input we gonna get the optimizers we may run
                for i in allnodes:
                    if all(x in ds for x in i):
                        l = allnodes[i]
                        for e in l:
                            filnodes += [e]

                filnodes = sorted(filnodes, key=lambda x: x[0])
                filnodes = list(map(lambda x: x[1],filnodes))

                self.d_iters += num_critics
                self.g_iters += 1

                for o in filnodes:
                    self.sess.run(o,feed_dict)
                
                return exceptions

            except tf.errors.ResourceExhaustedError as ree:
                instance_too_big = True
                exceptions += [ree]
                get_size = lambda k: instance[k].shape[1] # not general TODO

                def half_of(x): #not general and not adequated for batch with different sizes TODO
                    r = int(random()*get_size(x)/2)
                    return instance[x][:,r:int(r+get_size(x)//2),...]

                def one_less(x): #not general and not adequated for batch with different sizes TODO
                    r = int(random() > 0.5)
                    return instance[x][:,r:r-1,...]

                k = get_size(max(instance, key = get_size))
                for d in instance: 
                    if get_size(d) >= k:
                        instance[d] = half_of(d)
                    else:
                        instance[d] = one_less(d)


    def loss(self,instance): 
        feed_dict = {}
        losses = {}
        ds = []
        exceptions = []

        instance_too_big = True

        while instance_too_big:
            try:
                #building the feed dict
                for d in instance:
                    ds.append(d)
                    i = int(random() * (instance[d].shape[0] - 1))
                    instance[d] = instance[d][i:i+1]
                    feed_dict[self.input[d][0]] = instance[d]
                    feed_dict[self.input[d][1]] = np.tile(instance[d].shape[1],(instance[d].shape[0]))
                z_sample = {}
                for key in self.z:
                    z_sample[key] = np.random.normal(size=self.z[key].get_shape())
                    feed_dict[self.z[key]] = z_sample[key]

                #given the input we gonna get the losses we may compute
                allnodes = self.losses
                filnodes = losses
                for i in allnodes:
                    if all(x in ds for x in i):
                        l = allnodes[i]
                        for e in l:
                            dic = filnodes
                            for j in e[:-2]:
                                if dic.get(j) is None:
                                    dic[j] = {}
                                dic = dic[j]
                            dic[e[-2]] = e[-1]

                return  self.sess.run(filnodes,feed_dict), exceptions

            except tf.errors.ResourceExhaustedError as ree:
                instance_too_big = True
                exceptions += [ree]
                get_size = lambda k: instance[k].shape[1] # not general TODO

                def half_of(x): #not general and not adequated for batch with different sizes TODO
                    r = int(random()*get_size(x)/2)
                    return instance[x][:,r:int(r+get_size(x)//2),...]

                def one_less(x): #not general and not adequated for batch with different sizes TODO
                    r = int(random() > 0.5)
                    return instance[x][:,r:r-1,...]

                k = get_size(max(instance, key = get_size))
                for d in instance: 
                    if get_size(d) >= k:
                        instance[d] = half_of(d)
                    else:
                        instance[d] = one_less(d)


    def sample(self,instance):
        feed_dict = {}
        samples = {}
        ds = []
        exceptions = []

        instance_too_big = True

        while instance_too_big:
            try:
                for d in instance:
                    ds.append(d)
                    i = int(random() * (instance[d].shape[0] - 1))
                    instance[d] = instance[d][i:i+1]
                    feed_dict[self.input[d][0]] = instance[d]
                    feed_dict[self.input[d][1]] = np.tile(instance[d].shape[1],(instance[d].shape[0]))
                z_sample = {}
                for key in self.z:
                    z_sample[key] = np.random.normal(size=self.z[key].get_shape())
                    feed_dict[self.z[key]] = z_sample[key]

                allnodes = self.output
                filnodes = samples
                for i in allnodes:
                    if all(x in ds for x in i):
                        l = allnodes[i]
                        for e in l:
                            dic = filnodes
                            for j in e[:-2]:
                                if dic.get(j) is None:
                                    dic[j] = {}
                                dic = dic[j]
                            dic[e[-2]] = e[-1]

                sdict = self.sess.run(filnodes,feed_dict)

                datasets = self.config["datasets"]
                for d in datasets:
                    id = datasets[d]['id']
                    new_sample = []
                    for i in range(datasets[d]['new_sample_size']):
                        feed_dict = {}
                        z_sample = {}
                        for key in self.z:
                            z_sample[key] = np.random.normal(size=self.z[key].get_shape())
                            feed_dict[self.z[key]] = z_sample[key]
                        new_sample += [self.sess.run(self.block_sample,feed_dict)] 
                    if sdict.get(id) is None:
                        sdict[id] = {}
                    sdict[id]["new_sample"]=np.concatenate(new_sample,1)
                return sdict, exceptions

            except tf.errors.ResourceExhaustedError as ree:
                instance_too_big = True
                exceptions += [ree]
                get_size = lambda k: instance[k].shape[1] # not general TODO

                def half_of(x): #not general and not adequated for batch with different sizes TODO
                    r = int(random()*get_size(x)/2)
                    return instance[x][:,r:int(r+get_size(x)//2),...]

                def one_less(x): #not general and not adequated for batch with different sizes TODO
                    r = int(random() > 0.5)
                    return instance[x][:,r:r-1,...]

                k = get_size(max(instance, key = get_size))
                for d in instance: 
                    if get_size(d) >= k:
                        instance[d] = half_of(d)
                    else:
                        instance[d] = one_less(d)
