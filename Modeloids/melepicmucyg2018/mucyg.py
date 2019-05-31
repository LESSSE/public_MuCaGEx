##LESSSE
##06 December 2018
##MuCaGEx
##____________
##MuCyG Model
##____________

import numpy as np
import tensorflow as tf
from model import Model
from generator import Generator
from discriminator import Discriminator
from itertools import permutations
from copy import deepcopy
from collections import OrderedDict
from random import random

class MuCyG(Model):
    """Class that defines the MuCyG Model"""
    def __init__(self, sess, config, name='mucyg', reuse=None):
        super().__init__(sess, config, name)

        self.epochs = 0  #training epochs correspond to the number of times all trainning set has been used 
        self.d_iters = 0
        self.g_iters = 0
        self.iters = 0

        #print('[*] Building GAN...')
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.build()
  
    def build(self):
        """Build the model."""
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.nets = OrderedDict()
        datasets = self.config["datasets"]

        #we supose that we have only 2 datasets, but the solution for more than two is to make this for each pair

        self.G_vars = []
        self.D_vars = []
        self.output[()] = []
        self.losses[()] = []
        self.optimizers[()] = []

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
         
        self.nets["learning_rate"] = lr
        self.lr = lr
        self.losses[()] = [("learning_rate",self.lr)]

        for d in datasets:
            self.nets[datasets[d]['id']] = {}

            self.output[(datasets[d]["id"],)] = []
            self.losses[(datasets[d]["id"],)] = []
            self.optimizers[(datasets[d]["id"],)] = []

            data_shape = (datasets[d]['sizes']['batch_size'],
                              None,
                              datasets[d]['sizes']['num_timesteps'], 
                              datasets[d]['sizes']['num_pitch'],
                              datasets[d]['sizes']['num_track'])
            t_input = tf.placeholder(tf.float32, data_shape, name='in_ph_'+datasets[d]['id']) #input bool
            t_seqlen = tf.placeholder(tf.float32,[datasets[d]['sizes']['batch_size']],name='len_ph_'+datasets[d]['id'])


            self.input[datasets[d]['id']]=(t_input,t_seqlen)

        m = 1
        for pair in permutations(datasets,2):
            self.output[(datasets[pair[0]]["id"],datasets[pair[1]]["id"])] = []
            self.losses[(datasets[pair[0]]["id"],datasets[pair[1]]["id"])] = []
            self.optimizers[(datasets[pair[0]]["id"],datasets[pair[1]]["id"])] = []
            
            x_ = self.input[datasets[pair[0]]['id']][0]

            with tf.device('/gpu:'+str(1+m)):
               config = deepcopy(self.config)
               config["gen_from_ds"] = datasets[pair[0]]["sizes"]
               config["gen_to_ds"]  = datasets[pair[1]]["sizes"]
               self.nets[pair[0]+"_to_"+pair[1]] = Generator(self.input[datasets[pair[0]]['id']][0],self.input[datasets[pair[0]]['id']][1],config, name=pair[0]+"_to_"+pair[1], reuse=tf.AUTO_REUSE)

            #RNN
            with tf.device('/gpu:'+str(1+int(not m))):
               config["gen_from_ds"], config["gen_to_ds"] = config["gen_to_ds"], config["gen_from_ds"]
               self.nets[pair[1]+"_backto_"+pair[0]] = Generator(self.nets[pair[0]+"_to_"+pair[1]].tensor_out,self.nets[pair[0]+"_to_"+pair[1]].tensor_len,config, name=pair[1]+"_to_"+pair[0], reuse=tf.AUTO_REUSE)
           

            self.output[(datasets[pair[0]]["id"],)] += [
                                    (datasets[pair[0]]["id"],"input", self.input[datasets[pair[0]]['id']][0]),
                                    (datasets[pair[1]]["id"],"sample_mapped", self.nets[pair[0]+"_to_"+pair[1]].tensor_out),
                                    (datasets[pair[0]]["id"],"sample_recon", self.nets[pair[1]+"_backto_"+pair[0]].tensor_out),
                            ]

            with tf.device('/gpu:'+str(0)):#1+int(not m))):
               config = deepcopy(self.config)
               config["dis_ds"] = datasets[pair[0]]["sizes"]
               self.nets[pair[0]+"_to_"+pair[1]+"_real"] = Discriminator(self.input[datasets[pair[0]]['id']][0],self.input[datasets[pair[0]]['id']][1], config, name=pair[0]+"_D", reuse=tf.AUTO_REUSE)
               if self.nets.get(pair[0]+"_dis") is None:
                   self.nets[pair[0]+"_dis"] = self.nets[pair[0]+"_to_"+pair[1]+"_real"]

            with tf.device('/gpu:'+str(0)):#1+m)):
               config["dis_ds"] = datasets[pair[1]]["sizes"]
               self.nets[pair[0]+"_to_"+pair[1]+"_fake"] = Discriminator(self.nets[pair[0]+"_to_"+pair[1]].tensor_out,self.nets[pair[0]+"_to_"+pair[1]].tensor_len, config, name=pair[1]+'_D', reuse=tf.AUTO_REUSE)
               if self.nets.get(pair[1]+"_dis") is None:
                   self.nets[pair[1]+"_dis"] = self.nets[pair[0]+"_to_"+pair[1]+"_fake"]

            m = int(not m)

        self.nets["G_loss"] = 0
        self.nets["D_loss"] = 0

        ds = list(datasets)
        if len(ds) != 2:
            raise ValueError('It must be size 2')
        for i in range(len(ds)):
            self.D_real = self.nets[ds[i]+"_to_"+ds[not i]+"_real"]
            self.D_fake = self.nets[ds[not i]+"_to_"+ds[i]+"_fake"]
            self.G = self.nets[ds[not i]+"_to_"+ds[i]]
            self.G_inv = self.nets[ds[i]+"_backto_"+ds[not i]]
            self.G_vars += self.G.vars
            self.D_vars += self.D_fake.vars
            # Losses
            self.config["dis_ds"] = datasets[ds[i]]["sizes"]
            self.x_ = self.input[datasets[ds[i]]['id']][0]
            self.g_loss, self.d_loss = self.get_adversarial_loss(Discriminator,name=ds[i]+"_D") #Loss_(GAN_i) Loss_(D_i) 
            self.cons_loss = self.get_reconstruction_loss() #Loss_(CONS_j)
            self.nets["G_loss"] = self.nets["G_loss"] + self.g_loss + self.cons_loss
            self.nets["D_loss"] = self.nets["D_loss"] + self.d_loss

            self.losses[(datasets[ds[i]]["id"],datasets[ds[not i]]["id"])] += [(ds[i]+"_"+"G",self.g_loss)]
            self.losses[(datasets[ds[i]]["id"],datasets[ds[not i]]["id"])] += [(ds[i]+"_"+"D",self.d_loss)]
            self.losses[(datasets[ds[not i]]["id"],)] += [(ds[not i]+"_CONS",self.cons_loss)]
            self.losses[(datasets[ds[not i]]["id"],)] += [(ds[i]+"_mapped_densitity",tf.log(tf.reduce_sum(self.G.tensor_out)+1))]
            self.losses[(datasets[ds[not i]]["id"],)] += [(ds[not i]+"_recon_densitity",tf.log(tf.reduce_sum(self.G_inv.tensor_out)+1))]
            self.losses[(datasets[ds[i]]["id"],datasets[ds[not i]]["id"])] += [(datasets[ds[i]]["id"]+"_diff",tf.reduce_mean(self.x_) - tf.reduce_mean(self.G.tensor_out))]
        self.losses[(datasets[ds[i]]["id"],datasets[ds[not i]]["id"])] += [("G",self.nets["G_loss"])]
        self.losses[(datasets[ds[i]]["id"],datasets[ds[not i]]["id"])] += [("D",self.nets["D_loss"])]
        
        self.g_loss = self.nets["G_loss"]
        self.d_loss = self.nets["D_loss"]

        # Optimizers
        with tf.variable_scope('Optimizer'):
            self.g_optimizer = self.get_optimizer()
            self.g_step = self.g_optimizer.minimize(
                self.g_loss, self.global_step, self.G_vars)

            self.d_optimizer = self.get_optimizer()
            self.d_step = self.d_optimizer.minimize(
                self.d_loss, self.global_step, self.D_vars)


            # Apply weight clipping
            if self.config['gan']['type'] == 'wgan':
                with tf.control_dependencies([self.d_step]):
                    self.d_step = tf.group(
                        *(tf.assign(var, tf.clip_by_value(
                            var, -self.config['gan']['clip_value'],
                            self.config['gan']['clip_value']))
                          for var in self.D_vars))

            self.optimizers[((datasets[ds[i]]["id"],datasets[ds[not i]]["id"]))] += [(0,self.d_step)]
            self.optimizers[((datasets[ds[i]]["id"],datasets[ds[not i]]["id"]))] += [(1,self.g_step)]

        # Saver
        self.saver = tf.train.Saver()
        self.savers["global"] = self.saver 
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

                for o in filnodes:
                    filnodes[o] = self.sess.run(filnodes[o],feed_dict)

                return filnodes, exceptions

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

                for o in filnodes:
                    filnodes[o] = self.sess.run(filnodes[o],feed_dict)

                sdict = filnodes
                datasets = self.config["datasets"]
                ds = list(datasets.keys())
                for di in range(len(ds)):
                    d = ds[di]
                    id = datasets[d]['id']
                    d2 = ds[not di]
                    id2 = datasets[d2]['id']
                    size = datasets[d]['new_sample_size']
                    feed_dict = {}
                    shape = list(self.input[id2][0].get_shape())
                    shape[1] = tf.Dimension(size)
                    instance = np.random.uniform(size=shape)
                    feed_dict[self.input[id2][0]] = instance
                    feed_dict[self.input[id2][1]] = np.tile(instance.shape[1],(instance.shape[0]))
                    new_sample = self.sess.run(self.nets[d2+"_to_"+d].tensor_out,feed_dict)
                    if sdict.get(id) is None:
                        sdict[id]={}
                    sdict[id]["new_sample"]=new_sample
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


