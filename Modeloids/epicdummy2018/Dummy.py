##LESSSE
##4 November 2018
##MuCaGEx
##____________
##Dummy's class
##____________

from collections import OrderedDict
from random import random
import tensorflow as tf
import numpy as np
import os


class model:

    def __init__(self, sess, config, name="Dummy"):
        self.sess = sess
        self.name = name
        self.config = config

        self.iters = 0 #external counter for external update

        self.scope = None 
        self.global_step = None
        self.input = None #dictionary that matches the dataset 'id' with the pair of placeholders of it
        self.losses = None #dictionary that saves the loss nodes in a key that describes it 
        self.output = None #dictionary that saves a dict of nodes for each sample for each dataset id 
        self.optimizers = None #dictionary that matches a list of optimizers with the instances needed 
        self.components = None
        self.savers = None #dictionary of savers for each component of the network
        self.loaded = False

        '''input = {*dataset_name*: (data,size_of_each_seq)}'''
        self.input = OrderedDict()
        '''losses = {*dataset_name*_*track_number+1*_[W|bv|bh]: losses for weights and bias for each track of each dataset}'''
        self.losses = OrderedDict()
        '''output = {*dataset_name*: {"input": t_input,"sample": sample}}'''
        self.output = OrderedDict()
        ''' optimizers = {*dataset_name*_*track_number+1*: assign nodes to all weights}'''
        self.optimizers = {}
        '''savers = {RBM_*dataset_name*: saver for all variables of hrbm '''
        self.savers = {}

        with tf.variable_scope(name, reuse=False) as scope:
            self.scope = scope
            self.build()
            self.sess.run([self.counter.initializer,self.sumer.initializer])
        

    def build(self):
        datasets = self.config["datasets"]      
        for d in datasets:
            data_shape = (datasets[d]['sizes']['batch_size'],
                            datasets[d]['sizes']['num_bar'],
                            datasets[d]['sizes']['num_timesteps'], 
                            datasets[d]['sizes']['num_pitch'],
                            datasets[d]['sizes']['num_track'])
            self.x = tf.placeholder(tf.bool, data_shape, name='in_ph_'+datasets[d]['id']) #input bool
            self.x_ = tf.cast(self.x, tf.float32, name='in_ph_'+datasets[d]['id']+'_')
            t_input = self.x
            t_seqlen = tf.placeholder(tf.float32,[datasets[d]['sizes']['batch_size']],name='len_ph_'+datasets[d]['name'])

            self.input[datasets[d]['id']]=(t_input,t_seqlen)

        self.counter = tf.Variable(0, dtype=tf.float32, name="counter")
        self.sumer = tf.Variable(0,dtype=tf.float32, name="sumer")
        self.r = tf.reduce_sum(self.x_)
        self.s = self.sumer
        self.d = tf.divide(self.sumer,self.counter)

        self.a_step = (self.sumer.assign_add(self.r),self.counter.assign_add(1))

        self.sam = 30+tf.floormod(self.d,70)
        
        self.output[(datasets[d]['id'],)] = [(datasets[d]['id'],"input",self.x_), 
                                            (datasets[d]['id'],"sample",self.sam)]

        self.losses[(datasets[d]['id'],)] = [("DescribeLoss",tf.abs(self.d-tf.reduce_sum(self.x_)))]

        self.optimizers[(datasets[d]['id'],)] = [(0,self.a_step)]

        self.savers = {"counter": tf.train.Saver([self.counter], max_to_keep=2),
                        "sumer": tf.train.Saver([self.sumer], max_to_keep=2)}


    def train(self,instance):
        feed_dict = {}
        optimizers = {}
        ds = []
        exceptions = []

        instance_too_big = True

        while instance_too_big:
            try:

                #building the feed dict
                for d in instance: 
                    ds.append(d)
                    feed_dict[self.input[d][0]] = instance[d]
                    feed_dict[self.input[d][1]] = np.tile(instance[d].shape[1],(instance[d].shape[0]))

                #given the input we gonna get the optimizers we may run
                allnodes = self.optimizers
                filnodes = []
                for i in allnodes:
                    if all(x in ds for x in i):
                        l = allnodes[i]
                        for e in l:
                            filnodes += [e]

                filnodes = sorted(filnodes, key=lambda x: x[0])
                filnodes = list(map(lambda x: x[1],filnodes))

                for o in filnodes:
                    self.sess.run(o,feed_dict)

                return exceptions

            except tf.errors.ResourceExhaustedError as ree:
                instance_too_big = True
                exceptions += [ree]
                get_size = lambda k: instance[k].shape[1] # not general TODO

                def half_of(x): #not general and not adequated for batch with different sizes TODO
                    r = int(random()*get_size/2)
                    return x[:,r:r+get_size//2,...]

                def one_less(x): #not general and not adequated for batch with different sizes TODO
                    r = int(random() > 0.5)
                    return x[:,r:r-1,...]

                k = get_size(max(instance, key = get_size))
                for d in instance: 
                    if get_size(instance[d]) >= k:
                        instance[d] = half_of(instance[d])
                    else:
                        instance[d] = one_less(instance[d])

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

                return self.sess.run(filnodes,feed_dict), exceptions

            except tf.errors.ResourceExhaustedError as ree:
                instance_too_big = True
                exceptions += [ree]
                get_size = lambda k: instance[k].shape[1] # not general TODO

                def half_of(x): #not general and not adequated for batch with different sizes TODO
                    r = int(random()*get_size/2)
                    return x[:,r:r+get_size//2,...]

                def one_less(x): #not general and not adequated for batch with different sizes TODO
                    r = int(random() > 0.5)
                    return x[:,r:r-1,...]

                k = get_size(max(instance, key = get_size))
                for d in instance: 
                    if get_size(instance[d]) >= k:
                        instance[d] = half_of(instance[d])
                    else:
                        instance[d] = one_less(instance[d])


    def sample(self,instance):
        feed_dict = {}
        samples = {}
        ds = []
        exceptions = []

        instance_too_big = True

        while instance_too_big:
            try:

                #building the feed dict
                for d in instance:
                    ds.append(d)
                    feed_dict[self.input[d][0]] = instance[d]
                    feed_dict[self.input[d][1]] = np.tile(instance[d].shape[1],(instance[d].shape[0]))

                #given the input we gonna get the losses we may run
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

                #bulding a dummy sample
                sdict = self.sess.run(filnodes,feed_dict)
                for d in sdict:
                    sam = np.zeros_like(sdict[d]["input"])
                    indices = [7+int(sdict[d]["sample"])%3,3+int(sdict[d]["sample"])%2,0,10+(int(sdict[d]["sample"])%7)%2,12+(int(sdict[d]["sample"])%8)%3,7,0,0]
                    for i in range(len(indices)):
                        sam[:,0,:-1,int(sdict[d]["sample"]) + indices[i],i].fill(0.9)
                    sdict[d]["sample"] = sam

                return sdict, exceptions

            except tf.errors.ResourceExhaustedError as ree:
                instance_too_big = True
                exceptions += [ree]
                get_size = lambda k: instance[k].shape[1] # not general TODO

                def half_of(x): #not general and not adequated for batch with different sizes TODO
                    r = int(random()*get_size/2)
                    return x[:,r:r+get_size//2,...]

                def one_less(x): #not general and not adequated for batch with different sizes TODO
                    r = int(random() > 0.5)
                    return x[:,r:r-1,...]

                k = get_size(max(instance, key = get_size))
                for d in instance: 
                    if get_size(instance[d]) >= k:
                        instance[d] = half_of(instance[d])
                    else:
                        instance[d] = one_less(instance[d])

    def stats(self):
        """Return model statistics (number of paramaters for each component)."""
        def get_num_parameter(var_list):
            """Given the variable list, return the total number of parameters.
            """
            return int(np.sum([np.product([x.value for x in var.get_shape()])
                               for var in var_list]))
        num_par = get_num_parameter(tf.trainable_variables(
            self.scope.name))
        #num_par_g = get_num_parameter(self.G.vars)
        #num_par_d = get_num_parameter(self.D_fake.vars)
        #return ("Number of parameters: {}\nNumber of parameters in G: {}\n"
        #        "Number of parameters in D: {}".format(num_par, num_par_g,
        #                                               num_par_d))     
        return {"global": num_par, "loaded": self.loaded}

    def save(self):
        exceptions = []

        path = self.config["state"]["dir"]
        for name in self.savers:
            os.makedirs(os.path.join(path,name),exist_ok=True)
            self.savers[name].save(self.sess,os.path.join(path,name,"saver"),global_step=self.iters)

        return exceptions

    def load_latest(self):
        path = self.config["state"]["dir"]
        for name in self.savers:
            states = tf.train.get_checkpoint_state(os.path.join(path,name))
            self.savers[name].restore(self.sess, states.model_checkpoint_path)
            self.iters=int(states.model_checkpoint_path.split('-')[-1])
            self.loaded=self.iters
