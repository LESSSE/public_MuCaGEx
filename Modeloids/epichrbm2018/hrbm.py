##LESSSE
##4 November 2018
##MuCaGEx
##____________
##HRBM's model class
##____________

from collections import OrderedDict
from random import random
import tensorflow as tf
import rbm
import numpy as np
import os

class HRBM:
    """Restricted Boltzmann Machines Models"""
    def __init__(self, sess, config, name='HRBM', reuse=None):
        self.sess = sess
        self.name = name
        self.config = config

        self.iters = 0

        self.scope = None
        self.global_step = None
        self.input = None
        self.losses = None
        self.output = None
        self.optimizers = None
        self.components = None
        self.savers = None
        self.loaded = False

        '''input = {*dataset_name*: (data,size_of_each_seq)}'''
        self.input = OrderedDict()
        '''losses = {*dataset_name*_*track_number+1*_[W|bv|bh]: losses for weights and bias for each track of each dataset}'''
        self.losses = {}
        '''output = {*dataset_name*: {"input": t_input,"sample": sample}}'''
        self.output = {}
        ''' optimizeters = {*dataset_name*_*track_number+1*: assign nodes to all weights}'''
        self.optimizers = {}
        '''savers = {RBM_*dataset_name*: saver for all variables of hrbm '''
        self.savers = {}
        
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.build()
            self.sess.run(tf.global_variables_initializer())
            
    def build(self):
        """Build the model"""
    
        rbms = {}
        nets = OrderedDict()
        datasets = self.config["datasets"]

        self.global_step = tf.train.get_or_create_global_step()
       
        """lr = tf.train.noisy_linear_cosine_decay(self.config['init_lr'], 
                            self.global_step,
                            self.config['decay_steps_lr'], 
                            num_periods=self.config['cycles_lr'], 
                            alpha=0.0, 
                            beta=0.001,
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
            rbms[datasets[d]['id']] = {}
            nets[datasets[d]['id']+"_h1"] = []
            nets[datasets[d]['id']+"_v2"] = []
            nets[datasets[d]['id']+"_h2"] = []
            self.optimizers[(datasets[d]['id'],)]=[]
            self.losses[(datasets[d]['id'],)] = []
            diff = []

            with tf.variable_scope("t_input_"+datasets[d]['id']):
                t_input = tf.placeholder(tf.float32,
                                [datasets[d]['sizes']['batch_size'],
                                 None,
                                 datasets[d]['sizes']['num_timesteps'],
                                 datasets[d]['sizes']['num_pitch'],
                                 datasets[d]['sizes']['num_track']], name='in_ph_'+datasets[d]['id'])

                t_seqlen = tf.placeholder(tf.float32,[datasets[d]['sizes']['batch_size']],name='len_ph_'+datasets[d]['id'])
                self.input[datasets[d]['id']]=(t_input,t_seqlen)
                t_input = tf.floor(t_input+tf.constant(1-0.1)) #binarize
                nets["t_input"] = tf.transpose(tf.reshape(t_input,
                                        (-1,
                                        datasets[d]['sizes']['num_timesteps']*datasets[d]['sizes']['num_pitch'],
                                        datasets[d]['sizes']['num_track'])),
                                        [2,0,1])

            with tf.variable_scope("RBM_"+datasets[d]['id']):
                for t in range(datasets[d]['sizes']['num_track']):
                    rbms[datasets[d]['id']][t+1] = rbm.rbm(datasets[d]['id']+"_"+str(t),datasets[d]['sizes']['num_timesteps']*datasets[d]['sizes']['num_pitch'],self.config['n_features'],nets["t_input"][t])
                    rbm.up_nodes(datasets[d]['id']+"_"+str(t),rbms[datasets[d]['id']][t+1],1)
                    nets[datasets[d]['id']+"_h1"] += [rbms[datasets[d]['id']][t+1]["h1"]]

                with tf.variable_scope("concat_"+datasets[d]['id']):
                    nets[datasets[d]['id']+"gen-v0"] = tf.concat(nets[datasets[d]['id']+"_h1"],1)

                rbms[datasets[d]['id']][0] = rbm.rbm(datasets[d]['id']+"_gen",datasets[d]['sizes']['num_track']*self.config['n_features'],self.config['n_features'],nets[datasets[d]['id']+"gen-v0"])
                rbm.up_nodes(datasets[d]['id']+"_gen",rbms[datasets[d]['id']][0],1)
                nets[datasets[d]['id']+"gen-v1"] = tf.transpose(tf.reshape(rbm.down_nodes(datasets[d]['id']+"_gen",rbms[datasets[d]['id']][0],1),
                                          (-1,
                                           datasets[d]['sizes']['num_track'],
                                           self.config['n_features'])),
                                           [1,0,2])
                for t in range(datasets[d]['sizes']['num_track']):
                    rbm.down_nodes(datasets[d]['id']+"_"+str(t),rbms[datasets[d]['id']][t+1],1,nets[datasets[d]['id']+"gen-v1"][t])
                    rbm.up_nodes(datasets[d]['id']+"_"+str(t),rbms[datasets[d]['id']][t+1],2)
                    nets[datasets[d]['id']+"_v2"]+= [rbms[datasets[d]['id']][t+1]["v2"]]
                    nets[datasets[d]['id']+"_h2"]+= [rbms[datasets[d]['id']][t+1]["h2"]]
                with tf.variable_scope("concat_"+datasets[d]['id']):
                    nets[datasets[d]['id']+"gen_v2"] = tf.concat(nets[datasets[d]['id']+"_h2"],1)
                rbm.up_nodes(datasets[d]['id']+"_gen",rbms[datasets[d]['id']][0],2,nets[datasets[d]['id']+"gen_v2"])

            for t in range(datasets[d]['sizes']['num_track']+1):
                with tf.variable_scope("losses_"+datasets[d]['id']+"_"+str(t)):
                    if t > 0:
                        v = tf.reshape(rbms[datasets[d]['id']][t]["v1"],[-1,datasets[d]['sizes']['num_timesteps']*datasets[d]['sizes']['num_pitch'],1])
                        v_sample = tf.reshape(rbms[datasets[d]['id']][t]["v2"],[-1,datasets[d]['sizes']['num_timesteps']*datasets[d]['sizes']['num_pitch'],1])
                    else:
                        v = tf.reshape(rbms[datasets[d]['id']][t]["v1"],[-1,datasets[d]['sizes']['num_track']*self.config['n_features'],1])
                        v_sample = tf.reshape(rbms[datasets[d]['id']][t]["v2"],[-1,datasets[d]['sizes']['num_track']*self.config['n_features'],1])
                    h = tf.reshape(rbms[datasets[d]['id']][t]["h1"],[-1,1,self.config['n_features']])
                    h_sample = tf.reshape(rbms[datasets[d]['id']][t]["h2"],[-1,1,self.config['n_features']])
                    W_adder = tf.reduce_mean(tf.subtract(tf.matmul(v, h), tf.matmul(v_sample, h_sample)),0)
                    bv_a = tf.reduce_mean(tf.reduce_sum(tf.subtract(v, v_sample), 2, True),0)
                    if t > 0:
                        bv_adder = tf.reshape(bv_a,[1,datasets[d]['sizes']['num_timesteps']*datasets[d]['sizes']['num_pitch']])
                    else:
                        bv_adder = tf.reshape(bv_a,[1,datasets[d]['sizes']['num_track']*self.config['n_features']])
                    bh_adder = tf.reshape(tf.reduce_mean(tf.reduce_sum(tf.subtract(h,h_sample), 1, True),0),[1,self.config['n_features']])

                    if t > 0:
                        diff += [bv_adder]
                    self.losses[(datasets[d]['id'],)] +=[(datasets[d]['id']+"_"+str(t)+"_W" , tf.reduce_mean(W_adder))]
                    self.losses[(datasets[d]['id'],)] +=[(datasets[d]['id']+"_"+str(t)+"_bv",tf.reduce_mean(bv_adder))]
                    self.losses[(datasets[d]['id'],)] +=[(datasets[d]['id']+"_"+str(t)+"_bh",tf.reduce_mean(bh_adder))]

                    with tf.variable_scope("optimizers_"+datasets[d]['id']):
                        self.optimizers[(datasets[d]['id'],)] += [(0,rbms[datasets[d]['id']][t]["W"].assign_add(tf.multiply(self.lr,W_adder))),
                                (0,rbms[datasets[d]['id']][t]["bv"].assign_add(tf.multiply(self.lr,bv_adder))),
                                (0,rbms[datasets[d]['id']][t]["bh"].assign_add(tf.multiply(self.lr,bh_adder)))]

            self.losses[(datasets[d]['id'],)] +=[(datasets[d]['id']+"_diff" , tf.reduce_mean(tf.concat(diff,0)))]

            with tf.variable_scope("sample_"+datasets[d]['id']):
                sample = tf.concat(nets[datasets[d]['id']+"_v2"],1)
                sample = tf.reshape(sample,
                                    [datasets[d]['sizes']['batch_size'],
                                     -1,
                                     datasets[d]['sizes']['num_track'],
                                     datasets[d]['sizes']['num_timesteps'],
                                     datasets[d]['sizes']['num_pitch']])
                sample = tf.transpose(sample, [0,1,3,4,2])
                nets["sample"] = sample
                self.output[(datasets[d]['id'],)] = [
                                   (datasets[d]['id'],"input", t_input),
                                   (datasets[d]['id'],"train_sample", sample)
                                  ]

            self.savers["RBM_"+datasets[d]['id']] = tf.train.Saver(tf.get_collection(
                                    tf.GraphKeys.GLOBAL_VARIABLES,
                                     tf.get_default_graph().get_name_scope()+"/RBM_"+datasets[d]['id']), max_to_keep=2)

        self.components = nets
        self.reset_weights = tf.variables_initializer(var_list=tf.trainable_variables())

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
                    i = int(random() * (instance[d].shape[0] - 1))
                    instance[d] = instance[d][i:i+1]
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

                losses = self.sess.run(filnodes,feed_dict)
                return (losses, exceptions)
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

                #building the feed dict
                for d in instance:
                    ds.append(d)
                    i = int(random() * (instance[d].shape[0] - 1))
                    instance[d] = instance[d][i:i+1]
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

                
                sdict = self.sess.run(filnodes,feed_dict)
                
                datasets = self.config["datasets"]
                for d in datasets:
                    id = datasets[d]['id']
                    out = np.zeros((datasets[d]['sizes']['batch_size'],
                                 datasets[d]['new_sample_size'],
                                 datasets[d]['sizes']['num_timesteps'],
                                 datasets[d]['sizes']['num_pitch'],
                                 datasets[d]['sizes']['num_track']))
                    feed_dict[self.input[id][0]] = out
                    feed_dict[self.input[id][1]] = np.tile(out.shape[1],(out.shape[0]))
                    for i in range(15):
                        out = self.sess.run(self.components["sample"],feed_dict)
                        feed_dict[self.input[id][0]] = out
                        feed_dict[self.input[id][1]] = np.tile(out.shape[1],(out.shape[0]))
                    if sdict.get(id) is None:
                        sdict[id] = {}
                    sdict[id]["new_sample"]=out


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

    def stats(self):
        """Return model statistics (number of paramaters for each component)."""
        def get_num_parameter(var_list):
            """Given the variable list, return the total number of parameters.
            """
            return int(np.sum([np.product([x.value for x in var.get_shape()])
                               for var in var_list]))
        num_par = get_num_parameter(tf.trainable_variables(
            self.scope.name))
        return {"global": num_par,  "loaded": self.loaded} 

    def reset(self):
        self.sess.run(self.reset_weights)

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
            if states is not None:
                self.savers[name].restore(self.sess, states.model_checkpoint_path)
                self.iters=int(states.model_checkpoint_path.split('-')[-1])
                self.loaded=self.iters
            
