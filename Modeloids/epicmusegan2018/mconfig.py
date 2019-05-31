##LESSSE
##30 October 2018
##MuCaGEx
##____________
##Config for MuseGAN's model 
##____________

"""_______________________________________________
Each model directory must:
- include one mconfig.py where it is implemented one dictionary whose one of the entries is load_model() which should return the model loaded
- control its own building, loading and state update processes
- object returned for the load_model() method must implement the following methods:
    *model.train(instance)
    *model.loss(instance)
    *model.sample(instance)
    *model.validate(instance)
    *model.test(instance)
    *model.save()
"""

import os
import sys
import tensorflow as tf

mconfig = {
    'id': 'epicmusegan2018',
    'name' : 'MuseGAN Hybrid Epic Music',

    'tensorflow': {
            'config': tf.ConfigProto(),
            'gpu': '1'
    },

	'model' : {
        'init_lr': 2.51e-4,
        'decay_steps_lr': 20000,
        'cycles_lr': 10,

		'gan': {
        	'type': 'wgan-gp', # 'gan', 'wgan', 'wgan-gp'
        	'clip_value': .01,
        	'gp_coefficient': 10., 
            'many_critics': 1,
            'few_critics': 1,
    	},
    	'optimizer': {
        	# Parameters for the Adam optimizers
        	'beta1': .5,
        	'beta2': .9,
        	'epsilon': 1e-8
    	},
    	'datasets': {
    		'epic': {
                'id': 'epic2018',
                'name': 'Epic',
                'new_sample_size' : 4,
                'sizes': { 
                    'batch_size' : 1,
                    'num_beat' : 4,
                    'num_bar' : 4,
                    'beat_resolution' : 24,
                    'num_timesteps' : 4*4*24,
                    'num_pitch' : 128,
                    'num_track' : 8}
            },
	    },
        'state': {"dir": os.path.join(os.path.dirname(os.path.realpath(__file__)),"state")}
    }


}

#___________GENERATOR_______________
NET_G = {}

# Input latent sizes (NOTE: use 0 instead of None)
NET_G['z_dim_shared'] = 32
NET_G['z_dim_private'] = 32
NET_G['z_dim_temporal_shared'] = 32
NET_G['z_dim_temporal_private'] = 32
NET_G['z_dim'] = (NET_G['z_dim_shared'] + NET_G['z_dim_private']
                  + NET_G['z_dim_temporal_shared']
                  + NET_G['z_dim_temporal_private'])

# Temporal generators
NET_G['temporal_shared'] = [
    ('dense', (3*256), 'bn', 'lrelu'),
    ('reshape', (3, 1, 1, 256)),                                 # 1 (3, 1, 1)
    ('transconv3d', (NET_G['z_dim_shared'],                      # 2 (4, 1, 1)
                     (2, 1, 1), (1, 1, 1)), 'bn', 'lrelu'),
    ('reshape', (4, NET_G['z_dim_shared'])),
]

NET_G['temporal_private'] = [
    ('dense', (3*256), 'bn', 'lrelu'),
    ('reshape', (3, 1, 1, 256)),                                 # 1 (3, 1, 1)
    ('transconv3d', (NET_G['z_dim_private'],                     # 2 (4, 1, 1)
                     (2, 1, 1), (1, 1, 1)), 'bn', 'lrelu'),
    ('reshape', (4, NET_G['z_dim_private'])),
]

# Bar generator
NET_G['bar_generator_type'] = 'private'

NET_G['bar_main'] = [
    ('reshape', (4, 1, 1, NET_G['z_dim'])),
    ('transconv3d', (512, (1, 4, 1), (1, 4, 1)), 'bn', 'lrelu'), # 1 (4, 4, 1)
    ('transconv3d', (256, (1, 1, 3), (1, 1, 3)), 'bn', 'lrelu'), # 2 (4, 4, 3)
    ('transconv3d', (128, (1, 4, 1), (1, 4, 1)), 'bn', 'lrelu'), # 3 (4, 16, 3)
    ('transconv3d', (64, (1, 1, 5), (1, 1, 3)), 'bn', 'lrelu'),  # 4 (4, 16, 11)
]

NET_G['bar_pitch_time'] = [
    ('transconv3d', (32, (1, 1, 12), (1, 1, 12)), 'bn', 'lrelu'),# 0 (4, 16, 84)
    ('transconv3d', (16, (1, 6, 1), (1, 6, 1)), 'bn', 'lrelu'),  # 1 (4, 96, 84)
]

NET_G['bar_time_pitch'] = [
    ('transconv3d', (32, (1, 6, 1), (1, 6, 1)), 'bn', 'lrelu'),  # 0 (4, 96, 11)
    ('transconv3d', (16, (1, 1, 12), (1, 1, 12)), 'bn', 'lrelu'),# 1 (4, 96, 84)
]

NET_G['bar_merged'] = [
    ('transconv3d', (1, (1, 1, 1), (1, 1, 1)), 'bn', 'sigmoid'),
]

mconfig["model"]["net_g"] = NET_G

#___________________DISCRIMINATOR_____________________
"""Network architecture of the proposed discriminator
"""
NET_D = {}

NET_D['pitch_time_private'] = [
    ('conv3d', (32, (1, 1, 12), (1, 1, 12)), None, 'lrelu'),    # 0 (4, 96, 11)
    ('conv3d', (64, (1, 6, 1), (1, 6, 1)), None, 'lrelu'),      # 1 (4, 16, 11)
]

NET_D['time_pitch_private'] = [
    ('conv3d', (32, (1, 6, 1), (1, 6, 1)), None, 'lrelu'),      # 0 (4, 16, 132)
    ('conv3d', (64, (1, 1, 12), (1, 1, 12)), None, 'lrelu'),    # 1 (4, 16, 11)
]

NET_D['merged_private'] = [
    ('conv3d', (64, (1, 1, 1), (1, 1, 1)), None, 'lrelu'),      # 0 (4, 16, 11)
]

NET_D['shared'] = [
    ('conv3d', (128, (1, 4, 3), (1, 4, 2)), None, 'lrelu'),     # 0 (4, 4, 5)
    ('conv3d', (256, (1, 4, 5), (1, 4, 5)), None, 'lrelu'),     # 1 (4, 1, 1)
]

NET_D['onset'] = [
    ('sum', (3), True),                                         # 0 (4, 96, 1)
    ('conv3d', (32, (1, 6, 1), (1, 6, 1)), None, 'lrelu'),      # 1 (4, 16, 1)
    ('conv3d', (64, (1, 4, 1), (1, 4, 1)), None, 'lrelu'),      # 2 (4, 4, 1)
    ('conv3d', (128, (1, 4, 1), (1, 4, 1)), None, 'lrelu'),     # 3 (4, 1, 1)
]

NET_D['chroma'] = [
    ('conv3d', (64, (1, 1, 12), (1, 1, 12)), None, 'lrelu'),    # 0 (4, 4, 1)
    ('conv3d', (128, (1, 4, 1), (1, 4, 1)), None, 'lrelu'),     # 1 (4, 1, 1)
]

NET_D['merged'] = [
    ('conv3d', (512, (2, 1, 1), (1, 1, 1)), None, 'lrelu'),     # 0 (3, 1, 1)
    ('reshape', (3*512)),
    ('dense', 1),
]

mconfig["model"]["net_d"] = NET_D

#_________________tensorflow_config_______________
mconfig['tensorflow']['config'].gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = mconfig['tensorflow']['gpu']

def load_model():
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    import musegan
    sess = tf.Session(config=mconfig['tensorflow']['config'])
    with sess.as_default():
        m = musegan.Musegan(sess,mconfig["model"])
        m.id = mconfig['id']
        m.init_all()
        if os.path.isdir(mconfig['model']['state']['dir']):
            m.load_latest()
        return m, m.stats(), None

mconfig["load_model"]=load_model
