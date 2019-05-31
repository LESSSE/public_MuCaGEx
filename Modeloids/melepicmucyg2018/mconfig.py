##LESSSE
##4 November 2018
##MuCaGEx
##____________
##Config for MRBM's model 
##____________

"""_______________________________________________
Each model directory must:
- include one mconfig.py where it is implemented one dictionary whose one of the entries is load_model() which should return the model loaded
- control its own creation, loading and state update
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
    'id': "melepicmuscyg2018",
    'name': 'MuCyG Epic and Melody',

    'tensorflow': {
        'config': tf.ConfigProto(allow_soft_placement=True),
        'gpu': '2,3',
    },

	'model' : {
        'init_lr': 5e-4,
        'decay_steps_lr': 20000,
        'cycles_lr': 10,
        'cycle': {'type': "mse"},
        'gan': {
            'type': 'wgan', # 'gan', 'wgan', 'wgan-gp'
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
                'new_sample_size': 4,
                'sizes': { 
                    'batch_size' : 1,
                    'num_beat' : 4,
                    'num_bar' : 4,
                    'beat_resolution' : 24,
                    'num_timesteps' : 4*4*24,
                    'num_pitch' : 128,
                    'num_track' : 8}
            },

            'melody': {
                 'id': 'melody2018',
                 'name': 'Melody',
                 'new_sample_size': 4,
                 'sizes': { 
                    'batch_size' : 1,
                    'num_beat' : 4,
                    'num_bar' : 4,
                    'beat_resolution' : 24,
                    'num_timesteps' : 4*4*24,
                    'num_pitch' : 128,
                    'num_track' : 1}
            },
        },
        'state': {"dir": os.path.abspath(os.path.join(os.path.dirname(__file__),"state"))}
    },
}

#___________GENERATOR_______________
NET_G = {}

NET_G['bar_main'] = [
    ('transconv3d', (512, (1, 4, 1), (1, 4, 1)), 'bn', 'lrelu'), # 1 (1, 4, 1)
    ('transconv3d', (256, (1, 1, 3), (1, 1, 3)), 'bn', 'lrelu'), # 2 (1, 4, 3)
    ('transconv3d', (128, (1, 4, 1), (1, 4, 1)), 'bn', 'lrelu'), # 3 (1, 16, 3)
    ('transconv3d', (64, (1, 1, 5), (1, 1, 3)), 'bn', 'lrelu'),  # 4 (1, 16, 11)
]

NET_G['bar_pitch_time'] = [
    ('transconv3d', (32, (1, 1, 12), (1, 1, 12)), 'bn', 'lrelu'),# 0 (1, 16, 132)
    ('transconv3d', (16, (1, 6, 1), (1, 6, 1)), 'bn', 'lrelu'),  # 1 (1, 96, 132)
]

NET_G['bar_time_pitch'] = [
    ('transconv3d', (32, (1, 6, 1), (1, 6, 1)), 'bn', 'lrelu'),  # 0 (1, 96, 11)
    ('transconv3d', (16, (1, 1, 12), (1, 1, 12)), 'bn', 'lrelu'),# 1 (1, 96, 132)
]

NET_G['bar_merged'] = [
    ('transconv3d', (16, (4, 1, 1), (4, 1, 1)), 'bn', 'lrelu'),# 1 (4, 96, 132)
    ('transconv3d', (None, (1, 1, 1), (1, 1, 1)), 'bn', 'sigmoid'),
]

mconfig["model"]["net_g"] = NET_G

#___________________DISCRIMINATOR_____________________
"""Network architecture of the proposed discriminator
"""
NET_D = {}

NET_D['rnn_features'] = 512

NET_D['full_connected'] = [
    ('reshape', (512)),
    ('dense', 1),
]

mconfig["model"]["net_d"] = NET_D

#___________________CONV_____________________
"""Network architecture of the proposed convolutional
"""
NET_CONV = {}

NET_CONV['pitch_time_private'] = [
    ('conv3d', (32, (1, 1, 12), (1, 1, 12)), None, 'lrelu'),    # 0 (4, 96, 11)
    ('conv3d', (64, (1, 6, 1), (1, 6, 1)), None, 'lrelu'),      # 1 (4, 16, 11)
]

NET_CONV['time_pitch_private'] = [
    ('conv3d', (32, (1, 6, 1), (1, 6, 1)), None, 'lrelu'),      # 0 (4, 16, 132)
    ('conv3d', (64, (1, 1, 12), (1, 1, 12)), None, 'lrelu'),    # 1 (4, 16, 11)
]

NET_CONV['merged_private'] = [
    ('conv3d', (64, (1, 1, 1), (1, 1, 1)), None, 'lrelu'),      # 0 (4, 16, 11)
]

NET_CONV['shared'] = [
    ('conv3d', (128, (1, 4, 3), (1, 4, 2)), None, 'lrelu'),     # 0 (4, 4, 5)
    ('conv3d', (256, (1, 4, 5), (1, 4, 5)), None, 'lrelu'),     # 1 (4, 1, 1)
]

NET_CONV['onset'] = [
    ('sum', (3), True),                                         # 0 (4, 96, 1)
    ('conv3d', (32, (1, 6, 1), (1, 6, 1)), None, 'lrelu'),      # 1 (4, 16, 1)
    ('conv3d', (64, (1, 4, 1), (1, 4, 1)), None, 'lrelu'),      # 2 (4, 4, 1)
    ('conv3d', (128, (1, 4, 1), (1, 4, 1)), None, 'lrelu'),     # 3 (4, 1, 1)
]

NET_CONV['chroma'] = [
    ('conv3d', (64, (1, 1, 12), (1, 1, 12)), None, 'lrelu'),    # 0 (4, 4, 1)
    ('conv3d', (128, (1, 4, 1), (1, 4, 1)), None, 'lrelu'),     # 1 (4, 1, 1)
]

NET_CONV['merged'] = [
    ('conv3d', (512, (4, 1, 1), (4, 1, 1)), None, 'lrelu'),     # 0 (1, 1, 1)
]

mconfig["model"]["net_conv"] = NET_CONV

#_________________tensorflow_config_______________
mconfig['tensorflow']['config'].gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = mconfig['tensorflow']['gpu']

def load_model():
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    import mucyg
    sess = tf.Session(config=mconfig['tensorflow']['config'])
    with sess.as_default():
        m = mucyg.MuCyG(sess,mconfig['model'])
        m.id = mconfig['id']
        m.init_all()
        if os.path.isdir(mconfig['model']['state']['dir']):
            m.load_latest()
        return m, m.stats(), None

mconfig['load_model']=load_model
