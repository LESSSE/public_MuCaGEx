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
    'id': "epichbrm2018",
    'name': 'HRBM Epic',

    'tensorflow': {
        'config': tf.ConfigProto(
            device_count = {'GPU': 1}
            ),
        'gpu': '',
    },

    'model' : {
        'init_lr': 0.1,
        'decay_steps_lr': 20000,
        'cycles_lr': 10,
        'n_features' : 128,
        'datasets': {
            'epic': {
                'id': 'epic2018',
                'name': 'Epic',
                'new_sample_size' : 4,
                'sizes': { 
                    'batch_size' : 1,
                    'num_timesteps' : 4*4*24,
                    'num_pitch' : 128,
                    'num_track' : 8}
            },

            #'melody': {
            #     'id': 'melody2018',
            #     'name': 'Melody',
            #     'sizes': { 
            #         'batch_size' : 12,
            #         'num_timesteps' : 4*4*24,
            #         'num_pitch' : 128,
            #         'num_track' : 1}
            #},
        },
        'state': {"dir": os.path.abspath(os.path.join(os.path.dirname(__file__),"state"))}
    },
}

#_________________tensorflow_config_______________
mconfig['tensorflow']['config'].gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = mconfig['tensorflow']['gpu']

def load_model():
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    import hrbm
    with tf.Graph().as_default():  
      sess = tf.Session(config=mconfig['tensorflow']['config'])  
      with sess.as_default():
        m = hrbm.HRBM(sess,mconfig['model'])
        m.id = mconfig['id']
        if os.path.isdir(mconfig['model']['state']['dir']):
            m.load_latest()
        return m, m.stats(), None

mconfig['load_model']=load_model
