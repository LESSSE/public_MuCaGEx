##LESSSE
##1 November 2018
##MuCaGEx
##____________
##Config for Dummy's model 
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
    'id': 'epicdummy',
    'name': 'Dummy',

    'tensorflow': {
        'config': tf.ConfigProto(),
        'gpu': '0',
    },

	'model' : {
        'datasets': {
            'epic': {
                'id': 'epic2018',
                'name': 'Epic',
                'sizes': { 
                    'batch_size' : 12,
                    'num_beat' : 4,
                    'num_bar' : 4,
                    'beat_resolution' : 24,
                    'num_timesteps' : 4*24,
                    'num_pitch' : 128,
                    'num_track' : 8}
            },
        },
        'state': {"dir": os.path.abspath(os.path.join(os.path.dirname(__file__),"state"))}
    },
}

#_________________tensorflow_config_______________
mconfig['tensorflow']['config'].gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = mconfig['tensorflow']['gpu']

def load_model():
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    import Dummy
    sess = tf.Session(config=mconfig['tensorflow']['config'])
    with sess.as_default():
        m = Dummy.model(sess,mconfig['model'])
        m.id = mconfig['id']
        if os.path.isdir(mconfig['model']['state']['dir']):
            m.load_latest()
        return m, m.stats(), None


mconfig['load_model']=load_model