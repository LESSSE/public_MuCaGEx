# MuCaGEx

This project intends to implement a general purpose main cycle for the training, generation, validation and testing of generative models, but applied in this case to the symbolic musical generation task.

## Datasets

It's represented in a file system by a directory with a dsconfig.py file and all extra needed files

The class `dataset` have : 
* id: a unique id string
* name: name
* authors: authors
* doc: description
* preprocessing: pre-processing parameters
* split: list of identifiers for each instance from each part of the split: train, valid and test. (this identifier may be a path if each instance has its own file or the rows in a table for example)
* metrics: functions that were used to study the dataset and the correspondent results
* load_instance: function that loads an intance or list of instances given its identifiers or list of identifiers already in the representation that should be injected to a modeloid during the train 
* save_instance: function which given one instance it can save presistetly and give it a new identifier

### Epic Music

Dataset of symbolically represented orchestral epic music midi collection of epic orchestral pieces based in some kind of creative commons license collected from [epic orchestral group from musescore community](https://musescore.com/groups/epicorchestralmusic)

Reference for each one of the instances in References.xlsx

All information on this data should only be used for machine learning studies. Any use of this information for creative, artÃ­stic or commercial purposes should be informed to the original creators of the data.

In this dataset:
*  one song is a sequence of blocks
*  transposition correponds to one copy translated in pitch axis
* one instance is a set of transpositions of the same music
* Then, the structure will update its state taken into account on instance at a time, thus it is not stochastic gradient descent (SGD) nor Mini-batch gradient descent. To the best of my knowledge there is no name for this approach and so I'll name it Stochastic-Augmented Gradient Descent (SAGD) and it is a method of optimizing one structure by taken into account a batch of examples that were all obtained by augmenting the same original example in each step. 

### Melody

Dataset of symbolically represented monodic melodies. Collection generated using [Melisma Stochastic Melody Generator](http://www.link.cs.cmu.edu/melody-generator/)

Params used to generate each one of the instances in files params_x_y.txt

All information on this dataset should only be used for machine learning studies. Any use of this information for creative, artistic or commercial purposes should be reported to the creators of the software that created the melodies.

In this dataset we used the same scheme and terminology used for epic music dataset.

## Models and Modeloids

Models vs Modeloids:
* a modeloid has a fixed static architecture, a state and it is related with some specific datasets in a specific representation and it is a concretization of a model
* a model is a generalization of modeloids (usually inclues in its definition a way to create modeloids given some datasets)

In a file system a modeloid is a directory that has a mconfig.py and all modeloid's auxiliar files.

In this project I do not have a `modeloid` class but every class implementing a modeloid must include:
* id: a unique id string
* name: name do modeloide
* authors: the authors of the modeloid
* doc: a description of the modeloid
* model: modeloid specific parameter (tensorflow, learning rates, architecture ...)
* load_model: a function that is responsible for creating the python object that implements the modeloid with the right interface and in the state it should be

* train()
* loss()
* sample()
* save()

### Dummy

Just for testing all the rest of the components of the project

### HRBM

It is one RBM based model with two layers 

### MuseGAN

In 2017 musegan used GAN's do generate symbolic music, https://salu133445.github.io/musegan/

#### MuCyGAN

GAN's have been used to generate images since it's introduction in 2014. 

Some models such as CycleGAN (2017) and DiscoGAN (2017) have been proposed for style transfer tasks in visual field, where GAN's are improved with one cycle consistency or reconstruction loss.

To the best of our knowledge these models have not been applied to Symbolic Music Data.

In 2017, one usage of these models have been used in audio musical data represented as spectograms, http://gauthamzz.com/2017/09/23/AudioStyleTransfer/

This model presents an approach of symbolic music generation combining:
* DCGAN
* cycle consistency loss
* LSTMs

Aiming to improve the state of the art on symbolic music neural generative models in the following aspects:
* Music Dynamics
* Temporal relations modelling
* Instrumentation
* Introducing a mechanism of inspiration **\[Not style transfer!!\]**

#### Attention MuCyGAN

**Future Work**

## Other modules

#### Report

Report module allows to save all relevant data collected from the main cycle in the desired formats.

---

## Creativity Concerns

This project may be framed within the area of Computational Creativity. So I leave some topics that may:

* Randomness, overfitting and learning power
* The role of explainability on creativity process
* Defining what is dificult to define
* Meta operations 
* Emerging creative numeric artists in network: what is the role of that society has in creativity evaluation in real cases and what should it be in CC cases    
 

####
