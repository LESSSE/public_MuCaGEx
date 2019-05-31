##LESSSE
##11 November 2018
##MuCaGEx
##____________
##Main Config
##____________

exp_config = {
                "id" : "exp141118-MuseGAN",
                "name" : "MuseGAN Test", #name
                "doc": "Experiment using MuseGAN model on epic 2018",
                "authors" : "Luis Espirito Santo", #author's names
                "verbose" : True, #more output
                "datasets" : [
                                "../Datasets/epic2018", 
                                #"../Datasets/melody2018",
                             ], #paths to directories that represent datasets (i.e. includes a dsconfig.py file)
                "model" : "../Modeloids/epicmusegan2018", #path to a directory that represent a model (i.e. includes a mconfig.py file)
                "metrics" : [], #functions that will be used to evaluate 
                "epochs": 30, #number of epochs
                "freq": { #functions that control the frequency of each one of the actions during the cycles
                    "training?" : lambda x: True , 
                    "losses?" : lambda x: not (x%10), 
                    "sampling?" : lambda x: not (x%200), 
                    "validating?" : lambda x: False,#not (x%50), 
                    "testing?" : lambda x: False, 
                    "saving?" : lambda x: not (x%200), 
                }
             }
