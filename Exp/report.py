##LESSSE
##8 November 2018
##MuCaGEx
##____________
##Report Responsible Code Methods
##____________

## Methods that are responsible for saving in a folder the experiment important elements, namely:
## - Logs
## - Error
## - Losses
## - Samples
## - Metrics
## - Validating and test results
## - Saving the model final state

import os
import shutil
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load import get_dsconfig, get_mconfig
from datetime import datetime

rconfig=None
path="./exp"
authors=""
datasets_configs=""
model_config=""

verbose = False
start_time = 0
last_time = 0
loss_points = {}
loss_dataframe = pd.DataFrame()
metrics = {}

log = True
losses = True
samples = True
src = True

def copytree(src, dst, symlinks=False, ignore=None):
	for item in os.listdir(src):
		s = os.path.join(src, item)
		d = os.path.join(dst, item)
		if os.path.isdir(s):
			shutil.copytree(s, d, symlinks, ignore)
		else:
			shutil.copy2(s, d)

#__________INITIAL CONFIG______________
def config(config_):
	global path 
	global authors
	global model_config
	global datasets_configs
	global rconfig
	global loss_point
	loss_points = {}
	rconfig=config_
	path=rconfig["id"]
	authors=rconfig["authors"]
	datasets_configs=list(map(lambda x: get_dsconfig(x),rconfig["datasets"]))
	model_config=get_mconfig(rconfig["model"])
	verbose=rconfig["verbose"]
	create_dirs()
	start()
	new_epoch(0)
	readme()

#_______________START_AND_EPOCHS_______________
def start():
	global start_time
	global last_time
	global epoch_time
	global labels_str
	start_time=datetime.now()
	epoch_time=datetime.now()
	last_time=start_time

def new_epoch(n):
	global epoch_time
	if n > 0:
		print("\n__________________________________".ljust(55,'_'))
		print("Epoch$ Epoch Runtime:{0}\n".format(datetime.now()-epoch_time),flush=True)
	sys.stdout= open(path+"/log/"+str(n).rjust(4,"0")+".log",'w') 
	sys.stderr= open(path+"/log/"+str(n).rjust(4,"0")+".err",'w')
	epoch_time=datetime.now()
	out(("______Epoch_"+str(n)+"__________").ljust(55,'_'))

def create_dirs():
	"""Creates directories for experiment files"""
	if os.path.isdir(path):
		shutil.rmtree(path, ignore_errors=True)
	os.makedirs(path+"/log",exist_ok=True)
	os.makedirs(path+"/losses",exist_ok=True) 
	os.makedirs(path+"/samples",exist_ok=True)
	os.makedirs(path+"/model",exist_ok=True)
	os.makedirs(path+"/datasets",exist_ok=True)
	shutil.copy2("config.py", path+"/config.py")
	for i in rconfig["datasets"]:
		dsconfig = get_dsconfig(i)
		os.makedirs(path+"/datasets/"+dsconfig["id"],exist_ok=True)
		shutil.copy2(i+"/dsconfig.py", path+"/datasets/"+dsconfig["id"]+"/dsconfig.py")
		copytree(dsconfig["split"], path+"/datasets/"+dsconfig["id"]+"/split")

#_____________README_________________________
def readme():
	string = ""
	string += "# README\n"
	string += "## Experiment id: "+rconfig["id"]+"\n"
	string += "## Experiment Name: "+rconfig["name"]+"\n"
	string += "## Authors: "+authors+"\n"
	string += "## Start Date: "+str(datetime.now())+"\n"
	string += "## Datasets: "+datasets_configs[0]["name"]
	for d in datasets_configs[1:]:
		string += ", "+d["name"]
	string += "\n"
	string += "## Model: "+model_config["name"]+"\n"
	string += "## Description: "+rconfig["doc"]+"\n"
	string += "\nIn this directory we have the results and set up information for this experiment in some files described bellow:\n - README : this file which aim is to explain all the other files included in this directory\n - config.py : the configuration file for this experiment\n"


	string +=  " - log:\n\t - <n>.log : log file corresponding to epoch <n>\n"	
	string += " - losses:\n\t - losses.xlsx : an excel file representing a table with all loss values\n\t - *.png :  loss evolution graph\n"	
	string += " - samples:\n\t - all generated samples will be saved in this directory according to their dataset saving method or in a pickled version if it is not possible to identify the corresponding dataset. The filenames represent the corresponding iteration when the sample was saved. More information about the meaning of each file may be found in dataset's and model's config files\n"
	string += " - datasets:\n\t - here we may find the dsconfig files for each one of the datasets used for a deeper understanding of the experiment results\n"
	string += " - model:\n\t - in this directory it will be placed some directories that will represent modeloids in diferent trainning states (directory name is the iteration when it was saved) for reuse in other experiments\n"

	with open(path+"/README",'w') as f:
		f.write(string)

#______EXCEPTION________
def exception(excep):
	out("Exception$ {0}".format(excep.__class__.__name__))
	err("Exception$ {0}".format(str(excep)))


#______MODEL______
def dataset(dataset,stats=None,excep=None):
	if excep is not None:
		for e in excep:
			exception(e)

	lines = ("Dataset ".ljust(10)+"$ ").ljust(13)+dataset+"\n"

	if stats is not None:
		for s in stats:
			lines += (("   "+s).ljust(10)+"$ ").ljust(13)+str(stats[s])+"\n"
	
	out(lines[:-1])

def model(model,stats,excep=None):
	if excep is not None:
		for e in excep:
			exception(e)

	lines = ("Model ".ljust(10)+"$ ").ljust(13)+model+"\n"

	if stats is not None:
		for s in stats:
			lines += (("   "+s).ljust(10)+"$ ").ljust(13)+str(stats[s])+"\n"
	
	out(lines[:-1])

#_____OUT______
def out(string):
	global last_time
	print("\n"+string+"\nDate:{0}\nRuntime:{2}\nPartial:{1}".format(datetime.now(),datetime.now()-last_time,datetime.now()-start_time),flush=True)
	last_time=datetime.now()

def err(string):
	print("\n"+string+"\nDate:{0}\nRuntime:{2}\nPartial:{1}".format(datetime.now(),datetime.now()-last_time,datetime.now()-start_time),file=sys.stderr,flush=True)

#_____SAVING______
def save_train(dict_l,i,excep=None):
	if excep is not None:
		for e in excep:
			exception(e)

	l = [d for d in dict_l]
	
	out("______Trainning_{0}_{1}__________".format(i,l).ljust(55,'_'))
	for d in dict_l:
		out("Train$ {0} : {1}".format(d,dict_l[d]).ljust(55,'_'))

def save_losses(dict_l,i,excep=None):
	"""
	Receives a dict of losses with an hierarchy of directories where they are gona be saved
	"""
	if excep is not None:
		for e in excep:
			exception(e)

	global loss_points
	global loss_dataframe
	list_l1 = list(dict_l.items())
	dict_l2 = {}
	out("______Saving_Losses_{0}__________".format(i).ljust(55,'_'))
	d = {}
	while list_l1:
		l = list_l1.pop()
		if isinstance(l[1],dict):
			os.makedirs(path+"/losses/"+l[0],exist_ok=True)
			for elem in l[1]:
				list_l1.append((l[0]+"/"+elem,l[1][elem]))
		else:
			out("Loss$ {0} :\t{1}".format(l[0],l[1]))
			dict_l2[l[0]] = l[1]
			#Save in global dict
			d[l[0]]=float(l[1])
			if loss_points.get(l[0]) is None:
				loss_points[l[0]]=[]
			path1=path+"/losses/"+l[0]+".png"
			loss_points[l[0]]+=[(i,float(l[1]))]
			x, y = zip(*loss_points[l[0]])
			x = list(x)
			y = list(y)
			l1 = np.array(y)
			a = np.divide(np.cumsum(l1),np.cumsum(np.ones_like(l1)))
			fig,ax = plt.subplots( nrows=1, ncols=1 )
			ax.plot(x,y,"-k",x,a,"-r")
			if len(y) < 50:
				ax.plot(x,y,"ok",x,a,"or")
			plt.title(l[0]+" Plot")
			plt.xlabel('iterations')
			plt.legend(('Loss','Average Loss'))
			fig.savefig(path1)
	loss_dataframe = loss_dataframe.append(pd.DataFrame(dict_l2,[i]))
	loss_dataframe.to_excel(path+"/losses/losses.xlsx",sheet_name="Losses")

def save_samples(dict_s,i,excep=None):
	def save_samples_aux(a,path,i,ds):
		#print("save_samples_aux",type(a),path,i,ds)

		if isinstance(a,dict):
			for e in a:
				save_samples_aux(a[e],path+"/"+e,i,ds)
		else:
			os.makedirs(path,exist_ok=True)
			for d in datasets_configs:
				if d['id'] == ds:
					d["save_sample"](a,path+"/"+str(i))
					out("Sample$ {2} : {0} : {1}".format(path,i,ds))
					return
			with open(path+"/"+str(i)+".p", 'wb') as outfile:
				out("Sample$ --- : {0} : {1}".format(path,i))
				pickle.dump(a, outfile)

	if excep is not None:
		for e in excep:
			exception(e)

	out("______Saving_Samples_{0}__________".format(i).ljust(55,'_'))
	for dataset in dict_s:
		save_samples_aux(dict_s[dataset],path+"/samples/"+dataset,i,dataset)


def save_model(i,excep=None):
	if excep is not None:
		for e in excep:
			exception(e)

	out("______Saving_Model_{0}__________".format(i).ljust(55,'_'))
	try:
		shutil.copytree(rconfig['model'], path+"/model/"+str(i))
	except FileExistsError:
		shutil.rmtree(path+"/model/"+str(i))
		shutil.copytree(rconfig['model'], path+"/model/"+str(i))
