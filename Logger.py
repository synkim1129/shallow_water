import os
import time
import torch
import datetime as dt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from natsort import natsorted
import pickle

class Logger():
	def __init__(self,name,datetime=None,use_tensorboard=False):
		"""
		Logger logs metrics to CSV files / tensorboard
		:name: logging name (e.g. model name / dataset name / ...)
		:datetime: date and time of logging start (useful in case of multiple runs). Default: current date and time is picked
		:use_tensorboard: log output to tensorboard
		"""
   
		self.name = name
		if datetime:
			self.datetime=datetime
		else:
			self.datetime = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		
		self.use_tensorboard = use_tensorboard
		if use_tensorboard:
			directory = 'Logger/tensorboard/{} {}'.format(name,self.datetime)
			os.makedirs(directory,exist_ok=True)
			self.writer = SummaryWriter(directory)

	
	def log(self,item,value,index):
		"""
		log index value couple for specific item into csv file / tensorboard
		:item: string describing item (e.g. "training_loss","test_loss")
		:value: value to log
		:index: index (e.g. batchindex / epoch)
		"""

		
		if self.use_tensorboard:
			self.writer.add_scalar(item,value,index)
	
	def log_histogram(self,item,values,index):
		"""
		log index values-histogram couple for specific item to tensorboard
		:item: string describing item (e.g. "training_loss","test_loss")
		:values: values to log
		:index: index (e.g. batchindex / epoch)
		"""
		if self.use_tensorboard:
			self.writer.add_histogram(item,values,index)
	
	def log_model_gradients(self,item,model,index):
		"""
		log index model-gradients-histogram couple for specific item to tensorboard
		:item: string describing model item (e.g. "encoder","discriminator")
		:values: values to log
		:index: index (e.g. batchindex / epoch)
		"""
		if self.use_tensorboard:
			params = [p for p in model.parameters()]
			if len(params)!=0:
				gradients = torch.cat([p.grad.view(-1) for p in params if p.grad is not None])
				self.writer.add_histogram(f"{item}_grad_histogram",gradients,index)
				self.writer.add_scalar(f"{item}_grad_norm2",gradients.norm(2),index)
	
	
	def save_state(self,model,optimizer,index="final"):
		"""
		saves state of model and optimizer
		:model: model to save (if list: save multiple models)
		:optimizer: optimizer (if list: save multiple optimizers)
		:index: index of state to save (e.g. specific epoch)
		"""
		os.makedirs('Logger/{}/{}/states'.format(self.name,self.datetime),exist_ok=True)
		path = 'Logger/{}/{}/states/{}.state'.format(self.name,self.datetime,index)
		state = {}
		
		if type(model)is not list:
			model = [model]
		for i,m in enumerate(model):
			state.update({'model{}'.format(i):m.state_dict()})
		
		if type(optimizer) is not list:
			optimizer = [optimizer]
		for i,o in enumerate(optimizer):
			state.update({'optimizer{}'.format(i):o.state_dict()})
		
		torch.save(state, path)
	
	def save_dict(self,dic,index="final"):
		"""
		saves dictionary - helpful to save the population state of an evolutionary optimization algorithm
		:dic: dictionary to store
		:index: index of state to save (e.g. specific evolution)
		"""
		os.makedirs('Logger/{}/{}/states'.format(self.name,self.datetime),exist_ok=True)
		path = 'Logger/{}/{}/states/{}.dic'.format(self.name,self.datetime,index)
		with open(path,"wb") as f:
			pickle.dump(dic,f)
	
	def load_state(self,model,optimizer,datetime=None,index=None,continue_datetime=False,direct_path=None):
		"""
		loads state of model and optimizer
		:model: model to load (if list: load multiple models)
		:optimizer: optimizer to load (if list: load multiple optimizers; if None: don't load)
		:datetime: date and time from run to load (if None: take latest folder)
		:index: index of state to load (e.g. specific epoch) (if None: take latest index)
		:continue_datetime: flag whether to continue on this run. Default: False
		:return: datetime, index (helpful, if datetime / index wasn't given)
		"""
		if direct_path is not None:
			path = direct_path
			state = torch.load(path)
			if type(model) is not list:
				model = [model]
			for i,m in enumerate(model):
				m.load_state_dict(state['model{}'.format(i)])
			if optimizer is not None:
				if type(optimizer)is not list:
					optimizer = [optimizer]
				for i,o in enumerate(optimizer):
					o.load_state_dict(state['optimizer{}'.format(i)])
			return None, None

		if datetime is None:
			for _,dirs,_ in os.walk('Logger/{}/'.format(self.name)):
				datetime = sorted(dirs)[-1]
				if datetime == self.datetime:
					datetime = sorted(dirs)[-2]
				break
		
		if continue_datetime:
			#CODO: remove generated directories...
			os.rmdir()
			self.datetime = datetime
		
		if index is None:
			for _,_,files in os.walk('Logger/{}/{}/states/'.format(self.name,datetime)):
				index = os.path.splitext(natsorted(files)[-1])[0]
				break
		
		path = 'Logger/{}/{}/states/{}.state'.format(self.name,datetime,index)
		state = torch.load(path)
		
		if type(model) is not list:
			model = [model]
		for i,m in enumerate(model):
			m.load_state_dict(state['model{}'.format(i)])
		
		if optimizer is not None:
			if type(optimizer)is not list:
				optimizer = [optimizer]
			for i,o in enumerate(optimizer):
				o.load_state_dict(state['optimizer{}'.format(i)])
		
		return datetime, index
	
	def load_dict(self,dic,datetime=None,index=None,continue_datetime=False):
		"""
		loads state of model and optimizer
		:dic: (empty) dictionary to fill with state information
		:datetime: date and time from run to load (if None: take latest folder)
		:index: index of state to load (e.g. specific epoch) (if None: take latest index)
		:continue_datetime: flag whether to continue on this run. Default: False
		:return: datetime, index (helpful, if datetime / index wasn't given)
		"""
		
		if datetime is None:
			for _,dirs,_ in os.walk('Logger/{}/'.format(self.name)):
				datetime = sorted(dirs)[-1]
				if datetime == self.datetime:
					datetime = sorted(dirs)[-2]
				break
		
		if continue_datetime:
			#CODO: remove generated directories...
			os.rmdir()
			self.datetime = datetime
		
		if index is None:
			for _,_,files in os.walk('Logger/{}/{}/states/'.format(self.name,datetime)):
				index = os.path.splitext(natsorted(files)[-1])[0]
				break
		
		path = 'Logger/{}/{}/states/{}.dic'.format(self.name,datetime,index)
		with open(path,"rb") as f:
			state = pickle.load(f)
		
		for key in state.keys():
			dic[key] = state[key]
		
		return datetime, index

	def save_params_to_file(self, params):
		os.makedirs('Logger/{}/{}'.format(self.name,self.datetime),exist_ok=True)
		filename = 'Logger/{}/{}/params.txt'.format(self.name,self.datetime)
		with open(filename, 'w') as f:
			f.write(f'{self.name}, {self.datetime}\n')
			for key, value in vars(params).items():
				f.write(f'{key}: {value}\n')

t_start = 0


def t_step():
    """
    returns delta t from last call of t_step()
    """
    global t_start
    t_end = time.perf_counter()
    delta_t = t_end-t_start
    t_start = t_end
    return delta_t
