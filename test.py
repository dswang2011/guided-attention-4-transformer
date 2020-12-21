
import argparse
import models
import numpy as np
import data_helper
# import util
# import random
import tensorflow as tf
from keras.models import load_model


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='run the training.')
	parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
	parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
	parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
	parser.add_argument('--patience', type=int, default=7)
	parser.add_argument('--epoch_num', type=int, default=40)
	parser.add_argument('--search_times', type=int, default=60)
	parser.add_argument('--load_role',type=bool, default=False)
	parser.add_argument('--dataset', default="MR")
	parser.add_argument('--max_sequence_length', type=int,default=90)
	parser.add_argument('--k_roles', type=int,default=6)
	parser.add_argument('--cus_pos',default='N')
	args = parser.parse_args()
	# set parameters from config files
	util.parse_and_set(args.config,args)

	# load data
	data = data_helper.Data_helper(opt)
	train_test = data.load_train(opt.dataset, dataset_pool[opt.dataset])
	# split into input (X) and output (Y) variables
	if len(dataset_pool[opt.dataset])>1:
		train,test = train_test
	else:
		[train],test = train_test, None 
	model = load_model('model.h5')
	# evaluate the model
	score = model.evaluate(test[0], test[1], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))




if __name__ == '__main__':
	# initialize paras
	parser = argparse.ArgumentParser(description='run the training.')
	parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
	parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
	parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
	parser.add_argument('--patience', type=int, default=7)
	parser.add_argument('--epoch_num', type=int, default=40)
	parser.add_argument('--search_times', type=int, default=60)
	parser.add_argument('--load_role',type=bool, default=False)
	parser.add_argument('--dataset', default="MR")
	parser.add_argument('--max_sequence_length', type=int,default=90)
	parser.add_argument('--k_roles', type=int,default=6)
	parser.add_argument('--cus_pos',default='N')
	args = parser.parse_args()
	# set parameters from config files
	util.parse_and_set(args.config,args)
	# train
	print('== Currently train set is:==', args.dataset)
	
	train_grid(args)
