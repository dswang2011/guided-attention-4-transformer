"""
This file is to load the data for training
by Dongsheng, 2020, Aprial 07
"""

import os
#import stanfordnlp
import numpy as np
import codecs

import pickle
import argparse
from keras.utils import to_categorical
import numpy as np
from sklearn import preprocessing
import util
from mask import RoleMask
import re
import gc
from sklearn.utils import shuffle	# shuffle the input


punctuation_list = [',',':',';','.','!','?','...','…','。']


class SemtokGenerate(object):
	def __init__(self,opt):
		self.opt=opt  
		self.role_mask = RoleMask(self.opt)
		self.root = 'datasets/'	
		self.is_numberic = re.compile(r'^[-+]?[0-9.]+$')
		self.all_roles = ['positional','both_direct','major_rels','separator','rare_word']


	def load_sem_data(self,dataset,split):
		root = 'datasets/'+dataset+'/'
		texts,labels = pickle.load(open(os.path.join(root,split+'.pkl'),'rb'))
		return texts,labels

	# load the train, valid or test
	def prepare_train_data(self,dataset,splits):
		texts_list_train_test = []
		labels_train_test = []
		partition = {}
		glob_labels = {}
		le = preprocessing.LabelEncoder()
		for split in splits:
			texts,labels = self.load_sem_data(dataset,split)
			texts,labels = shuffle(texts,labels,random_state=9)	# shuffle the data to make it robust
			# global
			partition[split] = [split+'-'+str(i) for i in range(len(texts))]
			texts_list_train_test.append(texts)
			labels_train_test.append(labels)
			# glob
			y = le.fit_transform(labels)
			y = to_categorical(np.asarray(y))
			for i,label in enumerate(y):
				glob_labels[split+'-'+str(i)] = label

		self.opt.nb_classes = len(set(labels))
		print('[LABEL]',self.opt.nb_classes, ' labels:',set(labels))
		# max_num_words = self.opt.max_num_words
		if dataset in self.opt.pair_set.split(","):
			all_texts= [sentence for texts1,texts2 in texts_list_train_test for sentence in texts1]
		else:
			all_texts= [sentence for dataset in texts_list_train_test for sentence in dataset]
		
		# compute idf
		temp_txts = [doc.text for doc in all_texts]
		self.opt.idf_dict = util.get_idf_dict(temp_txts)
		
		# tokenize 
		word_index = self.tokenizer(all_texts,MAX_NB_WORDS=self.opt.max_nb_words)
		self.opt.word_index = word_index
		print('word_index:',len(word_index))

		# tag embedding
		tag_index = self.tag_index(all_texts,MAX_NB_WORDS=self.opt.max_nb_words)
		tag_onehot = to_categorical( list(tag_index.values()) )
		dep_dim = len(tag_onehot[0])

		# common parameters: word_index, tag_index, dep_dim, 
		pickle.dump([partition, glob_labels, word_index,tag_index,dep_dim,self.opt.nb_classes],open(os.path.join(self.root,dataset,'comm.pkl'), 'wb'))
		
		# release a bit
		del all_texts[:]
		del temp_txts[:]
		gc.collect()

		le = preprocessing.LabelEncoder()
		# labels = le.fit_transform(labels)
		# padding
		train_test = []
		split_count = 0
		for tokens_list,labels in zip(texts_list_train_test,labels_train_test):
			split_count+=1
			if dataset in self.opt.pair_set.split(","):
				x1 = self.tokens_list_to_sequences(tokens_list[0],word_index,self.opt.max_sequence_length)
				x2 = self.tokens_list_to_sequences(tokens_list[1],word_index,self.opt.max_sequence_length)
				x = [x1,x2]
			else:
				
				for i in range(0,len(tokens_list),10000):
					x = self.tokens_list_to_sequences(tokens_list[i:i+10000],word_index,self.opt.max_sequence_length)
					# one hot
					x_tag = self.tokens_list_to_tag_sequences(tokens_list[i:i+10000],tag_index,self.opt.max_sequence_length)
					# roles
					masks = self.role_mask.get_masks(tokens_list[i:i+10000],word_index,self.opt.max_sequence_length, self.all_roles)
					if split_count==1:
						pickle.dump([x,x_tag, masks],open(os.path.join(self.root,dataset,'train_'+str(i)+'.pkl'), 'wb'))
					else:
						pickle.dump([x,x_tag, masks],open(os.path.join(self.root,dataset,'test_'+str(i)+'.pkl'), 'wb'))


	def tokenizer(self, texts, MAX_NB_WORDS):
		word_index = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<MASK>':3, '<NUM>':4}
		index = 5
		for text in texts:
			for token in text:	# here the text is the doc
				# add to word_index
				if len(word_index)<MAX_NB_WORDS:
					token=token.text.lower()
					if token not in word_index.keys():
						word_index[token] = index
						index+=1
		return word_index

	def tag_index(self, texts, MAX_NB_WORDS):
		tag_index = {'<PAD>': 0}
		index = 1
		count = 0
		for text in texts:
			for token in text:	# here the text is the doc
				# add to word_index
				if len(tag_index)<100:	# less than 100
					tag=token.dep_
					if tag not in tag_index.keys():
						tag_index[tag] = index
						index+=1
				else:
					break
			count+=1
			if count>2000: break
		return tag_index


	# input is the generalized text; 
	def tokens_list_to_sequences(self, tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
		sequences = []
		for tokens in tokens_lists:
			sequence = [1]	# start
			for semtok in tokens:
				token = semtok.text.lower()
				if self.is_numberic.match(token):
					sequence.append(4)
				elif token in word_index.keys():
					token_index = word_index[token]
					sequence.append(token_index)
				else:
					sequence.append(0)
				
			sequence.append(2)	# end
			if len(sequence)>MAX_SEQUENCE_LENGTH:
				sequence = sequence[:MAX_SEQUENCE_LENGTH]
			else:
				sequence = sequence+np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()
			# print('seq:',sequence)
			sequences.append(sequence)
		return np.asarray(sequences,dtype=int)
		# return sequences

	def tokens_list_to_tag_sequences(self, tokens_lists, tag_index, MAX_SEQUENCE_LENGTH):
		sequences = []
		for tokens in tokens_lists:
			sequence = [0]	# start
			for semtok in tokens:
				tag = semtok.dep_
				if tag in tag_index.keys():
					index = tag_index[tag]
					sequence.append(index)
				else:
					sequence.append(0)
			sequence.append(0)	# end
			if len(sequence)>MAX_SEQUENCE_LENGTH:
				sequence = sequence[:MAX_SEQUENCE_LENGTH]
			else:
				sequence = sequence+np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()
			# print('seq:',sequence)
			sequences.append(sequence)
		return np.asarray(sequences,dtype=int)




if __name__ == '__main__':
		# initialize paras
	parser = argparse.ArgumentParser(description='run the training.')
	parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
	parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
	parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
	parser.add_argument('--patience', type=int, default=6)
	parser.add_argument('--load_role',type=bool, default=True)
	parser.add_argument('--all_roles', default=['positional','both_direct','major_rels','stop_word'])
	
	args = parser.parse_args()
	# set parameters from config files
	util.parse_and_set(args.config,args)

	data_help = Data_helper(args)
	splits = ['train','test']
	train,test = data_help.load_data('TREC', splits)

