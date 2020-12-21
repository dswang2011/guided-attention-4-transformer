"""
Read raw data, you can specify your data file path here or in config.init
"""


import os
import numpy as np
# import tensorflow_datasets as tfds


class RawLoader(object):
	def __init__(self,opt):
		self.opt = opt


	def load_WNLI_data(self,file_path, split='train'):
		texts1,texts2=[],[]
		labels=[]
		if split == 'train':
			file = self.opt.wnli_train_path
		elif split =='valid':
			file = self.opt.wnli_valid_path
		elif split=='test':
			ile = self.opt.wnli_test_path

		# with open(file, 'r', encoding='utf8', errors='ignore') as f:
		# 	for row in f:
		
		return [texts1,texts2],labels

	def load_TREC_data(self, split='train'):
		texts,labels = [],[]
		if split=='train':
			file_path = 'datasets/TREC/TREC.train.all'
		elif split=='test':
			file_path = 'datasets/TREC/TREC.test.all'
		with open(file_path,'r',encoding='utf8') as fr:
			for line in fr:
				line = self.processed_text(line)
				strs = line.strip().split(' ',1)
				texts.append(strs[1])
				labels.append(strs[0])
		return texts,labels


	def load_MR_data(self,split='train'):
		texts,labels = [],[]
		# file_path = 'datasets/MR/rt-polarity.all'
		folder_path = 'datasets/MR/'
		with open(folder_path+split+'.csv','r',encoding='utf8') as fr:
			for line in fr:
				# line = self.processed_text(line)
				strs = line.strip().split('\t',1)
				texts.append(strs[0].strip())
				labels.append(strs[1].strip())
		return texts,labels

	def load_IMDB_data(self, split='train'):
		texts, labels = [], []
		folder_path = 'datasets/IMDB/'
		with open(folder_path+split+'.csv', 'r', encoding='utf8') as fr:
			for line in fr:
				# line = self.processed_text(line)
				strs = line.strip().split(',', 1)
				texts.append(strs[1])
				labels.append(strs[0])
		return texts, labels

	def load_YELP_data(self, split='train'):
		texts, labels = [], []
		folder_path = 'datasets/YELP/'
		with open(folder_path+split+'.csv', 'r', encoding='utf8') as fr:
			for line in fr:
				# line = self.processed_text(line)
				strs = line.strip().split(',', 1)
				texts.append(strs[1])
				labels.append(strs[0])
		return texts, labels

	def load_DBPEDIA_data(self, split='train'):
		texts, labels = [], []
		folder_path = 'datasets/DBPEDIA/'
		with open(folder_path+split+'.csv', 'r', encoding='utf8') as fr:
			for line in fr:
				# line = self.processed_text(line)
				strs = line.strip().split(',', 1)
				texts.append(strs[1])
				labels.append(strs[0])
		return texts, labels

	def load_SST_data(self, split='train'):
		texts, labels = [], []
		folder_path = 'datasets/SST/'
		with open(folder_path+split+'.csv', 'r', encoding='utf8') as fr:
			for line in fr:
				# line = self.processed_text(line)
				strs = line.strip().split(',', 1)
				texts.append(strs[1])
				labels.append(strs[0])
		return texts, labels

	def load_ROTTENTOMATOES_data(self, split='train'):
		texts, labels = [], []
		folder_path = 'datasets/ROTTENTOMATOES/'
		with open(folder_path+split+'.csv', 'r', encoding='utf8') as fr:
			for line in fr:
				# line = self.processed_text(line)
				strs = line.strip().split(',', 1)
				texts.append(strs[1])
				labels.append(strs[0])
		return texts, labels

	def load_AGNews_data(self, split='train'):
		texts, labels = [], []
		folder_path = 'datasets/AGNews/'
		with open(folder_path+split+'.csv', 'r', encoding='utf8') as fr:
			for line in fr:
				# line = self.processed_text(line)
				strs = line.strip().split(',', 1)
				texts.append(strs[1])
				labels.append(strs[0].replace('\"',''))
		return texts, labels

	def load_SUBJ_data(self,split='train'):
		texts,labels = [],[]
		folder_path = 'datasets/SUBJ/'
		with open(folder_path+split+'.csv', 'r', encoding='utf8') as fr:
			for line in fr:
				# line = self.processed_text(line)
				strs = line.strip().split('\t', 1)
				texts.append(strs[0])
				labels.append(strs[1].strip())
		return texts, labels

	def load_wmt16_Data(self,split='train'):
		texts, labels = [],[]
		folder_path = 'datasets/wmt16/'
		with open(folder_path+split, 'r', encoding='utf8') as fr:
			for line in fr:
				strs = line.strip().split('\t',1)
				if len(strs)<2:
					continue
				texts.append(strs[0].strip())
				labels.append(strs[1].strip())
		return texts,labels

	def load_CR_data(self,split='train'):
		texts,labels = [],[]
		folder_path = 'datasets/CR/'
		with open(folder_path+split+'.csv', 'r', encoding='utf8') as fr:
			for line in fr:
				# line = self.processed_text(line)
				strs = line.strip().split('\t', 1)
				if len(strs)<2:
					print(line)
					continue
				texts.append(strs[0])
				labels.append(strs[1].strip())
		return texts, labels

	def load_MPQA_data(self,split='train'):
		texts,labels = [],[]
		folder_path = 'datasets/MPQA/'
		for file in ['neg','pos']:
			if file=='neg':
				label = 0
			else:
				label = 1
			with open(folder_path+file, 'r', encoding='utf8') as fr:
				for line in fr:
					# line = self.processed_text(line)
					texts.append(line.strip())
					labels.append(label)
		return texts, labels

	# the only one
	def load_data(self,dataset,split="train"):
		root = 'datasets/'
		if dataset in ['IMDB']:
			texts,labels = self.load_IMDB_data(split=split)
		elif dataset in ['YELP']:
			texts,labels = self.load_YELP_data(split=split)
		elif dataset in ['SST']:
			texts,labels = self.load_SST_data(split=split)
		elif dataset in ['ROTTENTOMATOES']:
			texts,labels = self.load_ROTTENTOMATOES_data(split=split)
		elif dataset in ['DBPEDIA']:
			texts,labels = self.load_DBPEDIA_data(split=split)
		elif dataset in ['WNLI']:
			texts,labels = self.load_WNLI_data(split=split)
		# other files
		elif dataset == 'MR':
			texts,labels = self.load_MR_data(split=split)
		elif dataset == 'TREC':
			texts,labels = self.load_TREC_data(split=split)
		elif dataset in ['AGNews']:
			texts,labels = self.load_AGNews_data(split=split)
		elif dataset in ['SUBJ']:
			texts,labels = self.load_SUBJ_data(split=split)
		elif dataset == 'wmt16':
			texts, labels = self.load_wmt16_Data(split=split)
		elif dataset == 'CR':
			texts,labels = self.load_CR_data(split=split)
		elif dataset =='MPQA':
			texts,labels = self.load_MPQA_data(split=split)
		return texts,labels

	def processed_text(self,text):
		text = text.replace('\\\\', '')
		text = text.replace('\n','')
		#stripped = strip_accents(text.lower())
		text = text.lower()
		return text

if __name__ == '__main__':
	rawLoader = RawLoader(None)
	texts,labels = rawLoader.load_data('CR',split='test')
	# sum_length,avg = 0,0
	# for text in texts:
	# 	strs = text.split(' ')
	# 	sum_length+=len(strs)
	# print('avg',sum_length/len(texts))

	print(len(texts),len(labels))
	# print(set(labels))
	print(texts[:5])
	print('-'*10)
	print(labels[:5])