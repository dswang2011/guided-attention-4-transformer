"""
This is to preprocess the raw files into semantic annotated token list.
by Dongsheng, April 6, 2020
"""

import spacy
import pickle  
import os
import argparse
from preprocessor.raw_data_loader import RawLoader

nlp = spacy.load("en")


class SemToken(object):
	def __init__(self,opt):
		self.opt = opt
		self.ispair = self.opt.pair_set.split(",")
		self.root = 'datasets/'	


	# text to lit of semtoks
	def get_semtok_list(self,text):
		doc = nlp(text.strip())
		return doc
		# return [token for token in doc]

	# texts to semtoks
	def semtok_on_texts(self,texts):
		return [self.get_semtok_list(text) for text in texts]

	# process the data
	def process_file(self, dataset,splits=['train','test']):
		# open data
		rawLoader = RawLoader(self.opt)
		for split in splits:
			texts,labels = rawLoader.load_data(dataset,split)
			# process and save
			if dataset in self.opt.pair_set.split(','):
				texts1,texts2 = self.semtok_on_texts(texts[0]),self.semtok_on_texts(texts[1])
				pickle.dump([[texts1,texts2],train_labels],open(os.path.join(self.root,dataset,split+'.pkl'), 'wb'))
				print('Generated: ', os.path.join(self.root,dataset,split+'.pkl'))
			else:
				texts = self.semtok_on_texts(texts)
				pickle.dump([texts,labels],open(os.path.join(self.root,dataset,split+'.pkl'), 'wb'))
				print('Generated: ', os.path.join(self.root,dataset,split+'.pkl'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='run the training.')
	parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
	parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
	parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
	parser.add_argument('--patience', type=int, default=6)
	parser.add_argument('--pair_set',default='GAP')
	args = parser.parse_args()
	
	semTok = SemToken(args)

	# custom
	dataset = 'MR'
	splits = ['train']

	semTok.process_file(dataset, splits)