import os, sys
import argparse
import util
import datasets.dataloader as dd
from keras.optimizers import *
from keras.callbacks import *
from models.GAHs import GAHs_trans
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from data_trans import Data_trans
import datasets.ljqpy as ljqpy
import numpy as np
import random
from models.Transformer import Transformer_trans, LRSchedulerPerStep
import nltk

itokens, otokens = dd.MakeS2SDict('datasets/wmt16/en2de.s2s.txt', dict_file='datasets/wmt16/en2de_word.txt')
Xtrain, Ytrain = dd.MakeS2SData('datasets/wmt16/en2de.s2s.txt', itokens, otokens, h5_file='datasets/wmt16/en2de.h5',max_len=70)	# np.array((samplesize, max_seq_lenth))
# Xvalid, Yvalid = dd.MakeS2SData('datasets/wmt16/en2de.s2s.valid.txt', itokens, otokens, h5_file='datasets/wmt16/en2de.valid.h5',max_len=70)
Xvalid, Yvalid = dd.MakeS2SData('datasets/wmt16/test', itokens, otokens, h5_file='datasets/wmt16/en2de.valid.h5',max_len=70)

# padd again
def padding(xs):
	seqs = []
	for i in range(len(xs)):
		seq = list(xs[i])+np.zeros(70-len(xs[i]),dtype=int).tolist()
		seqs.append(seq)
	return np.asarray(seqs)

Xtrain, Ytrain = padding(Xtrain), padding(Ytrain)
Xvalid, Yvalid = padding(Xvalid), padding(Yvalid)

print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num())
print('train shapes:', Xtrain.shape, Ytrain.shape)
print('valid shapes:', Xvalid.shape, Yvalid.shape)


'''
from rnn_s2s import RNNSeq2Seq
s2s = RNNSeq2Seq(itokens,otokens, 256)
s2s.compile('rmsprop')
s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, validation_data=([Xvalid, Yvalid], None))
'''

grid_pool ={
	"dropout" : [0.1,0.2,0.3],#
	"lr":[0.001, 0.0001, 0.0005], # 0.01 for CNN and LSTM, 0.0005,
	"batch_size":[32,64,96],#32,64,96
	"layers" : [2,4,6], # 2,4,6
	"n_head" : [6,8],#6,8
	"d_inner_hid" : [256,512],# ,512
}


def train(opt, train_masks,test_masks):

	# choose a random combinations
	para_str = ''
	for key in grid_pool:
		value = random.choice(grid_pool[key])
		setattr(opt, key, value)
		para_str = para_str + key + str(value)+'_'

	# drop or add one
	opt.sample_i = random.sample(range(5),k=opt.k_roles)
	opt.sample_i = [3]
	# para_str = str(opt.sample_i)+'_'+para_str
	print('[paras]:',para_str)
	setattr(opt,'para_str',para_str)


	d_model = 256	# embedding is 256 ? but there is no embedding I guess??
	# s2s = Transformer_trans(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=opt.d_inner_hid, \
	# 				   n_head=opt.n_head, layers=opt.layers, dropout=opt.dropout)

	s2s = GAHs_trans(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512, \
					   n_head=8, layers=2, dropout=0.1)

	mfile = 'saved_model/en2de.'+s2s.__class__.__name__+'_best_model.h5'

	lr_scheduler = LRSchedulerPerStep(d_model, 4000) 
	model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

	# initialize the model
	s2s.compile(Adam(opt.lr, 0.9, 0.98, epsilon=1e-9), opt=opt)
	
	try: s2s.model.load_weights(mfile)
	except: print('\n\nnew model')

	print('run mode:',opt.run_mode)
	
	if 'eval' in opt.run_mode:
		for x, y in s2s.beam_search('A black dog eats food .'.split(), delimiter=' '):
			print(x, y)
		print(s2s.decode_sequence_readout('A black dog eats food .'.split(), delimiter=' '))
		print(s2s.decode_sequence_fast('A black dog eats food .'.split(), delimiter=' '))
		while True:
			quest = input('> ')
			print(s2s.decode_sequence_fast(quest.split(), delimiter=' '))
			rets = s2s.beam_search(quest.split(), delimiter=' ')
			for x, y in rets: print(x, y)
	elif 'test' == opt.run_mode:
		valids = ljqpy.LoadCSV('datasets/wmt16/en2de.s2s.valid.txt')	# np.array([ [en_sent, de_sent] ])  
		# valids = ljqpy.LoadCSV('datasets/wmt16/test')  

		reference = [[x[1].split()] for x in valids]	# tgt_true

		en = [x[0].split() for x in valids]	# np.array([ [token_list] ]), e.g. [['a', 'man', 'went']]

		# first prediction
		rets = s2s.decode_sequence_readout(en, delimiter=' ')
		predict = []
		for i,x in enumerate(rets): 	# translated sentence; 
			predict.append(x.split())
		# print(len(predict),predict[:3])
		print(corpus_bleu(reference,predict))

		rets = s2s.beam_search(en, delimiter=' ', verbose=1)
		predict = []
		for i, x in enumerate(rets):
			# print('-'*20)
			# print(i,' reference sent: ',valids[i][1])
			pred,score = x[0]
			predict.append(pred.split())	# 34.72
			# for pred,score in x: 	# y is a tuple (translation_sent, score)
			# 	print(pred,score)
		print(corpus_bleu(reference,predict))

		rets = s2s.decode_sequence_fast(en, delimiter=' ', verbose=1)
		predict = []
		for i,x in enumerate(rets): 
			# print(i,':',x)
			predict.append(x.split())
		print(corpus_bleu(reference,predict))

	elif 'train' in opt.run_mode:
		s2s.model.summary()
	
		history = s2s.model.fit([Xtrain, Ytrain]+train_masks, None, batch_size=opt.batch_size, epochs=25, \
					validation_data=([Xvalid, Yvalid]+test_masks, None), \
					callbacks=[lr_scheduler, model_saver])
		max_his = str(max(history.history["val_accu"]))[:7] if max(history.history["val_accu"])>0.2 else '0.1'
		record = s2s.__class__.__name__ + max_his + para_str
		write_record(record,'translate_res.txt')
		dirname = 'saved_model/'
		os.rename(mfile,os.path.join( dirname,  record+".h5" ))

		# val_accu @ 30 epoch: 0.7045
	elif 'mytest' in opt.run_mode:
		# predict res
		# try: s2s.model.load_weights("saved_model/GAHs_trans0.71268[2, 1, 0, 4]_dropout0.1_lr0.0001_batch_size32_layers4_n_head8_d_inner_hid256_[2, 1, 0, 4].h5")
		try: s2s.model.load_weights("saved_model/GAHs_trans0.71093[3]_dropout0.1_lr0.0005_batch_size96_layers4_n_head8_d_inner_hid512_.h5")
		except: print('\n\nnew model')
		predicts = s2s.model.predict([Xvalid, Yvalid]+test_masks)
		# print(predicts[:2])
		# print('-'*5)
		
		# truth
		reference = Yvalid
		reference2 = []
		for ref in reference:
			temp_ref = []
			for tok in ref:
				if tok!=0: temp_ref.append(otokens.token(tok))
			reference2.append(temp_ref[1:len(temp_ref)-1])
		print(reference2[:5])
		# predicts
		predicts = np.argmax(predicts, axis=-1)
		# predicts2 = [predicts[i][:len(reference2[i])] for i in range(len(predicts))]
		predicts2 = []
		for i,pred in enumerate(predicts):
			temp_pre = []
			for j in range(len(reference2[i])+5):
				if otokens.token(pred[j])=='</S>':break
				temp_pre.append(otokens.token(pred[j]))
			predicts2.append(temp_pre)
		print(predicts2[:5])
		# scoring
		reference2 = [[ref] for ref in reference2]
		score = nltk.translate.bleu_score.corpus_bleu(reference2, predicts2)
		#score = nltk.translate.bleu_score.corpus_bleu(target, pred, smoothing_function=smoothing.method4)
		print('bleu:',score)

def write_record(content,file_path):
	with open(file_path,'a',encoding='utf8') as fw:
		fw.write(content)
		fw.write('\n')


if __name__ == '__main__':
		# initialize paras
	parser = argparse.ArgumentParser(description='run the training.')
	parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
	parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
	parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
	parser.add_argument('--patience', type=int, default=6)
	parser.add_argument('--load_role',type=bool, default=True)
	parser.add_argument('--all_roles', default=['posit','bothDi','majorR','sep','rareW'])
	# [['posit','bothDi','majorR','sep','rareW','noun','verb','adj','neg']] 
	parser.add_argument('--max_sequence_length', type=int,default=70)
	parser.add_argument('--run_mode',default='train')
	parser.add_argument('--k_roles', type=int,default=1)
	
	args = parser.parse_args()
	# set parameters from config files
	util.parse_and_set(args.config,args)

	data_trans = Data_trans(args)
	splits = ['train','test']
	train_masks,test_masks = data_trans.load_masks('wmt16', splits)
	# print(train_masks[:5])
	# print('-'*10)
	# print(test_masks[:5])
	if 'train' in args.run_mode:
		for i in range(20):
			print('search times:', i)
			train(args, train_masks,test_masks)
	else:
		train(args, train_masks,test_masks)


# 40.29, 37.86