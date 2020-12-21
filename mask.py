"""
This is role oriented mask generation

"""
from nltk.corpus import stopwords

import numpy as np
# np.set_printoptions(threshold=np.inf)
import heapq
import re
from scipy import sparse

class RoleMask(object):
	def __init__(self, opt):
		self.opt = opt

		# POS tag category
		self.noun_list = ['NN','NNS','NNP','NNPS']
		self.verb_list = ['VB', 'VBZ', 'VBD', 'VBG','VBN','VBP']
		self.adjective_list = ['JJ','JJR','JJS']

		self.punctuations = [';','?',',','.',':']
		self.major_rels = ['nsubj', 'dobj', 'amod', 'advmod']

		self.is_numberic = re.compile(r'^[-+]?[0-9.]+$')

	def enable_neibor(self,mask,i,neib_num,MAX_SEQUENCE_LENGTH,last=False):
		if last == True:
			for j in range(neib_num):
				if i-j>=0: mask[i][i-j]=1.
		else:
			for j in range(neib_num):
				if i+j<MAX_SEQUENCE_LENGTH: mask[i][i+j]=1.
				if i-j>=0: mask[i][i-j]=1.

	# tested
	def positional_masks_of_texts(self, tokens_lists, word_index, MAX_SEQUENCE_LENGTH,neib_num=3):
		masks = np.zeros((len(tokens_lists),MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH),dtype='float16')
		# test_count=0
		for text_id, text in enumerate(tokens_lists):
			mask = masks[text_id]
			# START
			lenth = min(len(text)+2,MAX_SEQUENCE_LENGTH)
			for i in range(lenth):
				if i == lenth-1: self.enable_neibor(mask,i,neib_num,MAX_SEQUENCE_LENGTH,last=True)
				else: self.enable_neibor(mask,i,neib_num,MAX_SEQUENCE_LENGTH)
		# masks = sparse.csr_matrix(masks)
		return masks

	# tested
	def POS_masks_of_texts(self, tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
		masks = np.zeros((len(tokens_lists),MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH),dtype='float16')
		if self.opt.cus_pos in ['A','a']:
			include_tags = self.adjective_list #+ self.verb_list, self.noun_list +
		elif self.opt.cus_pos in ['N','n']:
			include_tags = self.noun_list
		else:
			include_tags = self.verb_list

		for tid, text in enumerate(tokens_lists):
			mask = masks[tid]
			# Start
			val_index = [0]
			# Body
			row=1
			for semtok in text:
				if semtok.tag_ in include_tags:
					val_index.append(row)
				row+=1
				if row>=MAX_SEQUENCE_LENGTH: break
			# END
			if row<MAX_SEQUENCE_LENGTH:
				val_index.append(row)
			# assign val part
			for m in val_index:
				for n in val_index:
					mask[m][n]=1.
		return masks

	def POS_Noun_mask(self, tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
		masks = np.zeros((len(tokens_lists),MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH),dtype='float16')
		include_tags = self.noun_list

		for tid, text in enumerate(tokens_lists):
			mask = masks[tid]
			# Start
			val_index = [0]
			# Body
			row=1
			for semtok in text:
				if semtok.tag_ in include_tags:
					val_index.append(row)
				row+=1
				if row>=MAX_SEQUENCE_LENGTH: break
			# END
			if row<MAX_SEQUENCE_LENGTH:
				val_index.append(row)
			# assign val part
			for m in val_index:
				for n in val_index:
					mask[m][n]=1.
		return masks

	def POS_Verb_mask(self, tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
		masks = np.zeros((len(tokens_lists),MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH),dtype='float16')
		include_tags = self.verb_list

		for tid, text in enumerate(tokens_lists):
			mask = masks[tid]
			# Start
			val_index = [0]
			# Body
			row=1
			for semtok in text:
				if semtok.tag_ in include_tags:
					val_index.append(row)
				row+=1
				if row>=MAX_SEQUENCE_LENGTH: break
			# END
			if row<MAX_SEQUENCE_LENGTH:
				val_index.append(row)
			# assign val part
			for m in val_index:
				for n in val_index:
					mask[m][n]=1.
		return masks

	def POS_Adjective_mask(self, tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
		masks = np.zeros((len(tokens_lists),MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH),dtype='float16')
		include_tags = self.adjective_list

		for tid, text in enumerate(tokens_lists):
			mask = masks[tid]
			# Start
			val_index = [0]
			# Body
			row=1
			for semtok in text:
				if semtok.tag_ in include_tags:
					val_index.append(row)
				row+=1
				if row>=MAX_SEQUENCE_LENGTH: break
			# END
			if row<MAX_SEQUENCE_LENGTH:
				val_index.append(row)
			# assign val part
			for m in val_index:
				for n in val_index:
					mask[m][n]=1.
		return masks

	def POS_masks_of_texts2(self, tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
		masks = np.zeros((len(tokens_lists),MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH),dtype='float16')
		if self.opt.cus_pos in ['A','a']:
			include_tags = self.adjective_list #+ self.verb_list, self.noun_list +
		elif self.opt.cus_pos in ['N','n']:
			include_tags = self.noun_list
		else:
			include_tags = self.verb_list

		for tid, text in enumerate(tokens_lists):
			mask = masks[tid]
			# Start
			val_index = [0]
			# Body
			row=1
			for semtok in text:
				if semtok.tag_ in include_tags:
					val_index.append(row)
				row+=1
				if row>=MAX_SEQUENCE_LENGTH: break
			# END
			if row<MAX_SEQUENCE_LENGTH:
				val_index.append(row)
			# assign val part
			lenth = min(len(text)+2,MAX_SEQUENCE_LENGTH)
			for m in range(lenth):
				for n in val_index:
					mask[m][n]=1.
		return masks

	# negation mask
	def negation_mask(self,tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
		masks = np.zeros((len(tokens_lists),MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH),dtype='float16')
		include_tags = ['neg']
		test_count=0
		for tid, text in enumerate(tokens_lists):
			mask = masks[tid]
			val_index = [0]	
			row = 1
			for semtok in text:
				if semtok.dep_ in include_tags: 
					val_index.append(row)
					# related tokens
					val_index.append(semtok.head.i+1)
					for child in semtok.children: val_index.append(child.i+1)
				row+=1
				if row>=MAX_SEQUENCE_LENGTH: break
			# END
			if row<MAX_SEQUENCE_LENGTH:
				val_index.append(row)
			# assign
			val_index = [val for val in val_index if val<MAX_SEQUENCE_LENGTH]
			val_index = list(set(val_index))
			lenth = min(len(text)+2,MAX_SEQUENCE_LENGTH)
			for m in range(lenth):
				for n in val_index:
					mask[m][n]=1.
		return masks


	def major_rel_of_texts(self,tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
		masks = np.zeros((len(tokens_lists),MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH),dtype='float16')
		include_tags = self.major_rels
		test_count=0
		for tid, text in enumerate(tokens_lists):
			mask = masks[tid]
			val_index = [0]	
			row = 1
			for semtok in text:
				if semtok.dep_ in include_tags: 
					val_index.append(row)
					# related tokens
					val_index.append(semtok.head.i+1)
					for child in semtok.children: val_index.append(child.i+1)
				row+=1
				if row>=MAX_SEQUENCE_LENGTH: break
			# END
			if row<MAX_SEQUENCE_LENGTH:
				val_index.append(row)
			# assign
			val_index = [val for val in val_index if val<MAX_SEQUENCE_LENGTH]
			val_index = list(set(val_index))
			for m in val_index:
				for n in val_index:
					mask[m][n]=1.
		return masks

	# tested
	def major_rel_of_texts2(self,tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
		masks = np.zeros((len(tokens_lists),MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH),dtype='float16')
		include_tags = self.major_rels
		test_count=0
		for tid, text in enumerate(tokens_lists):
			mask = masks[tid]
			val_index = [0]	
			row = 1
			for semtok in text:
				if semtok.dep_ in include_tags: 
					val_index.append(row)
					# related tokens
					val_index.append(semtok.head.i+1)
					for child in semtok.children: val_index.append(child.i+1)
				row+=1
				if row>=MAX_SEQUENCE_LENGTH: break
			# END
			if row<MAX_SEQUENCE_LENGTH:
				val_index.append(row)
			# assign
			val_index = [val for val in val_index if val<MAX_SEQUENCE_LENGTH]
			val_index = list(set(val_index))
			lenth = min(len(text)+2,MAX_SEQUENCE_LENGTH)
			for m in range(lenth):
				for n in val_index:
					mask[m][n]=1.
		return masks

	# tested; 
	def both_direct_masks_of_texts(self,tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
		masks = np.zeros((len(tokens_lists),MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH),dtype='float16')
		# test_count = 0 
		for text_id, text in enumerate(tokens_lists):
			# for each text or sentence
			mask = masks[text_id]
			mask[0][0]=1.
			i = 1
			for semtok in text:
				looks = [i]
				looks.append(semtok.head.i+1)	# parent
				looks+=[child.i+1 for child in semtok.children]	# children
				# looks+=[sib.i+1 for sib in semtok.head.children]	# siblings
				looks = list(set(looks))
				for look in looks:
					if look<MAX_SEQUENCE_LENGTH: mask[i][look]=1.
				i+=1
				if i>=MAX_SEQUENCE_LENGTH: break
			if i<MAX_SEQUENCE_LENGTH: mask[i][i]=1.
		return masks

	# tested; 
	def stop_word_mask(self,tokens_lists,word_index,MAX_SEQUENCE_LENGTH):
		masks = np.zeros((len(tokens_lists),MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH),dtype='float16')
		test_count = 0 
		for text_id, text in enumerate(tokens_lists):
			mask = masks[text_id]
			keep_index = [0]
			row = 1
			for semtok in text:
				if semtok.text.lower() not in stopwords.words(): keep_index.append(row)
				row+=1
				if row>= MAX_SEQUENCE_LENGTH: break
			if row<MAX_SEQUENCE_LENGTH: keep_index.append(row)
			# assign
			for m in keep_index:
			# for m in range(min(len(text)+2,MAX_SEQUENCE_LENGTH)):
				for n in keep_index:
					mask[m][n]=1.
		return masks

 
 	# 1/10
	def rare_word_mask(self,tokens_lists,word_index,MAX_SEQUENCE_LENGTH):
		masks = np.zeros((len(tokens_lists),MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH),dtype='float16')
		for text_id, text in enumerate(tokens_lists):
			mask = masks[text_id]
			rare_num = max(2,len(text)//10)
			keep_index = [(0.0,0) for i in range(rare_num)]
			row = 1
			for semtok in text:
				token = semtok.text.lower()
				if len(token)>1 and not self.is_numberic.match(token):
					idf = self.opt.idf_dict[token] if token in self.opt.idf_dict else 0.0
					heapq.heappushpop(keep_index,(idf,row))
				row+=1
				if row>= MAX_SEQUENCE_LENGTH: break
			
			rare_words = [keep_index[i][1] for i in range(rare_num)]
			# assign
			# for m in keep_index:
			for m in range(min(len(text)+2,MAX_SEQUENCE_LENGTH)):
				for n in rare_words:
					mask[m][n]=1.
		return masks

	# separator and punctuations
	def separator_mask(self,tokens_lists,word_index,MAX_SEQUENCE_LENGTH):
		masks = np.zeros((len(tokens_lists),MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH),dtype='float16') 
		for text_id, text in enumerate(tokens_lists):
			# for each text or sentence
			mask = masks[text_id]
			# get all separators and punctuations
			sep = [0]
			i=1
			for semtok in text:
				if semtok.text in self.punctuations: sep.append(i)
				i+=1
				if i>=MAX_SEQUENCE_LENGTH: break
			if i<MAX_SEQUENCE_LENGTH: sep.append(i)
			# assign
			for m in range(min(len(text)+2,MAX_SEQUENCE_LENGTH)):
				for n in sep:
					mask[m][n]=1.
		return masks


	def get_masks(self,tokens_lists,word_index, MAX_SEQUENCE_LENGTH,mask_list=['major_rels']):
		res = []
		for mask in mask_list:
			if mask == 'majorR':
				res.append(self.major_rel_of_texts2(tokens_lists,word_index, MAX_SEQUENCE_LENGTH))
			elif mask == 'posit':
				res.append(self.positional_masks_of_texts(tokens_lists,word_index, MAX_SEQUENCE_LENGTH))
			elif mask == 'POS':
				res.append(self.POS_masks_of_texts2(tokens_lists,word_index, MAX_SEQUENCE_LENGTH))
			elif mask == 'bothDi':
				res.append(self.both_direct_masks_of_texts(tokens_lists,word_index, MAX_SEQUENCE_LENGTH))
			elif mask == 'sep':
				res.append(self.separator_mask(tokens_lists,word_index, MAX_SEQUENCE_LENGTH))
			elif mask == 'stopW':
				res.append(self.stop_word_mask(tokens_lists,word_index,MAX_SEQUENCE_LENGTH))
			elif mask == 'rareW':
				res.append(self.rare_word_mask(tokens_lists,word_index,MAX_SEQUENCE_LENGTH))
			elif mask == 'noun':
				res.append(self.POS_Noun_mask(tokens_lists,word_index,MAX_SEQUENCE_LENGTH))
			elif mask == 'verb':
				res.append(self.POS_Verb_mask(tokens_lists,word_index,MAX_SEQUENCE_LENGTH))
			elif mask == 'adj':
				res.append(self.POS_Adjective_mask(tokens_lists,word_index,MAX_SEQUENCE_LENGTH))
			elif mask == 'neg':
				res.append(self.negation_mask(tokens_lists,word_index,MAX_SEQUENCE_LENGTH))
			else:
				print('='*10)
				print('failed to load mask:', mask)
		return res
