# # LOADING DATA
import sys
import pandas as pd
import numpy as np
import re
from collections import Counter
import itertools
import json
from sklearn.model_selection import train_test_split
import os
import shutil
from datetime import datetime
from nltk import word_tokenize
from nltk.stem import SnowballStemmer



# In[2]:
def stemming(x, stem_value):
	stemmer = SnowballStemmer('spanish')
	st=[]
	for phrase in x:
		ph=[]
		for word in phrase:
			if stem_value == 1:
				ph.append(stemmer.stem(word))
			elif stem_value == 2:
				ph.append(stemmer.stem(stemmer.stem(word)))
			elif stem_value == 0:
				ph.append(word)
			else:
				print("\n\nERROR: Invalid stemming value, please change it to either 0, 1 or 2\n\n\n")
				sys.exit()
		st.append(ph)

	return st

def limpieza(s):
	#s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
	s = re.sub(r":", " : ", s)
	s = re.sub(r"\.", " . ", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"¡", " ¡ ", s)
	s = re.sub(r"\(", " ( ", s)
	s = re.sub(r"\)", " ) ", s)
	s = re.sub(r"\?", " ? ", s)
	s = re.sub(r"¿", " ¿ ", s)
	s = re.sub(r"/", " / ", s)
	return s.strip().lower()


def PAD_SENTENCES(x, default="<FAKE>", max_len=None):
	if max_len is None:
		length = max(len(i) for i in x)
	else:
		length = max_len

	x_padded = []
	for phrase in x:
		n_paddings = length - len(phrase)

		if n_paddings < 0:
			phrase_padded = phrase[0:length]
		else:
			phrase_padded = phrase + [default]*n_paddings
		x_padded.append(phrase_padded)
	return x_padded
	

def RAE(x):
	wc = Counter(itertools.chain(*x))
	vocabulary_sorted = [word[0] for word in wc.most_common()]
	vocabulary_dict = {word: index for index, word in enumerate(vocabulary_sorted)}
	return vocabulary_dict
	
	
def Read_Data(file, stem_value):
	data=pd.read_csv(file,header=None, usecols=[0,1])
	data = data.drop(data[data[1] == 'liniaObertaWSLOE'].index)
	
	#Y
	categorias = sorted(list(set(data[1].tolist())))
	num_cat = len(categorias)
	one_hot = np.zeros((num_cat, num_cat), int)
	np.fill_diagonal(one_hot, 1)
	cat_dict = dict(zip(categorias, one_hot))
	y = data[1].apply(lambda y: cat_dict[y]).tolist()
	y_train = np.array(y)
	print("\n\n------------------------------------------------------")
	print("------------------DEBUG Categories-------------------")
	print("------------------------------------------------------\n\n")
	for i,j in enumerate(np.array(y).sum(axis=0)):
		print('----->In', categorias[i], 'there are',j, 'entries')
	print("\n**",len(categorias) , "Categories succefully loaded\n\n")
	
	#X
	print("\n\n------------------------------------------------------")
	print("-------------------DEBUG Variations-------------------")
	print("------------------------------------------------------\n\n")
	x = data[0].apply(lambda x: limpieza(x).split(' ')).tolist()
	x = stemming(x, stem_value)
	x = PAD_SENTENCES(x)
	vocabulario_dict = RAE(x)
	x_train = np.array([[vocabulario_dict[word] for word in variacion] for variacion in x])
	print("\n**Variations succefully loaded\n\n")
	
	return x_train, y_train, vocabulario_dict, categorias

def Ini_embeddings(voc):
	embeds={}
	i=0
	for word in voc:
		embeds[word] = np.random.uniform(-0.25, 0.25, 300)
	return embeds
	
	
def batch_iter(data, batch_size, epochs, random=True, cv=False):
	data = np.array(data)
	batches_epoch = int(len(data)/ batch_size) + 1
	for epoch in range(epochs):
		if random:
			indices = np.random.permutation(np.arange(len(data)))
			data_f = data[indices]
		else:
			data_f = data
		for i in range(batches_epoch):
			complete=100*i/batches_epoch
			if(cv==False):
				print("Época", epoch, ':', round(complete, 2), '% completado.', i, '/', batches_epoch, 'pasos')
			start = i*batch_size
			end = min((i+1)*batch_size, len(data))
			if(start!=end):
				yield data_f[start:end]