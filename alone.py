import os
import re
import csv
import sys
import json
import shutil
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from nltk import word_tokenize
from nltk.stem import SnowballStemmer

def stemming(x):
    stemmer = SnowballStemmer('spanish')
    st=[]
    ph=[]
    for word in x:
        ph.append(stemmer.stem(word))
    return ph

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
    return s.strip().lower()

def padding(x, default="<DUMMY/>", maximo=None):
    if maximo is None:
        length = max(len(i) for i in x)
    else:
        length = maximo

    padded_x = []
    n_padding = length - len(x)

    if n_padding < 0:
        padded_phrase = x[0:length]
    else:
        padded_phrase = x + [default] * n_padding
    return padded_phrase


def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	Voc_Dict = json.loads(open(trained_dir + 'words_index.json').read())
	categorias = json.loads(open(trained_dir + 'labels.json').read())

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	M_embedding = np.array(fetched_embedding, dtype = np.float32)
	return params, Voc_Dict, categorias, M_embedding

def load_test_data(data, categorias, params, Voc_Dict):

	x = (limpieza(data).split(' '))
	x = stemming(x)
	x = padding(x, maximo=params['sequence_length'])
	x_test = map_word_to_RAE(x, Voc_Dict)
	x_test = np.array(x_test)

	return x_test

def map_word_to_RAE(x, Voc_Dict):
	temp = []
	for palabra in x:
		if palabra in Voc_Dict:
			temp.append(Voc_Dict[palabra])
		else:
			temp.append(0)
	return temp

def STEP(x, params, W_m, b_m, dropout):
	with tf.device('/cpu:0'):
		with tf.name_scope('emedding_layer'):
			W_embed = W_m['embed']
			EMBED = tf.nn.embedding_lookup(W_embed, x)
			x_emb = tf.expand_dims(EMBED, -1)
	
	LEN=x.shape[1]
	filter_sizes = params['filter_sizes'].split(',')
	x_conv_pool=[]
	for i, filter_size in enumerate(filter_sizes):
		name='convolutional_layer' + filter_size
		with tf.name_scope(name):
			filter_size = int(filter_size)

			filter_shape = [filter_size, params['embedding_dim'], 1, params['num_filters']]
			W_conv = W_m['convolutional'][i]
			b_conv = b_m['convolutional'][i]

			convolution = tf.nn.conv2d(x_emb, W_conv, strides=[1, 1, 1, 1], padding='VALID', name='convolution')
			relu = tf.nn.relu(tf.add(convolution, b_conv, name='add'), name='activation-relu')
			
			pooled = tf.nn.max_pool(relu, ksize=[1, LEN - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')
			x_conv_pool.append(pooled)
	num_filters_total = params['num_filters'] * len(filter_sizes)
	x_f = tf.concat(x_conv_pool, 3)
	x_f_flat = tf.reshape(x_f, [-1, num_filters_total])
	
	# DROPOUT
	with tf.name_scope("dropout"):
		x_drop = tf.nn.dropout(x_f_flat, dropout)

	for i in range(len(W_m['HL'])):
		if i==0:
			out = tf.nn.xw_plus_b(x_drop, W_m['HL'][i], b_m['HL'][i], name="layer")
		else:
			out = tf.nn.xw_plus_b(out, W_m['HL'][i], b_m['HL'][i], name="layer")
		if i != len(W_m['HL'])-1:
			out = tf.nn.relu(out, name='relu')
	
	with tf.name_scope("output"):
		scores = out
		scores2 = tf.nn.softmax(scores, name='softmax')

	return scores2

def predict():
	trained_dir = sys.argv[1]
	if not trained_dir.endswith('/'):
		trained_dir += '/'

	params, Voc_Dict, categorias, embed = load_trained_params(trained_dir)


	n_cat = len(categorias)
	num_filters_total = params['num_filters'] * len(params['filter_sizes'].split(','))

	#Define the W and b:
	if params['hidden_units']=="":
		W_model = {'embed': [], 'convolutional': [0]*len(params['filter_sizes'].split(',')), 'HL': [0]}
		b_model = {'convolutional': [0]*len(params['filter_sizes'].split(',')), 'HL': [0]}
	else:
		W_model = {'embed': [], 'convolutional': [0]*len(params['filter_sizes'].split(',')), 'HL': [0]*(1+len(params['hidden_units'].split(',')))}
		b_model = {'convolutional': [0]*len(params['filter_sizes'].split(',')), 'HL': [0]*(1+len(params['hidden_units'].split(',')))}

	W_model['embed'] = tf.Variable(embed, name='W_embed')

	for i, filter_size in enumerate(params['filter_sizes'].split(',')):
		filter_shape = [int(filter_size), params['embedding_dim'], 1, params['num_filters']]
		W_model['convolutional'][i] = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name=('W_conv'))
		b_model['convolutional'][i] = tf.Variable(tf.constant(0.1, shape=[params['num_filters']]), name=('b_conv'))

	for i, n_neurons in enumerate(params['hidden_units'].split(',')):
		if params['hidden_units']=="":
			break
		n_neurons = int(n_neurons)
		if i==0:
			pre = len(params['filter_sizes'].split(','))*params['num_filters']
			post = n_neurons
		else:
			pre = int((params['hidden_units'].split(','))[i-1])
			post = n_neurons
		W_model['HL'][i] = tf.Variable(tf.truncated_normal([pre, post], stddev=0.1), name="W_HL")
		b_model['HL'][i] = tf.Variable(tf.constant(0.1, shape=[post]), name="b_HL")
	if params['hidden_units']=="":
		pre = len(params['filter_sizes'].split(','))*params['num_filters']
	else:
		pre = int((params['hidden_units'].split(','))[-1])
	post = n_cat
	W_model['HL'][-1] = tf.Variable(tf.truncated_normal([pre, post], stddev=0.1), name="W_HL")
	b_model['HL'][-1] = tf.Variable(tf.constant(0.1, shape=[post]), name="b_HL")

	print('W_model:\n', W_model)
	print('b_model:\n', b_model)


	X_feed = tf.placeholder(tf.int32, [None, int(params['sequence_length'])], name="X_feed")
	dropout = tf.placeholder(tf.float32, [], name='dropout')
	scores = STEP(X_feed, params, W_model, b_model, dropout)

	with tf.Session() as sess:
		model = trained_dir + 'model.ckpt'
		saver = tf.train.Saver(tf.all_variables())
		saver = tf.train.import_meta_graph("{}.meta".format(model))
		saver.restore(sess, model)
		print("\n\nModelo cargado satisfactoriamente\n")
		finish=False
		while(finish==False):
			query = input('Dime tu consulta:\n')

			x = load_test_data(query, categorias, params, Voc_Dict)
			x = np.reshape(x, (1, params['sequence_length']))
			#Placeholders for data
			feed_dict = {X_feed: x, dropout: 1}	
			scores_batch = sess.run(scores, feed_dict)
			positions = np.argsort(scores_batch, axis=1)
			answer1 = positions[:,-1]
			answer2 = positions[:,-2]
			answer3 = positions[:,-3]
			print('Respuesta 1:', categorias[int(answer1)], '(', scores_batch[0][int(answer1)], ')')
			print('Respuesta 2:', categorias[int(answer2)], '(', scores_batch[0][int(answer2)], ')')
			print('Respuesta 3:', categorias[int(answer3)], '(', scores_batch[0][int(answer3)], ')')
			a = input('¿Eso es todo? (S/N)\n')
			if(a=='S' or a=='s' or a=='si' or a=='SI' or a=='Si'):
				finish = True
				print('\n')
			elif(a=='N' or a=='n' or a=='no' or a=='NO' or a=='No'):
				finish = False
				print('\n')
			else:
				print('No he entendido la resupuesta, vuelvo al bucle...\n')
		

if __name__ == '__main__':
	predict()
