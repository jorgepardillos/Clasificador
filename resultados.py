import os
import csv
import sys
import json
import shutil
import pickle
import numpy as np
import pandas as pd
from Step import STEP
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from Auxiliar import limpieza, stemming, PAD_SENTENCES, batch_iter


def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	Voc_Dict = json.loads(open(trained_dir + 'words_index.json').read())
	categorias = json.loads(open(trained_dir + 'labels.json').read())

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	M_embedding = np.array(fetched_embedding, dtype = np.float32)
	return params, Voc_Dict, categorias, M_embedding

def load_test_data(file, categorias, params, Voc_Dict):
	data=pd.read_csv(file,header=None, usecols=[0,1]) #0:x, 1:y

	num_cat = len(categorias)
	one_hot = np.zeros((num_cat, num_cat), int)
	np.fill_diagonal(one_hot, 1)
	cat_dict = dict(zip(categorias, one_hot))
	y = data[1].apply(lambda y: cat_dict[y]).tolist()
	y_test = np.array(y)
	A=categorias
	B=list(set(data[1]))
	A.sort()
	B.sort()
	if(A==B):
		print("\n**Test categories succefully loaded\n\n")
	else:
		print("\n**Test categories succefully loaded, though in", file, "there are only", len(list(set(data[1]))),"out of", len(categorias), "categories\n\n")

	x = data[0].apply(lambda x: limpieza(x).split(' ')).tolist()
	x = stemming(x, params['stemming'])
	x = PAD_SENTENCES(x, max_len=params['sequence_length'])
	x_test = map_word_to_RAE(x, Voc_Dict)
	x_test = np.array(x_test)

	return x_test, y_test

def map_word_to_RAE(x, Voc_Dict):
	x_ = []
	for frase in x:
		temp = []
		for palabra in frase:
			if palabra in Voc_Dict:
				temp.append(Voc_Dict[palabra])
			else:
				temp.append(0)
		x_.append(temp)
	return x_

def predict_unseen_data():
	trained_dir = sys.argv[1]
	if not trained_dir.endswith('/'):
		trained_dir += '/'
	test_file = 'Test.csv'

	params, Voc_Dict, categorias, embed = load_trained_params(trained_dir)

	test_file = params['test file']

	x, y = load_test_data(test_file, categorias, params, Voc_Dict)


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

	X_feed = tf.placeholder(tf.int32, [None, int(x.shape[1])], name="X_feed")
	Y_feed = tf.placeholder(tf.int32, [None, y.shape[1]], name="Y_feed")
	dropout = tf.placeholder(tf.float32, [], name='dropout')
	loss, acc, scores = STEP(X_feed, Y_feed, params, W_model, b_model, dropout)

	with tf.Session() as sess:
		model = trained_dir + 'model.ckpt'
		saver = tf.train.Saver(tf.all_variables())
		saver = tf.train.import_meta_graph("{}.meta".format(model))
		saver.restore(sess, model)
		print("\n\nModelo cargado satisfactoriamente\n")

		batches = batch_iter(list(zip(x, y)), params['batch_size'], 1, random=False)
		accuracy1=[]
		accuracy2=[]
		accuracy3=[]
		save_guess=[]
		save_true=[]
		created = 0
		for batch in batches:
			x_batch, y_batch = zip(*batch)
			x_batch = np.array(x_batch)
			y_batch = np.array(y_batch)
			#Placeholders for data
			feed_dict = {X_feed: x_batch, Y_feed: y_batch, dropout: 1}
			scores_batch = sess.run(scores, feed_dict)
			if created == 0:
				tot_scores = scores_batch
				created = 1
			else:
				tot_scores = np.concatenate((tot_scores, scores_batch), axis=0)
			positions = np.argsort(scores_batch, axis=1)
			true = np.argmax(y_batch, axis=1)
			answer1 = positions[:,-1]
			answer2 = positions[:,-2]
			answer3 = positions[:,-3]
			for i in range(len(answer1)):
				save_true.append(categorias[true[i]])
				save_guess.append(categorias[answer1[i]])
				if(answer1[i]==true[i]):
					accuracy1.append(1)
				else:
					accuracy1.append(0)

				if(answer2[i]==true[i]):
					accuracy2.append(1)
				else:
					accuracy2.append(0)

				if(answer3[i]==true[i]):
					accuracy3.append(1)
				else:
					accuracy3.append(0)
		a1 = round(sum(accuracy1)/len(accuracy1),4)
		a2 = round(sum(accuracy2)/len(accuracy2),4)
		a3 = round(sum(accuracy3)/len(accuracy3),4)
		print("% of accuracy in the three first options:", round(a1+a2+a3,4), '(', a1, '+', a2, '+', a3, ')')
		df_train = pd.DataFrame({'Prediction': save_guess,'Real':save_true})
		df_train.to_csv(trained_dir+'OUTPUT.txt', sep=',')

		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for i in range(tot_scores.shape[1]):
			fpr[categorias[i]], tpr[categorias[i]], _ = roc_curve(y[:, i], tot_scores[:, i])
			roc_auc[categorias[i]] = auc(fpr[categorias[i]], tpr[categorias[i]])
		with open(trained_dir + 'AUCROC.csv', 'w') as csv_file:
		    writer = csv.writer(csv_file)
		    for key, value in roc_auc.items():
		       writer.writerow([key, value])
		

if __name__ == '__main__':
	predict_unseen_data()
