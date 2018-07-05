import re
import os
import sys
import json
import copy
import shutil
import pickle
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from Step import STEP
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Auxiliar import limpieza, PAD_SENTENCES, RAE, Read_Data, Ini_embeddings, batch_iter, stemming

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


def load_test_data(file, categorias, params, Voc_Dict, seq_len):
	data=pd.read_csv(file,header=None, usecols=[0,1])

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
	x = PAD_SENTENCES(x, max_len=seq_len)
	x_test = map_word_to_RAE(x, Voc_Dict)
	x_test = np.array(x_test)
	print("\n**Test variations succefully loaded\n\n")

	return x_test, y_test


def train(train_file, test_file, config):
	best_acc= 0
	ACCURACY_CV = []
	ACCURACY_TRAIN = []
	LOSS_CV = []
	LOSS_TRAIN = []
	params = json.loads(open(config).read())
	train_file = params["training file"]
	test_file = params["test file"]
	X_0, Y_0, Voc_Dict, Categorias =Read_Data(train_file, params['stemming'])
	embeddings= Ini_embeddings(Voc_Dict)
	M_embeddings = [embeddings[word] for index, word in enumerate(Voc_Dict)]
	M_embeddings = np.array(M_embeddings, dtype = np.float32)

	x_test, y_test = load_test_data(test_file, Categorias, params, Voc_Dict, X_0.shape[1])
	
	_, x_train_v, _, y_train_v = train_test_split(X_0, Y_0, test_size=params['train_val'])
	x_train = X_0
	y_train = Y_0
	print("\n\n------------------------------------------------------")
	print("Training with", len(x_train), "phrases,", len(x_test), "phrases in the test,", len(x_train_v), "phrases are in the training validation")
	print("------------------------------------------------------\n\n")
	try:
		trained_dir = sys.argv[1]
		if not trained_dir.endswith('/'):
			trained_dir += '/'
	except:
		now=datetime.now()
		trained_dir = './entrenamiento_' + str(now.day)+'-'+str(now.month)+'-'+str(now.year) + '/'
	if os.path.exists(trained_dir):
		shutil.rmtree(trained_dir)
	os.makedirs(trained_dir)

	n_cat = Y_0.shape[1]
	num_filters_total = params['num_filters'] * len(params['filter_sizes'].split(','))

	#Define the W and b:
	if params['hidden_units']=="":
		W_model = {'embed': [], 'convolutional': [0]*len(params['filter_sizes'].split(',')), 'HL': [0]}
		b_model = {'convolutional': [0]*len(params['filter_sizes'].split(',')), 'HL': [0]}
	else:
		W_model = {'embed': [], 'convolutional': [0]*len(params['filter_sizes'].split(',')), 'HL': [0]*(1+len(params['hidden_units'].split(',')))}
		b_model = {'convolutional': [0]*len(params['filter_sizes'].split(',')), 'HL': [0]*(1+len(params['hidden_units'].split(',')))}

	W_model['embed'] = tf.Variable(M_embeddings, name='W_embed')

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

	X_feed = tf.placeholder(tf.int32, [None, int(X_0.shape[1])], name="X_feed")
	Y_feed = tf.placeholder(tf.int32, [None, Y_0.shape[1]], name="Y_feed")
	dropout = tf.placeholder(tf.float32, [], name='dropout')
	loss, acc, _  = STEP(X_feed, Y_feed, params, W_model, b_model, dropout)

	optimizer=tf.train.AdamOptimizer(0.001)
	train_op = optimizer.minimize(loss)

	saver=tf.train.Saver()

	start = tf.global_variables_initializer()

	st = [0]


	with tf.Session() as sess:
		sess.run(start)
		#Loop in training batches
		batches = batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
		for i, batch in enumerate(batches):
			#Load batch
			x_train_batch, y_train_batch = zip(*batch)
			x_train_batch = np.array(x_train_batch)
			y_train_batch = np.array(y_train_batch)
			#Dicts for placeholders
			feed_dict = {X_feed: x_train_batch, Y_feed: y_train_batch, dropout: params['dropout_keep_prob']}
			#Optimizer
			sess.run(train_op, feed_dict)
			if(i%10==0):
				print('\n---------------------------------------------------')
				print('-----------------------DEBUG-----------------------')
				print('---------------------------------------------------')
				batches_test = batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1, cv=True)
				L_cv=0
				A_cv=0
				for j, batch_test in enumerate(batches_test):
					x_cv_batch, y_cv_batch = zip(*batch_test)
					x_cv_batch = np.array(x_cv_batch)
					y_cv_batch = np.array(y_cv_batch)
					feed_dict = {X_feed: x_cv_batch, Y_feed: y_cv_batch, dropout: 1}
					L_iter, A_iter = sess.run([loss,acc], feed_dict)
					L_cv += L_iter
					A_cv += A_iter*x_cv_batch.shape[0]
				A_cv = A_cv/x_test.shape[0]
				print("Loss (CV):", L_cv)
				print("Accuracy (CV):", A_cv)
				ACCURACY_CV.append(A_cv)
				LOSS_CV.append(L_cv)
				if best_acc<A_cv:
					best_acc = A_cv
					best_W_dict = dict(W_model)
					best_b_dict = dict(b_model)

				batches_tr = batch_iter(list(zip(x_train_v, y_train_v)), params['batch_size'], 1, cv=True)
				L_cv=0
				A_cv=0
				for j, batch_test in enumerate(batches_tr):
					x_cv_batch, y_cv_batch = zip(*batch_test)
					x_cv_batch = np.array(x_cv_batch)
					y_cv_batch = np.array(y_cv_batch)
					feed_dict = {X_feed: x_cv_batch, Y_feed: y_cv_batch, dropout: 1}
					L_iter, A_iter = sess.run([loss,acc], feed_dict)
					L_cv += L_iter
					A_cv += A_iter*x_cv_batch.shape[0]
				A_cv = A_cv/x_train_v.shape[0]
				print("Loss (train):", L_cv)
				print("Accuracy (train):", A_cv)
				print('\n---------------------------------------------------')
				print('---------------------------------------------------')
				print('---------------------------------------------------')
				ACCURACY_TRAIN.append(A_cv)
				LOSS_TRAIN.append(L_cv)
				st.append(st[-1]+1)

		W_model = best_W_dict
		b_model = best_b_dict
		print('Best model had a validation accuracy of', best_acc,'\n\n')
		print("\n-->Entrenamiento completado\n\n")
		# Save the model files to trained_dir. predict.py needs trained model files. 
		saver.save(sess, trained_dir + "model.ckpt")

	# Save trained parameters and files since predict.py needs them
	with open(trained_dir + 'words_index.json', 'w') as outfile:
		json.dump(Voc_Dict, outfile, indent=4, ensure_ascii=False)
	with open(trained_dir + 'embeddings.pickle', 'wb') as outfile:
		pickle.dump(M_embeddings, outfile, pickle.HIGHEST_PROTOCOL)
	with open(trained_dir + 'labels.json', 'w') as outfile:
		json.dump(Categorias, outfile, indent=4, ensure_ascii=False)

	params['sequence_length'] = x_train.shape[1]
	params['validation_accuracy'] = best_acc
	with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
		json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)
	print("Modelo guardado en:", trained_dir)

	del st[-1]
	st = [x*params['num_epochs']/len(st) for x in st]

	plt.figure(1)
	plt.subplot(211)
	plt.plot(st, ACCURACY_CV, 'r-', st, ACCURACY_TRAIN, 'g-')
	plt.xlabel('Step')
	plt.ylabel('Accuracy')
	plt.title('Accuracy of the model')
	plt.legend(('CV', 'Trainning'), loc='upper center', shadow=True)
	plt.subplot(212)
	plt.plot(st, LOSS_CV, 'r-', st, LOSS_TRAIN, 'g-')
	plt.xlabel('Step')
	plt.ylabel('Loss')
	plt.title('Loss function of the model')
	plt.legend(('CV', 'Trainning'), loc='upper center', shadow=True)
	plt.show()

	df_train = pd.DataFrame({
		'Step': st,
		'Loss':LOSS_TRAIN,
		'Accuracy': ACCURACY_TRAIN
		})
	df_Test = pd.DataFrame({
		'Step': st,
		'Loss':LOSS_CV,
		'Accuracy': ACCURACY_CV
		})
	df_train.to_csv(trained_dir+'Trainning.txt', sep='\t')
	df_Test.to_csv(trained_dir+'Test.txt', sep='\t')


if __name__ == '__main__':
	train_data="variaciones.csv"
	config="config.json"
	test_file = 'Test.csv'
	train(train_data, test_file, config)

