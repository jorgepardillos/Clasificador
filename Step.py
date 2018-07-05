import pandas as pd
import numpy as np
import re
from collections import Counter
import itertools
import json
from sklearn.model_selection import train_test_split
import os
import shutil
import tensorflow as tf
from datetime import datetime
import sys



def STEP(x, y, params, W_m, b_m, dropout):
	#EMBEDDING LAYER
	#--> IN: (RAE, embedding_dim)
	#--> OUT: (batch_size, length_phrase, embedding_dim, 1)
	with tf.device('/cpu:0'):
		with tf.name_scope('emedding_layer'):
			W_embed = W_m['embed']
			EMBED = tf.nn.embedding_lookup(W_embed, x)
			x_emb = tf.expand_dims(EMBED, -1)
	
	#CONVOLUTIONAL LAYER + POOLING
	#--> IN: (batch_size, length_phrase, embedding_dim, 1)
	#--> OUT: (batch_size, 1, 1, channels)
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
	
	n_cat = int(y.shape[1])
	with tf.name_scope("output"):
		scores = out
		scores2 = tf.nn.softmax(scores, name='softmax')
		predictions = tf.argmax(scores, 1, name="predictions")
	
	# Mean cross-entropy loss
	with tf.name_scope("loss"):
		losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y)
		loss = tf.reduce_mean(losses)
		
	# Accuracy
	with tf.name_scope("accuracy"):
		correct_predictions = tf.equal(predictions, tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

	return loss, accuracy, scores2