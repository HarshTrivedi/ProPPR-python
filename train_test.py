import sparse
import glob
import tensorflow as tf
import logging
import numpy as np
import os.path
from sklearn.preprocessing import normalize
from sklearn import metrics
import cPickle as pickle
from setting import *
import copy
import os
import glob
import math
import nltk
import re

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
np.set_printoptions(threshold=np.nan)

##############################################################################
###################### (START) CONFIGURATION #################################
run = "Learn"  # Predict or Learn
# run = "Predict"  # Predict or Learn
###################### (END) CONFIGURATION #################################

processed_data_dir = os.path.join('ProcessedData', program_name)
feature_vector_size_path = os.path.join(processed_data_dir, 'feat_size.txt')
with open(feature_vector_size_path, 'r') as f:
    edge_feature_vector_size = int(f.read().strip())

feature_weights = tf.ones([edge_feature_vector_size, 1], dtype=np.float32)
one_hot_query_vector = tf.placeholder(tf.float32, [None, 1])
featured_adjacency_matrix = tf.placeholder(
    tf.float32, [None, None, edge_feature_vector_size])
correct_answer_vector = tf.placeholder(tf.float32, [None, 1])
incorrect_answer_vector = tf.placeholder(tf.float32, [None, 1])

with tf.name_scope('weights'):
    feature_weights = tf.Variable(feature_weights)


def aggregate_by_log_sum(answer_vector, pi):
    smoothing = tf.constant(np.float32('1.0e-20'))
    aggregation = tf.reduce_sum(
        tf.log(tf.multiply(answer_vector, pi) + smoothing))
    return aggregation


def get_transition_matrix():

    featured_flattened_adjacency_matrix = tf.reshape(
        featured_adjacency_matrix, [-1, edge_feature_vector_size])
    flattened_adjacency_matrix = tf.matmul(featured_flattened_adjacency_matrix,
                                           feature_weights)
    nodes_count = tf.shape(one_hot_query_vector)[0]
    transition_matrix = tf.reshape(flattened_adjacency_matrix,
                                   [nodes_count, nodes_count])

    ######
    # transition_matrix = tf.nn.relu(transition_matrix)
    ###
    transition_mask = tf.cast(transition_matrix, tf.bool)
    transition_mask = tf.cast(transition_mask, tf.int32)
    transition_mask = tf.cast(transition_mask, tf.float32)
    transition_matrix = tf.multiply(
        tf.nn.relu(transition_matrix), transition_mask)
    ######

    sum_reduced = tf.reduce_sum(transition_matrix, axis=1, keep_dims=True)
    condition = tf.greater(sum_reduced,
                           tf.zeros_like(sum_reduced, dtype=tf.float32))
    safe_sum_reduced = tf.where(
        condition, sum_reduced,
        (tf.constant(np.inf, dtype=tf.float32)) * tf.ones_like(sum_reduced))
    transition_matrix = (transition_matrix / safe_sum_reduced)
    transition_matrix = tf.transpose(transition_matrix)

    return transition_matrix


def compute_pi():

    transition_matrix = get_transition_matrix()
    pi = tf.matmul(transition_matrix, one_hot_query_vector)
    for _ in range(10):
        pi = tf.matmul(transition_matrix, pi)

    return pi


def compute_loss(pi):
    loss = aggregate_by_log_sum(incorrect_answer_vector,
                                pi) - aggregate_by_log_sum(
                                    correct_answer_vector, pi)
    return loss


with tf.name_scope('pi'):
    pi = compute_pi()

with tf.name_scope('loss'):
    instance_loss = compute_loss(pi)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer()
    gradients = optimizer.compute_gradients(instance_loss)
    train_step = optimizer.apply_gradients(gradients)

with tf.name_scope('saver'):
    saver = tf.train.Saver()

###################### (END) Tensorflow Computation Graph ###########################

print("Computation Graph Built")

###################### (START) Input Tensor Load ###########################


def get_auc(correct_scores, incorrect_scores):
    scores = incorrect_scores + correct_scores
    labels = [-1] * len(incorrect_scores) + [1] * len(correct_scores)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    return metrics.auc(fpr, tpr)


def get_mrr(correct_scores, incorrect_scores):

    retrieval = [(e, -1.0)
                 for e in incorrect_scores] + [(e, 1.0) for e in correct_scores]
    retrieval.sort(key=(lambda x: x[0]), reverse=True)
    mean_reciprocal_rank = np.mean(
        [1 / float(i + 1) for i, e in enumerate(retrieval) if e[1] == 1.0])
    return mean_reciprocal_rank


def get_correct_incorrect_scores(np_pi, np_correct_answer_vector,
                                 np_incorrect_answer_vector):

    np_pi = list(np_pi.reshape([-1]))
    np_correct_answer_vector = list(np_correct_answer_vector.reshape([-1]))
    np_incorrect_answer_vector = list(np_incorrect_answer_vector.reshape([-1]))

    correct_scores = [
        a for a, b in zip(np_pi, np_correct_answer_vector) if b > 0
    ]
    incorrect_scores = [
        a for a, b in zip(np_pi, np_incorrect_answer_vector) if b > 0
    ]

    return correct_scores, incorrect_scores


###################### (END) Input Tensor Load ###########################

if run == 'Learn':
    set_name = 'train'
else:
    set_name = 'test'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed_dict = {}

    tensors_dir = os.path.join(processed_data_dir, 'Tensors', set_name)
    tensor_file_names = glob.glob(os.path.join(tensors_dir, '*-sparse.pickle'))
    tensor_idxs = [
        int(os.path.basename(name).split('-')[0]) for name in tensor_file_names
    ]
    tensor_idxs.sort()

    for idx in tensor_idxs:

        with open(
                os.path.join(tensors_dir, '{}-sparse.pickle'.format(idx)),
                'rb') as f:
            sparse_data = pickle.load(f)
            tensor = [item.todense() for item in sparse_data]
            feed_dict[idx] = {
                one_hot_query_vector: tensor[0],
                featured_adjacency_matrix: tensor[1],
                correct_answer_vector: tensor[2],
                incorrect_answer_vector: tensor[3]
            }

    if run == 'Predict':
        print "In Predict Mode"

        model_dir = os.path.join(processed_data_dir, 'model')
        model_path = os.path.join(model_dir, 'weights.ckpt')
        saver.restore(sess, model_path)

        mrrs = []
        aucs = []
        for idx in tensor_idxs:
            train_loss, np_pi, np_correct_answer_vector, np_incorrect_answer_vector = sess.run(
                [
                    instance_loss, pi, correct_answer_vector,
                    incorrect_answer_vector
                ],
                feed_dict=feed_dict[idx])
            np_feature_weights = sess.run(feature_weights)
            correct_scores, incorrect_scores = get_correct_incorrect_scores(
                np_pi, np_correct_answer_vector, np_incorrect_answer_vector)
            mrr = get_mrr(correct_scores, incorrect_scores)
            auc = get_auc(correct_scores, incorrect_scores)
            mrrs.append(mrr)
            aucs.append(auc)

        print("Loss: {}".format(train_loss))
        # print( "Weights: {}".format( list( np.reshape( np_feature_weights, [-1] ) )) )
        print("MRR: {}".format(np.mean(mrrs)))
        print("AUC: {}".format(np.mean(aucs)))
        print('')

    if run == 'Learn':
        print "In Learn Mode"
        results = []
        best_mmr = 0.0
        best_epoch = 0
        for i in range(5):

            mrrs = []
            aucs = []
            for idx in tensor_idxs:

                train_loss, np_pi, np_correct_answer_vector, np_incorrect_answer_vector = sess.run(
                    [
                        instance_loss, pi, correct_answer_vector,
                        incorrect_answer_vector
                    ],
                    feed_dict=feed_dict[idx])
                np_feature_weights = sess.run(feature_weights)
                correct_scores, incorrect_scores = get_correct_incorrect_scores(
                    np_pi, np_correct_answer_vector, np_incorrect_answer_vector)

                mrr = get_mrr(correct_scores, incorrect_scores)
                auc = get_auc(correct_scores, incorrect_scores)
                mrrs.append(mrr)
                aucs.append(auc)

            print("Epoch: {}".format(i))
            print("Loss: {}".format(train_loss))
            # print("Weights: {}".format(list(np.reshape(np_feature_weights,[-1]))))
            mmr_mean = np.mean(mrrs)
            auc_mean = np.mean(aucs)
            print("MRR: {}".format(mmr_mean))
            print("AUC: {}".format(auc_mean))
            print('')

            for idx in tensor_idxs:
                sess.run(train_step, feed_dict=feed_dict[idx])

            if mmr_mean > best_mmr:
                best_epoch = i
                model_dir = os.path.join(processed_data_dir, 'model')
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                model_path = os.path.join(model_dir, 'weights.ckpt')
                saver.save(sess, model_path)
                best_mmr = mmr_mean

        print 'best epoch: {}'.format(best_epoch)
