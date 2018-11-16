#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module include vgg model, generate image vector logits to combine with dnn part logits"""
import six
import numpy as np
import tensorflow as tf

import os
import sys
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)
from lib.read_conf import Config

categorical_column_with_hash_bucket = tf.feature_column.categorical_column_with_hash_bucket
embedding_column = tf.feature_column.embedding_column

CONF = Config()

class BaseSequenceModel(object):
    def __init__(self):
        pass
    def build(self, feature):
        raise NotImplementedError

    def dot_production_attention(self, q, k, v, dim, masked=False):
        """Calculate relative position-aware dot-product self-attention.
        The attention calculation is augmented with learned representations for the
        relative position between each element in q and each element in k and v.
        Args:
            q: a Tensor with shape [batch, heads, length, depth].
            k: a Tensor with shape [batch, heads, length, depth].
            v: a Tensor with shape [batch, heads, length, depth].
        Returns:
            A Tensor.
        Raises:
            ValueError: if max_relative_position is not > 0.
        """
        Q = tf.layers.dense(q, dim)
        K = tf.layers.dense(key, dim)
        V = tf.layers.dense(value, dim)

        # Scaled Dot Product Attention
        with tf.variable_scope("scaled-dot-product-attention"):
            QK_T = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
            if masked:
                mask = tf.ones_like(QK_T)
                mask = tf.linalg.LinearOperatorLowerTriangular(mask, tf.float32).to_dense()
                QK_T = tf.matmul(QK_T, mask)
            attention = tf.nn.softmax(QK_T * tf.sqrt(1/dim))
            att_V = tf.matmul(attention, V)

        #output = tf.layers.dense(att_V, dim)
        output = tf.reduce_sum(att_V, axis=1)
        return output


class SequenceModel2(BaseSequenceModel):
    def __init__(self):
        super(BaseSequenceModel, self).__init__()

    def build(self, features, input_layer_partitioner = None):
        feature_conf_dic = CONF.read_feature_conf()

        conf = feature_conf_dic['f20164']
        f_type, f_tran, f_param = conf["type"], conf["transform"], conf["parameter"]
        input_op = features['f20164']

        hash_bucket_size = f_param
        embed_dim = 16 #embedding_dim(hash_bucket_size)
        seq_embedding = tf.get_variable('seq_embedding', shape=[hash_bucket_size, embed_dim], 
                initializer= tf.contrib.layers.xavier_initializer())
        embed_input = tf.nn.embedding_lookup(seq_embedding, input_op)

        conf = feature_conf_dic['f20164len']
        f_type, f_tran, f_param = conf["type"], conf["transform"], conf["parameter"]
        input_len = features['f20164len']
        print(input_len)
        with tf.Session() as sess:
            print ('abcddd')
            print(sess.run(input_len.eval()))

        mask = tf.sequence_mask(input_len, maxlen=f_param, dtype=tf.float32)
        embed_input = embed_input * tf.expand_dims(mask, axis=2)

        net = self.dot_production_attention(embed_input, embed_input, embed_input, embed_dim)
        return net



class SequenceModel(BaseSequenceModel):
    def __init__(self):
        super(BaseSequenceModel, self).__init__()

    def build(self, features, input_layer_partitioner = None):
        feature_conf_dic = CONF.read_feature_conf()


        conf = feature_conf_dic['f20164']
        f_type, f_tran, f_param = conf["type"], conf["transform"], conf["parameter"]
                
        hash_bucket_size = f_param
        embed_dim = 16 #embedding_dim(hash_bucket_size)
        col = categorical_column_with_hash_bucket('f20164',
                    hash_bucket_size=hash_bucket_size,
                    dtype=tf.string)

        columns = []
        columns.append(embedding_column(col,
                    dimension=embed_dim,
                    combiner='mean',
                    initializer=None,
                    ckpt_to_load_from=None,
                    tensor_name_in_ckpt=None,
                    max_norm=None,
                    trainable=True))

        with tf.variable_scope(
                    'input_from_feature_columns',
                    values=tuple(six.itervalues(features)),
                    partitioner=input_layer_partitioner,
                    reuse=tf.AUTO_REUSE):
            net = tf.feature_column.input_layer(features=features, feature_columns=columns)
        return net


