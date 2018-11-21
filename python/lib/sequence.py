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
        """
        Q = q
        K = k
        V = v
        # Scaled Dot Product Attention
        with tf.variable_scope("scaled-dot-product-attention"):
            QK_T = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
            
            if masked:
                pad_length = QK_T.get_shape().as_list()[-1]
                mask = 1 - tf.diag(tf.ones([pad_length], tf.float32))
                QK_T = QK_T * mask
            
            attention = tf.nn.softmax(QK_T * tf.sqrt(1/tf.to_float(dim)))
            att_V = tf.matmul(attention, V)
               
            '''
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                ms, qkt, qktm, at = sess.run([mask, QK_T, QK_TM, attention])
                print ('MASK-----------------------------------')
                print (ms)
                print ('QK_T-----------------------------------')
                print (qkt)
                print ('QK_TM==========================')
                print (qktm)
                print ('attention===========================')
                print (at)
            '''
        output = tf.reduce_mean(att_V, axis=1)
        return output

    def positional_encoding(x, pad_length, embed_dim):
        # First part of the PE function: sin and cos argumen
        position_enc = np.array([[pos / (1000 ** (2.0 * i / embed_dim)) for i in range(embed_dim)] for pos in range(pad_length)])
        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        # 
        output = tf.convert_to_tensor(position_enc, dtype=tf.float32)
        return output

class SequenceModel2(BaseSequenceModel):
    def __init__(self):
        super(BaseSequenceModel, self).__init__()

    def build(self, features, input_layer_partitioner = None):
        feature_conf_dic = CONF.read_feature_conf()

        # input sequence
        conf = feature_conf_dic['f20164']
        f_type, f_tran, f_param = conf["type"], conf["transform"], conf["parameter"]
        input_seq = features['f20164']
        #input_seq = tf.convert_to_tensor(features['f20164'])

        # input embedding
        hash_bucket_size = f_param
        embed_dim = 6 #embedding_dim(hash_bucket_size)
        seq_embedding = tf.get_variable('seq_embedding', shape=[hash_bucket_size, embed_dim], 
                initializer= tf.contrib.layers.xavier_initializer())
        embed_input = tf.nn.embedding_lookup(seq_embedding, input_seq)

        # input sequence length
        conf = feature_conf_dic['f20164len']
        f_type, f_tran, f_param = conf["type"], conf["transform"], conf["parameter"]
        #input_seq_len_op = features['f20164len']
        input_seq_len = tf.reshape(features['f20164len'], [-1])
       
        # positional encoding
        embed_pos = self.positional_encoding(f_param, embed_dim)
        encoding = tf.add(embed_input, embed_pos)

        # input embedding mask
        mask = tf.sequence_mask(input_seq_len, maxlen=f_param, dtype=tf.float32)
        encoding = encoding * tf.expand_dims(mask, axis=2)

        '''
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print ('-----------------------------------')
            in_op, seq_op, seq_len, seq_pe, seq_mask, seq_emb = sess.run([input_seq, features['f20164len'], input_seq_len, embed_pos, embed_input, encoding])
            print ('input-----------------------------------')
            print (in_op)
            print ('seq_op-----------------------------------')
            print (seq_op)
            print ('seq_len-----------------------------------')
            print (seq_len)
            print ('seq_pe-----------------------------------')
            print (seq_pe)
            print ('seq_mask-----------------------------------')
            print (seq_mask)
            print ('seq_emb-----------------------------------')
            print (seq_emb)
        '''

        projection = tf.layers.dense(encoding, embed_dim)
        net = self.dot_production_attention(projection, projection, encoding, embed_dim, True)
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


