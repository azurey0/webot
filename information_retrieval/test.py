# '''
# #Trains a Bidirectional LSTM on the IMDB sentiment classification task.
# Output after 4 epochs on CPU: ~0.8146
# Time per epoch on CPU (Core i7): ~150s.
# '''
#
# from __future__ import print_function
# import numpy as np
#
# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
# from keras.datasets import imdb
#
#
# max_features = 20000
# # cut texts after this number of words
# # (among top max_features most common words)
# maxlen = 100
# batch_size = 32
#
# print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')
#
# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
# print(y_train.shape)
# print(y_test.shape)
# print(y_test.data)
# #
# model = Sequential()
# model.add(Embedding(max_features, 128, input_length=maxlen))
# model.add(Bidirectional(LSTM(64)))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
#
# # try using different optimizers and different optimizer configs
# model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
#
# print('Train...')
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=4,
#           validation_data=[x_test, y_test])
import numpy as np
from numpy import random
# q_lst = []
# x_embedding = {}
# for i in range( 50000):
#     if i>0 and i % 10000 == 0 : # save to redis every 1w data
#         print(f'generating embeddings for {i}')
#         q_vec = np.random.rand(len(q_lst),7)
#         print(q_vec.shape)
#         key = f"xembedding:{i}"
#         value = {}
#         for j in range(10000):
#             # print(q_vec[j])
#             value[i-j] = q_vec[j]
#             # print('value',value)
#         x_embedding[key] = value
#         print('x_embedding', len(x_embedding))
#         q_lst = []
#
#     q_lst.append(i)

q_vec = str(np.random.rand(100,7))
print(type(q_vec))

import struct
import redis
import numpy as np

def toRedis(r,a,n):
   """Store given Numpy array 'a' in Redis under key 'n'"""
   h, w = a.shape
   shape = struct.pack('>II',h,w)
   encoded = shape + a.tobytes()

   # Store encoded data in Redis
   r.set(n,encoded)
   return

def fromRedis(r,n):
   """Retrieve Numpy array from Redis key 'n'"""
   encoded = r.get(n)
   h, w = struct.unpack('>II',encoded[:8])
   a = np.frombuffer(encoded, dtype=np.uint16, offset=8).reshape(h,w)
   return a

# Create 80x80 numpy array to store
# a0 = np.arange(6400,dtype=np.uint16).reshape(80,80)
#
# # Redis connection
# r = redis.Redis(host='localhost', port=6379, db=0)
#
# # Store array a0 in Redis under name 'a0array'
# toRedis(r,a0,'a0array')
#
# # Retrieve from Redis
# a1 = fromRedis(r,'a0array')
#
# np.testing.assert_array_equal(a0,a1)
import os
embeddings = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/xq_embedding_50000.npy"))
# embeddings2 = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/xq_embedding_100000.npy"))
# ab = np.concatenate((embeddings1, embeddings2), axis=0)
# print(ab.shape)
for i in range(2,10):

    embeddings_load = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"dataset/xq_embedding_{50000*i}.npy"))
    embeddings = np.concatenate((embeddings, embeddings_load), axis=0)

print(embeddings.shape)
import pickle
from sklearn.neighbors import BallTree
print('constructing KDTree...')
question_tree = BallTree(embeddings)
print('finish embed process! ')
pickle_out = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/xq_BallTree.pickle"), "wb")
pickle.dump(question_tree, pickle_out)
pickle_out.close()
print('wrote to file , xq_BallTree.pickle')