import pandas as pd
import os
import torch
import pickle
from pytorch_pretrained_bert import BertTokenizer
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, TimeDistributed
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.callbacks import EarlyStopping
from keras_contrib.layers import CRF
from keras.models import Model, Input
from bert_serving.client import BertClient
from keras.initializers import Constant
from keras.models import load_model

save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/sent_analy_model.h5")
bc = BertClient()
'''
# generate embeddings

df = pd.read_csv('dataset/weibo_senti_100k.csv')
# df = df.iloc[150000:]
# df = df.sample(n=500)
print(df.info())
tags = df['label'].values
sentences = df['review'].values
sen = df['review'].values.tolist()

q_lst = []
for i in range(len(sen)):
    if i > 0 and i % 20000 == 0:
        print(f'generating embeddings for {i}')
        q_vec = bc.encode(q_lst)
        print('q_vec.shape: ', q_vec.shape)
        print(f'sucess getting embeddings for {i}')
        np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"dataset/sent_embedding_{i}.npy"), q_vec)
        print(f'wrote to file {i}')
        q_lst = []

    q_lst.append(str(sen[i]))


# construct train matrix X
embeddings = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"dataset/sent_embedding_20000.npy"))
for i in range(2, 6):
    embeddings_to_concat = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), f"dataset/sent_embedding_{i * 20000}.npy"))
    embeddings = np.concatenate((embeddings, embeddings_to_concat), axis=0)
# print(embeddings.shape)
X = embeddings

# Y
df = pd.read_csv('dataset/weibo_senti_100k.csv')
# df = df.iloc[150000:]
# df = df.sample(n=500)
print(df.info())
tags = df['label'].values
tags = tags[:100000]
Y = pd.get_dummies(tags).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

# model structure
model = Sequential()
model.add(Dense(2, activation='sigmoid', input_shape=(768,)))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

epochs = 10
batch_size = 16

# train
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


save_model = model.save(save_dir)

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
'''
'''
# some test samples
model= load_model(save_dir)
print('load model')
x = ['哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈嗝']
w = ['我好难过']
x = bc.encode(x)
w = bc.encode(w)
print('encode')
pred = model.predict(x)
pred1 = model.predict(w)
print(pred, pred1)
'''