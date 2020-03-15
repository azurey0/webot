import os
import re
from tqdm import tqdm, trange

# data prepare
def get_sen_tag():
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/nlpcc-iccpol-2016.kbqa.training-data.txt")
    f = open(dir, 'r')
    sentences = []
    entities = []
    for line in f:
        if line.startswith('<q'):
            sentences.append(re.split('>', line)[1][1:-1])
        if line.startswith('<t'):
            triple = re.split('>',line)[1]
            idx = triple.find('|')
            entities.append(triple[:idx][1:-1])
    tags = []
    for sen,ent in zip(sentences,entities):
        s = re.search(re.escape(ent), sen)
        if s==None:
            tag = [0] * len(sen)
            tags.append(tag)
            continue
        tag = [0] * len(sen)
        for i in range(s.start(), s.end()):
            tag[i] = 1
        tags.append(tag)
    return sentences, tags



import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.utils import to_categorical
import numpy as np
from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional
from keras.models import Model, Input
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint

# max sequence length
# batch size
max_features = 20000
MAX_LEN = 75
bs = 32
embedding = 40
epochs = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentences, tags = get_sen_tag()
# print(sentences[0])

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

# count the num of unique characters in all sentences
unique_char = []
for list in tokenized_texts:
    for chars in list:
        if chars not in unique_char:
            unique_char.append(chars)


# pad tokenized text to desired length
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
tags = pad_sequences(tags,
                     maxlen=MAX_LEN, value=0, padding="post",
                     dtype="long", truncating="post")
tags = [to_categorical(i, num_classes = 3) for i in tags]
print(input_ids.shape)
print(np.array(tags).shape)

# Model architecture
input = Input(shape = (MAX_LEN,))
model = Embedding(input_dim = len(unique_char) + 2, output_dim = embedding, input_length = MAX_LEN)(input)
model = Bidirectional(LSTM(units = 50, return_sequences=True, recurrent_dropout=0.1))(model)
model = TimeDistributed(Dense(50, activation="relu"))(model)
crf = CRF(3)  # CRF layer
out = crf(model)  # output

model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()

checkpointer = ModelCheckpoint(filepath = 'model.h5',
                       verbose = 0,
                       mode = 'auto',
                       save_best_only = True,
                       monitor='val_loss')

history = model.fit(input_ids, np.array(tags), batch_size=bs, epochs=epochs,
                    validation_split=0.1, callbacks=[checkpointer])