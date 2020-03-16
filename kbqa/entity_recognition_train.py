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

def get_relation_tag():
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/nlpcc-iccpol-2016.kbqa.training-data.txt")
    f = open(dir, 'r')
    sentences = []
    relations = []
    for line in f:
        if line.startswith('<q'):
            sentences.append(re.split('>', line)[1][1:-1])
        if line.startswith('<t'):
            triple = re.split('>',line)[1]
            a = re.search(r'\|\|\|',triple)
            triple = triple[a.end() + 1:]
            b = re.search(r'\|\|\|', triple)
            triple = triple[:b.start() - 1]
            relations.append(triple)
    # print(relations)
    tags = []
    for sen,ent in zip(sentences, relations):
        s = re.search(re.escape(ent), sen)
        if s == None:
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
from sklearn.model_selection import train_test_split
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn_crfsuite.metrics import flat_classification_report
from keras.models import load_model
from keras_contrib.utils import save_load_utils
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
import pickle
# max sequence length
# batch size
max_features = 20000
MAX_LEN = 75
bs = 32
embedding = 40
epochs = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sentences, tags = get_sen_tag()
sentences, tags = get_relation_tag()
# print(sentences[0])

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
print('tokenizing...')
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

# count the num of unique characters in all sentences
unique_char = []
for list in tokenized_texts:
    if list == '[UNK]':
        continue
    for chars in list:
        if chars not in unique_char:
            unique_char.append(chars)
pickle_out1 = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/unique_char.pickle"), "wb")
pickle.dump(unique_char, pickle_out1)
pickle_out1.close()
print('wrote to file , unique_char.pickle')

# word is key and its value is corresponding index
word_to_index = {w : i + 1 for i, w in enumerate(unique_char)}
word_to_index["PAD"] = 0

# save word dict
pickle_out = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/word_to_index.pickle"), "wb")
pickle.dump(word_to_index, pickle_out)
pickle_out.close()
print('wrote to file , word_to_index.pickle')
# Converting each sentence into list of index from list of tokens
input_ids =[]
for sen in sentences:
    s = []
    for w in sen:
        if w in word_to_index.keys():
            s.append(word_to_index[w])
        else:
            s.append(word_to_index['[UNK]'])
    input_ids.append(s)
# Padding each sequence to have same length  of each word
input_ids = pad_sequences(maxlen = MAX_LEN, sequences = input_ids, padding = "post", value = word_to_index["PAD"])
#
tags = pad_sequences(tags,
                     maxlen=MAX_LEN, value=0, padding="post",
                     dtype="long", truncating="post")
tags = [to_categorical(i, num_classes = 3) for i in tags]
print(input_ids.shape)
print(np.array(tags).shape)

X_train, X_test, y_train, y_test = train_test_split(input_ids, tags, test_size = 0.15)
# Model architecture
# input = Input(shape = (MAX_LEN,))
# model = Embedding(input_dim = len(unique_char) + 1, output_dim = embedding, input_length = MAX_LEN)(input)
# model = Bidirectional(LSTM(units = 50, return_sequences=True, recurrent_dropout=0.1))(model)
# model = TimeDistributed(Dense(50, activation="relu"))(model)
# crf = CRF(3)  # CRF layer
# out = crf(model)  # output
#
# model = Model(input, out)
# model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
#
# model.summary()

save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/relation_model.h5")
#
# checkpointer = ModelCheckpoint(filepath = save_dir,
#                        verbose = 0,
#                        mode = 'auto',
#                        save_best_only = True,
#                        monitor='val_loss')

# history = model.fit(X_train, np.array(y_train), batch_size=bs, epochs=epochs,
#                     validation_split=0.1)
# save_model = model.save(save_dir)
# print(history.history.keys())
custom_objects={'CRF': CRF,'crf_loss':crf_loss,'crf_viterbi_accuracy':crf_viterbi_accuracy}
model= load_model(save_dir, custom_objects = custom_objects)

# Evaluation
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)
y_test_true = np.argmax(y_test, -1)

print(y_pred.shape, 'y test true:', y_test_true.shape)
# print("F1-score is : {:.1%}".format(f1_score(y_test_true, y_pred)))

report = flat_classification_report(y_pred=y_pred, y_true=y_test_true)
print(report)


# At every execution model picks some random test sample from test set.
i = np.random.randint(0,X_test.shape[0]) # choose a random number between 0 and len(X_te)b
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
print('p:',p.shape)
print('X_test[i]:' ,X_test[i].shape)
true = np.argmax(y_test[i], -1)

print("Sample number {} of {} (Test Set)".format(i, X_test.shape[0]))
# Visualization
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_test[i], true, p[0]):
    if w != 0:
        print("{:15}: {:5} {}".format(unique_char[w-1], t, pred))