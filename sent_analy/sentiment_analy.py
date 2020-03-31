import pandas as pd
import numpy as np
from keras.models import load_model
import os
from bert_serving.client import BertClient

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/sent_analy_model.h5")
model = load_model(model_dir)
bc = BertClient()

def get_emotion(sentence):

    sen_list = []
    sen_list.append(sentence)
    x = bc.encode(sen_list)
    pred = model.predict(x)

    return pred[0][0], pred[0][1]

if __name__ == '__main__':
    print(get_emotion('春天到了'))
    print(get_emotion('我不开心'))
    print(get_emotion('哈哈哈哈'))