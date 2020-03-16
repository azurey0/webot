import os
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras_contrib.utils import save_load_utils
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras_contrib.layers import CRF
import numpy as np
import pickle



# x = np.ones((1, 75))
# i = 5
# print(np.reshape(x, (75, )).shape)
def get_pred(input_ids, model):
    '''
    :param input_ids: sentences convert to id
    :param model: model name, "dataset/relation_model.h5" for relation or "dataset/model.h5" for entity
    :return: predicted str in sentence
    '''
    unique_char = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/unique_char.pickle")
    with open(unique_char, "rb") as fir2_f:
        fir2_rawdata = fir2_f.read()
    unique_char = pickle.loads(fir2_rawdata)

    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), model)
    custom_objects = {'CRF': CRF, 'crf_loss': crf_loss, 'crf_viterbi_accuracy': crf_viterbi_accuracy}
    model = load_model(model_dir, custom_objects=custom_objects)

    y_pred = model.predict(input_ids)
    y_pred = np.argmax(y_pred, axis=-1)

    pred_str = ''
    for w, pred in zip(np.reshape(input_ids, (75,)), y_pred[0]):
        if w != 0:
            print("{:15}: {:5}".format(unique_char[w - 1], pred))
            if pred != 0:
                pred_str += str(unique_char[w - 1])
    return pred_str

def get_ent_rel(sentence):
    '''
    :param sentence:  sentence, str
    :return: entity, relation identified in sentence, string
    '''
    MAX_LEN = 75
    sentences = [sentence]
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    print('tokenizing..')
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    word_dict_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/word_to_index.pickle")
    with open(word_dict_dir, "rb") as fir_f:
        fir_rawdata = fir_f.read()
    word_to_index = pickle.loads(fir_rawdata)

    # Converting each sentence into list of index from list of tokens
    input_ids = []
    for sen in sentences:
        s = []
        for w in sen:
            if w in word_to_index.keys():
                s.append(word_to_index[w])
            else:
                s.append(word_to_index['[UNK]'])
        input_ids.append(s)
    # Padding each sequence to have same length  of each word
    input_ids = pad_sequences(maxlen=MAX_LEN, sequences=input_ids, padding="post", value=word_to_index["PAD"])
    print(input_ids.shape)

    # relation model
    relation = get_pred(input_ids, "dataset/relation_model.h5")
    # entity model
    entity = get_pred(input_ids, "dataset/model.h5")
    return entity, relation

if __name__ == '__main__':
    print(get_ent_rel('你知道2月1日的后一天是哪一天？'))