import os
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
sentences = ['记得……记得，你煮的菜好难吃。','啦啦啦种太阳']

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
# print(tokenized_texts)

unique_char = []
for list in tokenized_texts:
    if list == '[UNK]':
        continue
    for chars in list:
        if chars not in unique_char:
            unique_char.append(chars)
# print(unique_char)
#
word_to_index = {w : i + 1 for i, w in enumerate(unique_char)}

word_to_index["PAD"] = 0
# print(word_to_index)

input_ids =[]
for sen in sentences:
    s = []
    for w in sen:
        print(w)
        if w in word_to_index.keys():
            s.append(word_to_index[w])
        else:
            s.append(word_to_index['[UNK]'])
    input_ids.append(s)
# input_ids = [[word_to_index[w] for w in s] for s in sentences]
print(input_ids)
# #
