import redis
"""
# r = redis.Redis()
# r.mset({"Croatia": "Zagreb", "Bahamas": "Nassau"})
# print(type(r.get("Bahamas").decode("utf-8")))# convert redis return "byte" to "str"

import random

random.seed(444)
hats = {f"hat:{random.getrandbits(32)}": i for i in (
    {
        "color": "black",
        "price": 49.99,
        "style": "fitted",
        "quantity": 1000,
        "npurchased": 0,
    },
    {
        "color": "maroon",
        "price": 59.99,
        "style": "hipster",
        "quantity": 500,
        "npurchased": 0,
    },
    {
        "color": "green",
        "price": 99.99,
        "style": "baseball",
        "quantity": 200,
        "npurchased": 0,
    })
}

# r = redis.Redis(db=1)

# with r.pipeline() as pipe:#With a pipeline, all the commands are buffered on the client side and then sent at once
#     for h_id, hat in hats.items():
#         pipe.hmset(h_id, hat)
#     pipe.execute()
# print(r.bgsave())
# print(r.hgetall("hat:56854717"))
# print( r.keys() ) # Careful on a big DB. keys() is O(N)
# r.hincrby("hat:56854717", "quantity", -1) #reduce quantity by 1
# r.hget("hat:56854717", "quantity") # get quantity
# r.hincrby("hat:56854717", "npurchased", 1)# increase npurchased by 1


import json
import os

goal_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/chat_qa.json")
print(goal_dir)
print('openning file, please wait...')

with open(goal_dir, 'r', encoding='utf-8-sig') as f:
    data = json.load(f)
"""
# import pandas as pd
# import os
# dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/xiaohuangji.tsv")
# df=pd.read_csv(dir, sep='\t')
# df.columns=['question','answer']
#
# xqa = {}
# for i in range(len(df)):
#     key=f"xqa:{i}"
#     value = {'question':df.loc[i, 'question'],
#              'answer':df.loc[i, 'answer']}
#     xqa[key]=value
#     # print(qa)
#
# r = redis.Redis(db=3)
#
# with r.pipeline() as pipe:#With a pipeline, all the commands are buffered on the client side and then sent at once
#     for xqa_id, xqa_content in xqa.items():
#         pipe.hmset(xqa_id, xqa_content)
#     pipe.execute()
#
# print(r.hget("xqa:0",'answer').decode("utf-8"))
# print(e.shape)  # (67140, 768)
#
import os
import pickle
import numpy as np
from scipy import spatial

r = redis.Redis(db=3)
fir_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/cq_BallTree.pickle")
sec_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/xq_BallTree.pickle")

with open(fir_dir, "rb") as fir_f:
    fir_rawdata = fir_f.read()
fir_question_tree = pickle.loads(fir_rawdata)

with open(sec_dir, "rb") as sec_f:
    sec_rawdata = sec_f.read()
sec_question_tree = pickle.loads(sec_rawdata)

from bert_serving.client import BertClient
bc = BertClient()

class IRbot:
    def chat(sentence):
        # number of nn to return
        k = 6
        # answers idx ranked 2-5
        potential_answer_idx = {}
        # encode input sentence
        embeddings = bc.encode(sentence.split())[0].reshape(1, -1)
        # print(embeddings.shape)
        # look for the index of similar embeddings in BallTree of questions
        dist, index = fir_question_tree.query(embeddings, k)
        # # first look up cqa, if no similar answer, look up xqa
        # print(type(index[0][1:]))
        if float(dist[0][0]) > 9:
            dist, index = sec_question_tree.query(embeddings, k)
            potential_answer_idx['xqa'] = index[0][1:].tolist()
            answer = r.hget(f"xqa:{int(index[0][0])}", 'answer').decode("utf-8")
            # print(answer, potential_answer_idx)
            return answer, potential_answer_idx
        # get answer according to the index in dataset
        answer = r.hget(f"cqa:{int(index[0][0])}", 'answer').decode("utf-8")
        potential_answer_idx['cqa'] = index[0][1:].tolist()
        # print(answer,potential_answer_idx)
        return answer, potential_answer_idx

# use for local testing
if __name__ == "__main__":
    bot = IRbot()

    while True:
        sen = input('>>')
        print(bot.chat(sen))

