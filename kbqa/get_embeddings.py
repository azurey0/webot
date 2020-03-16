import os
import re
import numpy as np

def get_sen_ent_rel_tail():
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "dataset/nlpcc-iccpol-2016.kbqa.training-data.txt")
    f = open(dir, 'r')

    entities = []
    relations = []
    tails = []
    for line in f:
        if line.startswith('<t'):
            triple = re.split('>', line)[1]
            idx = triple.find('|')
            entities.append(triple[:idx][1:-1])

            a = re.search(r'\|\|\|', triple)
            triple = triple[a.end() + 1:]
            b = re.search(r'\|\|\|', triple)
            triple = triple[:b.start() - 1]
            relations.append(triple)

            triple = re.split('>', line)[1]
            ll = triple.rfind('|')
            tails.append(triple[ll+2:])
            # print(triple[ll+2:])
    return entities, relations, tails

def embed(lst):
    from bert_serving.client import BertClient
    bc = BertClient()
    vec = bc.encode(lst)
    print('vec.shape: ', vec.shape)
    return vec


# save embeds to redis
def save():
    '''
    save embeds of entities and relations to file, save entities, relations, tiles to redis
    :return:
    '''
    import redis
    r = redis.Redis()
    r = redis.Redis(db=3)

    entities, relations, tails = get_sen_ent_rel_tail()
    print(tails)
    # entities_embed = embed(entities)
    # relations_embed = embed(relations)
    # np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/entities_embed.npy"), entities_embed)
    # np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/relations_embed.npy"), relations_embed)

    kb_db = {}
    i = 0
    for entity, relation, tail in zip(entities, relations, tails):
        key=f"kb_db:{i}"
        value = {'entity': entity,
                 'relation': relation,
                 'tail': tail}
        kb_db[key]=value
        i += 1


    with r.pipeline() as pipe:#With a pipeline, all the commands are buffered on the client side and then sent at once
        for kb_db_id, kb_db_content in kb_db.items():
            pipe.hmset(kb_db_id, kb_db_content)
        pipe.execute()
    print('successfully save to file and redis!')


if __name__ =='__main__':
    save()