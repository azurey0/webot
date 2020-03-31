from information_retrieval.get_answer import IRbot
import queue
import redis
from sent_analy.sentiment_analy import get_emotion

r = redis.Redis(db=3)

ENABLE_SENTIMENT_ANALYSIS = True

class Dialogue:
    bot_history = queue.Queue(4)
    user_history = queue.Queue(4)

    # def global_manager(message):
    def global_manager(self, message):

        if ENABLE_SENTIMENT_ANALYSIS == True:
            neg_score, pos_score = get_emotion(message)

        ans = self.general_chat(message)
        return ans

    def general_chat(self, message):
    # def general_chat(self, message):
        ans, potential_answer_idx = IRbot.chat(message)
        bot_history_lst = [elem for elem in list(Dialogue.bot_history.queue)]
        if ans not in bot_history_lst:
            if Dialogue.bot_history.full():
                Dialogue.bot_history.get()
            Dialogue.bot_history.put(ans)
            return ans
        else:
            if 'xqa' in potential_answer_idx.keys():
                dic = 'xqa'
            else:
                dic = 'cqa'
            for i in range(len(potential_answer_idx[dic])):
                idx = potential_answer_idx[dic][i]
                potential_answer = r.hget(f"{dic}:{idx}", 'answer').decode("utf-8")
                if potential_answer not in bot_history_lst:
                    if Dialogue.bot_history.full():
                        Dialogue.bot_history.get()
                    Dialogue.bot_history.put(potential_answer)
                    return potential_answer


# if __name__ == "__main__":
#     d = Dialogue()
#     # print(d.general_chat('你好啊'))
#
#     while True:
#         sen = input('>>')
#         print(d.global_manager(sen))
