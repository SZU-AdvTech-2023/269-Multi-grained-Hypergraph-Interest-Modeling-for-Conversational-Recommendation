import json
import os
import pickle as pkl
from copy import copy

from loguru import logger
from tqdm import tqdm

from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import TextCollection
from nltk import Text
import string

from crslab.config import DATASET_PATH
from crslab.data.dataset.base import BaseDataset

class HReDialDataset(BaseDataset):
    def __init__(self, opt, restore=False, save=False):
        self.review_path = "/data/xiaoyin/MHIM-main/MHIM/data/dataset/hredial/nltk/movieid2review_dict.pkl"

    def _raw_data_process(self, raw_data):
        augmented_convs = [[self._merge_conv_data(conv["dialog"], conv["user_id"], conv["conv_id"]) for conv in convs] for convs in tqdm(raw_data)]
        augmented_conv_dicts = []
        with open('output.txt', 'w') as file:
            for item in self.whole_movies:
                file.write(str(item) + ',')
        for conv_list in tqdm(augmented_convs):
            # get related entities
            related_entities = list()
            for conv in conv_list[:-1]:
                current_related_entities = set()
                for uttr in conv:
                    for entity_id in uttr['entity']:
                        current_related_entities.add(entity_id)
                    for item_id in uttr['movie']:
                        current_related_entities.add(item_id)
                related_entities.append(list(current_related_entities))
            # get related items(movies)
            related_items = list()
            for conv in conv_list[:-1]:
                current_related_items = set()
                for uttr in conv:
                    for item_id in uttr['movie']:
                        current_related_items.add(item_id)
                related_items.append(list(current_related_items))
            # get related tokens
            related_tokens = [self.tok2ind['__start__']]
            for conv in conv_list[:-1]:
                for uttr in conv:
                    related_tokens += uttr['text'] + [self.tok2ind['_split_']]
            if related_tokens[-1] == self.tok2ind['_split_']:
                related_tokens.pop()
            related_tokens += [self.tok2ind['__end__']]
            if len(related_tokens) > 1024:
                related_tokens = [self.tok2ind['__start__']] + related_tokens[-1023:]
            # add related entities and items to augmented_conv_list
            user_id = conv_list[-1][-1]['user_id']
            conv_id = conv_list[-1][-1]['conv_id']
            augmented_conv_list = self._augment_and_add(conv_list[-1])
            for i in range(len(augmented_conv_list)):
                augmented_conv_list[i]['related_entities'] = related_entities[:]
                augmented_conv_list[i]['related_items'] = related_items[:]
                augmented_conv_list[i]['related_tokens'] = related_tokens[:]
                augmented_conv_list[i]['user_id'] = user_id
                augmented_conv_list[i]['conv_id'] = int(conv_id)
            # add them to augmented_conv_dicts
            augmented_conv_dicts.extend(augmented_conv_list)
        # hyper edge extension, add extended_items
        for i in tqdm(range(len(augmented_conv_dicts))):
            extended_items = self._search_extended_items(augmented_conv_dicts[i]['conv_id'], augmented_conv_dicts[i]['context_items'])
            augmented_conv_dicts[i]['extended_items'] = extended_items[:]

        return augmented_conv_dicts


    def _merge_conv_data(self, dialog, user_id, conv_id):
        augmented_convs = []
        last_role = None
        for utt in dialog:
            text_token_ids = [self.tok2ind.get(word, self.tok2ind['__unk__']) for word in utt["text"]]
            movie_ids = [self.entity2id[movie] for movie in utt['movies'] if movie in self.entity2id]
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if entity in self.entity2id]
            self.whole_movies += movie_ids
            # 1. 判断对话中是否涉及到items
            if len(movie_ids)>0:
                # 2. 是的话就将text中的word拼接起来变为text
                    # 使用空格将单词列表拼接成一句话
                    sentence = ' '.join(utt["text"])
                    # 3. 使用函数 find_same_sentiment_and_item_text(text, movie_ids)找到相同感情倾向
                    #    并且提及了item的电影评论movie_comment_text
                    movie_comment_text = self._find_same_sentiment_and_item_text(sentence, movie_ids)
                    # 4. 提取movie_comment_text中的movie或entity，加入到movie_ids或entity_ids中
                    #    使用函数 extract_keywords() 得到 movie_comment_text 的关键词 并将movie_comment_text转为id，加入text_token_ids中
                    summary_of_movie_review = self._extract_keywords(movie_comment_text,20)
                    # print("summary_of_movie_review: ",summary_of_movie_review)
                    movie_comment_text_id = [self.tok2ind.get(word, self.tok2ind['__unk__']) for word in summary_of_movie_review]
            else:
                movie_comment_text_id=[]

            # 如果某一个role连续说了两句话，则后一句话的信息加在前一句话里
            if utt["role"] == last_role:
                augmented_convs[-1]["text"] += text_token_ids + movie_comment_text_id
                augmented_convs[-1]["movie"] += movie_ids
                augmented_convs[-1]["entity"] += entity_ids
            else: # 记得这里也要改
                augmented_convs.append({
                    "user_id": user_id,
                    "conv_id": conv_id,
                    "role": utt["role"],
                    "text": text_token_ids + movie_comment_text_id,
                    "entity": entity_ids,
                    "movie": movie_ids
                })
            last_role = utt["role"]

        return augmented_convs

    def _extract_keywords(self, text, num_keywords=20):
        # 分词和去除标点符号
        words = [word.lower() for word in word_tokenize(text) if word.isalpha() and word not in string.punctuation]

        # 去除停用词
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        # 计算词频
        word_freq = FreqDist(words)

        # 创建文本集合
        text_collection = TextCollection([words])

        # 使用TF-IDF计算关键词权重
        tfidf_scores = {}
        for word in set(words):
            tf = word_freq[word] / len(words)
            idf = text_collection.idf(word)
            tfidf_scores[word] = tf * idf

        # 根据TF-IDF权重排序关键词
        sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)

        # 提取前N个关键词
        top_keywords = [keyword for keyword, score in sorted_keywords[:num_keywords]]

        return top_keywords



    def _find_same_sentiment_and_item_text(self, text, movie_ids):
        # 会话中某句话text，及其提到的 movie_ids
        # review_path:外部电影评论数据，形如 [('1', ['a', 'b']), ('5', ['c', 'b']), ('2', ['a', 'b'])]
        # 每个元组包括电影及其评论列表
        with open(self.review_path, 'rb') as file:
            data = pkl.load(file)
            review_items = list(data.items())
            # data_keys：外部电影评论数据中的所有电影id（其实是token去掉@）
            data_keys = list(data.keys())
            # 通过token2id.json转为movie id
            # 在data_keys中的每个元素前面加上"@"
            data_keys_with_at = ['@' + key for key in data_keys]
            # 创建一个字典，将tok2ind中每个元组的第一个值作为键，第二个值作为值
            tok2ind_dict = dict(self.tok2ind)
            # 对处理后的data_keys列表进行比对，将匹配到的值存入新的列表，没有匹配到的添加空值
            # review_movie_token2id 外部电影评论数据的所有电影id
            review_movie_token2id = [tok2ind_dict.get(key, None) for key in data_keys_with_at]

        extended_sentiment_text=""
        # 还没加评论中的entity
        extended_sentiment_entities=[]
        current_text_sentiment = self._sentiment_of_text(text)
        for mov_id in movie_ids:
            # print(mov_id)
            # text中提到的电影在评论中出现了，并且情感极性一样
            if mov_id in review_movie_token2id:
                # print("yes!")
                review_list = review_items[review_movie_token2id.index(mov_id)][1]
                # print(review_list)
                for i in review_list:
                    if self._sentiment_of_text(i) == current_text_sentiment:
                        extended_sentiment_text += i
        # print(extended_sentiment_text)
        return extended_sentiment_text


    def _sentiment_of_text(self, text):
        analysis = TextBlob(text)
        # 获取文本的极性
        # 极性是一个范围在[-1, 1]之间的浮点数，其中-1表示负面情感，1表示正面情感，0表示中性。
        sentiment_polarity = analysis.sentiment.polarity
        
        # 根据极性判断情感倾向
        if sentiment_polarity > 0:
            sentiment = 1
        elif sentiment_polarity < 0:
            sentiment = -1
        else:
            sentiment = 0
        return sentiment

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context_tokens, context_entities, context_items = [], [], []
        entity_set = set()
        for i, conv in enumerate(raw_conv_dict):
            text_tokens, entities, movies = conv["text"], conv["entity"], conv["movie"]
            if len(context_tokens) > 0:
                conv_dict = {
                    "role": conv['role'],
                    "context_tokens": copy(context_tokens),
                    "response": text_tokens,
                    "context_entities": copy(context_entities),
                    "context_items": copy(context_items),
                    "items": movies,
                }
                augmented_conv_dicts.append(conv_dict)

            context_tokens.append(text_tokens)
            context_items += movies
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)

        return augmented_conv_dicts

    def _search_extended_items(self, conv_id, context_items):
        if len(context_items) == 0:
            return []
        context_items = set(context_items)
        conv_and_ratio = list()
        for conv in self.conv2items:
            if int(conv) == conv_id:
                continue
            ratio = len(set(self.conv2items[conv]) & context_items) / len(self.conv2items[conv])
            conv_and_ratio.append((conv, ratio))
        conv_and_ratio = sorted(conv_and_ratio, key=lambda x: x[1], reverse=True)
        extended_items = list()

        for i in range(10):
            if conv_and_ratio[i][1] < 0.60:
                break
            extended_items.append(self.conv2items[conv_and_ratio[i][0]])

        return extended_items
