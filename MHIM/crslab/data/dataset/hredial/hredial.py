# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2021/1/3, 2020/12/19
# @Author : Kun Zhou, Xiaolei Wang, Yuanhang Zhou
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, sdzyh002@gmail

r"""
ReDial
======
References:
    Li, Raymond, et al. `"Towards deep conversational recommendations."`_ in NeurIPS 2018.

.. _`"Towards deep conversational recommendations."`:
   https://papers.nips.cc/paper/2018/hash/800de15c79c8d840f4e78d3af937d4d4-Abstract.html

"""

import json
import os
import pickle as pkl
from copy import copy

from loguru import logger
from tqdm import tqdm
import spacy
import re

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
from .resources import resources


class HReDialDataset(BaseDataset):
    """

    Attributes:
        train_data: train dataset.
        valid_data: valid dataset.
        test_data: test dataset.
        vocab (dict): ::

            {
                'tok2ind': map from token to index,
                'ind2tok': map from index to token,
                'entity2id': map from entity to index,
                'id2entity': map from index to entity,
                'word2id': map from word to index,
                'vocab_size': len(self.tok2ind),
                'n_entity': max(self.entity2id.values()) + 1,
                'n_word': max(self.word2id.values()) + 1,
            }

    Notes:
        ``'unk'`` must be specified in ``'special_token_idx'`` in ``resources.py``.

    """

    def __init__(self, opt, tokenize, restore=False, save=False):
        """Specify tokenized resource and init base dataset.

        Args:
            opt (Config or dict): config for dataset or the whole system.
            tokenize (str): how to tokenize dataset.
            restore (bool): whether to restore saved dataset which has been processed. Defaults to False.
            save (bool): whether to save dataset after processing. Defaults to False.

        """
        resource = resources[tokenize]
        dpath = os.path.join(DATASET_PATH, "hredial", tokenize)
        self.whole_movies=[]
        self.entity_dictionary={}
        # 初始化一个集合用于存储已添加的键
        self.added_keys = set()
        self.review_path = "/data1/cxy/MHIM-main/MHIM/data/dataset/hredial/nltk/movieid2review_dict.pkl"
        super().__init__(opt, dpath, resource, restore, save)

    def _load_data(self):
        train_data, valid_data, test_data = self._load_raw_data()
        self._load_vocab()
        self._load_other_data()

        vocab = {
            'tok2ind': self.tok2ind,
            'ind2tok': self.ind2tok,
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'vocab_size': len(self.tok2ind),
            'n_entity': self.n_entity
        }

        return train_data, valid_data, test_data, vocab

    def _load_raw_data(self):
        # load train/valid/test data
        with open(os.path.join(self.dpath, 'train_data.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.info(f"[Load train data from {os.path.join(self.dpath, 'train_data.json')}]")
        with open(os.path.join(self.dpath, 'valid_data.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.info(f"[Load valid data from {os.path.join(self.dpath, 'valid_data.json')}]")
        with open(os.path.join(self.dpath, 'test_data.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.info(f"[Load test data from {os.path.join(self.dpath, 'test_data.json')}]")

        return train_data, valid_data, test_data

    def _load_vocab(self):
        # self.entityWord2entityAddress = json.load(open('/data/xiaoyin/MHIM-main/MHIM/data/processed_dataset/entityWord2entityAdd_dict.json', 'r'))
        self.tok2ind = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}

        logger.info(f"[Load vocab from {os.path.join(self.dpath, 'token2id.json')}]")
        logger.info(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.info(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

    def _load_other_data(self):
        # edge extension data
        self.conv2items = json.load(open(os.path.join(self.dpath, 'conv2items.json'), 'r', encoding='utf-8'))
        # dbpedia
        self.entity2id = json.load(
            open(os.path.join(self.dpath, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        self.side_data = pkl.load(open(os.path.join(self.dpath, 'side_data.pkl'), 'rb'))
        logger.info(
            f"[Load entity dictionary and KG from {os.path.join(self.dpath, 'entity2id.json')} and {os.path.join(self.dpath, 'dbpedia_subkg.json')}]")

    def _data_preprocess(self, train_data, valid_data, test_data):
        
        # processed_train_data = self._raw_data_process(train_data)
        # # 将字典写入 JSON 文件 add_review_processed_train_data_keywords.json / train_data.json
        # with open('/data1/cxy/MHIM-main/MHIM/data/processed_dataset/add_entity_v3_train_data_8000+.json', 'w') as json_file:
        #     json.dump(processed_train_data, json_file)
        # logger.info("[Finish train data process]")

        # 引入电影评论数据集的加工数据集
        # with open('/data1/cxy/MHIM-main/MHIM/data/processed_dataset/add_entity_v3_train_data.json', 'r', encoding='utf-8') as file:
        #     processed_train_data = json.load(file)
        # 原数据集
        with open('/data1/cxy/MHIM-main/MHIM/data/dataset/hredial/processed_data/train_data.json', 'r', encoding='utf-8') as file:
            processed_train_data = json.load(file)

        # processed_valid_data = self._raw_data_process(valid_data)
        # # 将字典写入 JSON 文件
        # with open('/data1/cxy/MHIM-main/MHIM/data/processed_dataset/add_entity_v3_valid_data.json', 'w') as json_file:
        #     json.dump(processed_valid_data, json_file)
        # logger.info("[Finish valid data process]")

        # 引入电影评论数据集的加工数据集
        # with open('/data1/cxy/MHIM-main/MHIM/data/processed_dataset/add_entity_v3_valid_data.json', 'r', encoding='utf-8') as file:
        #     processed_valid_data = json.load(file)
        # 原数据集
        with open('/data1/cxy/MHIM-main/MHIM/data/dataset/hredial/processed_data/valid_data.json', 'r', encoding='utf-8') as file:
            processed_valid_data = json.load(file)

        # processed_test_data = self._raw_data_process(test_data)
        # # 将字典写入 JSON 文件
        # with open('/data1/cxy/MHIM-main/MHIM/data/processed_dataset/add_entity_v3_test_data.json', 'w') as json_file:
        #     json.dump(processed_test_data, json_file)
        # logger.info("[Finish test data process]")

        # 引入电影评论数据集的加工数据集
        # with open('/data1/cxy/MHIM-main/MHIM/data/processed_dataset/add_entity_v3_test_data.json', 'r', encoding='utf-8') as file:
        #     processed_test_data = json.load(file)
        # 原数据集
        with open('/data1/cxy/MHIM-main/MHIM/data/dataset/hredial/processed_data/test_data.json', 'r', encoding='utf-8') as file:
            processed_test_data = json.load(file)

        processed_side_data = self.side_data
        logger.info("[Finish side data process]")
        return processed_train_data, processed_valid_data, processed_test_data, processed_side_data
        # return processed_train_data
        # return processed_valid_data
        # return processed_test_data

    def _raw_data_process(self, raw_data):
        # print(raw_data[0])
        augmented_convs = [[self._merge_conv_data(conv["dialog"], conv["user_id"], conv["conv_id"]) for conv in convs] for convs in tqdm(raw_data)]
        # with open('/data1/cxy/MHIM-main/MHIM/data/processed_dataset/words2entity_train8000+.json', 'w') as json_file:
        #     json.dump(self.entity_dictionary, json_file, indent=2)
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

            # # 使用空格将单词列表拼接成一句话
            # sentence = ' '.join(utt["text"])
            # if entity_ids != []:
            #     entities_in_text = self._ner_extraction(sentence)
            #     if len(entities_in_text) != 0:
            #         if not entities_in_text[0]['text'].startswith('@'):
            #             for entity_link in utt['entity']:
            #                 entity_wordlist = self._extract_entity_wordlist(entity_link)
            #                 if any(word in entities_in_text[0]['text'] for word in entity_wordlist):
            #                     # 检查键是否已经存在于集合中，如果不存在则加入字典，同时更新集合
            #                     if entities_in_text[0]['text'] not in self.added_keys:
            #                         self.entity_dictionary.update({entities_in_text[0]['text']: entity_link})
            #                         self.added_keys.add(entities_in_text[0]['text'])

            # 1. 判断对话中是否涉及到items
            if len(movie_ids)>0:
                # 2. 是的话就将text中的word拼接起来变为text
                # 使用空格将单词列表拼接成一句话
                sentence = ' '.join(utt["text"])
                # 3. 使用函数 find_same_sentiment_and_item_text(text, movie_ids)找到相同感情倾向
                #    并且提及了item的电影评论movie_comment_text
                movie_comment_text = self._find_same_sentiment_and_item_text(sentence, movie_ids)
                # 4. 提取movie_comment_text中的movie或entity，加入到movie_ids或entity_ids中
                entitiesID_in_review = self._find_movie_review_entity(movie_comment_text)
                #    使用函数 extract_keywords() 得到 movie_comment_text 的关键词 并将movie_comment_text转为id，加入text_token_ids中
                # summary_of_movie_review = self._extract_keywords(movie_comment_text,20)
                # print("summary_of_movie_review: ",summary_of_movie_review)
                # 不用关键词提取，按论文的方法截取前20个token
                # summary_of_movie_review = movie_comment_text.split()[:20]
                # movie_comment_text_id = [self.tok2ind.get(word, self.tok2ind['__unk__']) for word in summary_of_movie_review]
            else:
                entitiesID_in_review = []

            # 如果某一个role连续说了两句话，则后一句话的信息加在前一句话里
            if utt["role"] == last_role:
                # augmented_convs[-1]["text"] += text_token_ids + movie_comment_text_id
                augmented_convs[-1]["text"] += text_token_ids
                augmented_convs[-1]["movie"] += movie_ids
                if len(entitiesID_in_review)>10:
                    augmented_convs[-1]["entity"] += entity_ids + entitiesID_in_review[:10]
                else:
                    augmented_convs[-1]["entity"] += entity_ids + entitiesID_in_review
            else: # 记得这里也要改
                if len(entitiesID_in_review)>10:
                    augmented_convs.append({
                        "user_id": user_id,
                        "conv_id": conv_id,
                        "role": utt["role"],
                        "text": text_token_ids,
                        # "text": text_token_ids + movie_comment_text_id,
                        "entity": entity_ids + entitiesID_in_review[:10],
                        "movie": movie_ids
                    })
                else:
                    augmented_convs.append({
                        "user_id": user_id,
                        "conv_id": conv_id,
                        "role": utt["role"],
                        "text": text_token_ids,
                        # "text": text_token_ids + movie_comment_text_id,
                        "entity": entity_ids + entitiesID_in_review,
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
    
    def _find_movie_review_entity(self, movie_comment_text):
        # 在这里填合并后的字典地址
        with open('/data1/cxy/MHIM-main/MHIM/data/dataset/hredial/nltk/words2entityAdd_v3.json', 'r') as file:
            entity_dictionary = json.load(file)
        entitiesADD_list_in_movie_review = self._find_entity_in_sentence(movie_comment_text, entity_dictionary)

        # entitiesID_in_review=[]
        # # 使用 NLTK 进行分词
        # tokens = word_tokenize(movie_comment_text)
        # entitiesAddressList = [self.entityWord2entityAddress[token] for token in tokens if token in self.entityWord2entityAddress]
        # entitiesID_in_review = [self.entity2id[entityAdd] for entityAdd in entitiesAddressList if entityAdd in self.entity2id]
        entitiesID_in_review = [self.entity2id[entityAdd] for entityAdd in entitiesADD_list_in_movie_review if entityAdd in self.entity2id]
        return entitiesID_in_review


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
    
    def _find_entity_in_sentence(self, sentence, entity_dictionary):
        entityADD_list = []
        for key in entity_dictionary:
            if key in sentence:
                entityADD_list.append(entity_dictionary[key])
        return entityADD_list 

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

        for i in range(20):
            if conv_and_ratio[i][1] < 0.60:
                break
            extended_items.append(self.conv2items[conv_and_ratio[i][0]])

        return extended_items

    def _ner_extraction(self, text):
            # 加载spaCy英文模型
            nlp = spacy.load("en_core_web_sm")
            
            # 处理文本
            doc = nlp(text)
            
            # 提取命名实体
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char,
                    'label': ent.label_
                })
            
            return entities
    
    def _extract_entity_wordlist(self, entity_link):
    # 使用正则表达式提取链接中的部分
        match = re.search(r'<http://dbpedia.org/resource/(.*?)>', entity_link)
        if match:
            # 获取匹配到的部分，并用下划线分割成单词列表
            entity_words = match.group(1).split('_')
            return entity_words
        else:
            return None