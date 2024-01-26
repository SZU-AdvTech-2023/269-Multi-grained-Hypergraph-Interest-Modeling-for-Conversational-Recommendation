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
from snownlp import SnowNLP

from crslab.config import DATASET_PATH
from crslab.data.dataset.base import BaseDataset
from .resources import resources
import jieba
import langid
import nltk
from translate import Translator


class HTGReDialDataset(BaseDataset):
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
        dpath = os.path.join(DATASET_PATH, "htgredial", tokenize)
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
            logger.debug(f"[Load train data from {os.path.join(self.dpath, 'train_data.json')}]")
        with open(os.path.join(self.dpath, 'valid_data.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.debug(f"[Load valid data from {os.path.join(self.dpath, 'valid_data.json')}]")
        with open(os.path.join(self.dpath, 'test_data.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.debug(f"[Load test data from {os.path.join(self.dpath, 'test_data.json')}]")

        return train_data, valid_data, test_data

    def _load_vocab(self):
        self.tok2ind = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}

        logger.debug(f"[Load vocab from {os.path.join(self.dpath, 'token2id.json')}]")
        logger.debug(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

    def _load_other_data(self):
        # 加载中文停用词
        self.stopwords = set()
        with open('/data1/cxy/MHIM-main/MHIM/cn_stopwords.txt', 'r', encoding='utf-8') as stopword_file:
            for line in stopword_file:
                self.stopwords.add(line.strip())
        # 电影名到评论的映射表
        self.movieName2review = json.load(open(os.path.join(self.dpath, 'movieName2review.json'), 'r', encoding='utf-8'))
        # 电影名到电影描述的映射表
        self.moviesName2storyLine = json.load(open(os.path.join(self.dpath, 'moviesName2storyLine.json'), 'r', encoding='utf-8'))
        # edge extension data
        self.conv2items = json.load(open(os.path.join(self.dpath, 'conv2items.json'), 'r', encoding='utf-8'))
        # hypergraph
        self.user2hypergraph = json.load(open(os.path.join(self.dpath, 'user2hypergraph.json'), 'r', encoding='utf-8'))
        # dbpedia
        self.entity2id = json.load(
            open(os.path.join(self.dpath, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        self.side_data = pkl.load(open(os.path.join(self.dpath, 'side_data.pkl'), 'rb'))
        logger.debug(
            f"[Load entity dictionary and KG from {os.path.join(self.dpath, 'entity2id.json')} and {os.path.join(self.dpath, 'dbpedia_subkg.json')}]")

    def _data_preprocess(self, train_data, valid_data, test_data):
        processed_train_data = self._raw_data_process(train_data)
        logger.debug("[Finish train data process]")
        processed_valid_data = self._raw_data_process(valid_data)
        logger.debug("[Finish valid data process]")
        processed_test_data = self._raw_data_process(test_data)
        logger.debug("[Finish test data process]")
        processed_side_data = self.side_data
        logger.debug("[Finish side data process]")
        return processed_train_data, processed_valid_data, processed_test_data, processed_side_data

    def _raw_data_process(self, raw_data):
        augmented_convs = [[self._merge_conv_data(conv["dialog"], conv["user_id"], conv["conv_id"]) for conv in convs] for convs in tqdm(raw_data)]
        augmented_conv_dicts = []
        for conv_list in tqdm(augmented_convs):
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

            # extract hypergraph
            related_items = self.user2hypergraph[str(user_id)]

            for i in range(len(augmented_conv_list)):
                augmented_conv_list[i]['related_entities'] = related_items[:]  # related_entities[:]
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

            # sentence = ' '.join(utt["text"])
            # review_20words = self._get_first_20tokens_of_review_(sentence,movie_ids)
            # storyline_20words = self._get_first20_tokens_of_storyline(movie_ids)
            # review_20tokens = [self.tok2ind.get(word, self.tok2ind['__unk__']) for word in review_20words]
            # storyline_20tokens = [self.tok2ind.get(word, self.tok2ind['__unk__']) for word in storyline_20words]

            if utt["role"] == last_role:
                augmented_convs[-1]["text"] += text_token_ids
                # augmented_convs[-1]["text"] += text_token_ids + review_20tokens + storyline_20tokens
                augmented_convs[-1]["movie"] += movie_ids
                augmented_convs[-1]["entity"] += entity_ids
            else:
                augmented_convs.append({
                    "user_id": user_id,
                    "conv_id": conv_id,
                    "role": utt["role"],
                    "text": text_token_ids,
                    # "text": text_token_ids + review_20tokens + storyline_20tokens,
                    "entity": entity_ids,
                    "movie": movie_ids
                })
            last_role = utt["role"]

        return augmented_convs

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

        for i in range(50):
            if conv_and_ratio[i][1] < 0.05:
                break
            extended_items.append(self.conv2items[conv_and_ratio[i][0]])

        return extended_items

    def _get_first_20tokens_of_review_(self,sentence,movie_ids):
        curSentiment = self.get_sentiment(sentence)
        # 将 entity2id 字典反转，以便通过 id 获取电影名
        id2entity_dict = {v: k for k, v in self.entity2id.items()}
        text = ""
        reviews = []
        # 遍历 movie_ids
        for movie_id in movie_ids:
            # 获取电影名
            movie_name = id2entity_dict.get(movie_id)
            
            # 检查电影名是否在 movieName2review.json 的键中
            if movie_name and movie_name in self.movieName2review:
                # 获取评论列表
                reviews = self.movieName2review[movie_name]
        
        if len(reviews)>0:
            # 将具有相同情感极性的评论拼接成一个文本
            for review in reviews:
                if self.get_sentiment(review) == curSentiment:
                    text = text + " " + review
            
            # 中文分词
            words = jieba.cut(text)

            # 过滤停用词
            filtered_words = [word for word in words if word not in self.stopwords]
            if len(filtered_words)>20:
                return filtered_words[:20]
            else:
                return filtered_words
        else:
            return []
        
    def _get_first20_tokens_of_storyline(self,movie_ids):
        # moviesName2storyLine
        # 将 entity2id 字典反转，以便通过 id 获取电影名
        id2entity_dict = {v: k for k, v in self.entity2id.items()}
        storyline = ""
        # 遍历 movie_ids
        for movie_id in movie_ids:
            # 获取电影名
            movie_name = id2entity_dict.get(movie_id)
            
            # 检查电影名是否在 moviesName2storyLine.json 的键中
            if movie_name and movie_name in self.moviesName2storyLine:
                # 获取描述
                storyline = self.moviesName2storyLine[movie_name]
        
        if storyline == "":
            return []
        else:
            # 判断文本语言
            lang, _ = langid.classify(storyline)
            
            if lang == 'zh':
                # 中文分词
                words = jieba.lcut(storyline)
            else:
                # 英文翻译为中文，然后中文分词
                translator = Translator(to_lang="zh")
                translation = translator.translate(storyline)
                words = jieba.lcut(translation)

            # 过滤停用词
            filtered_words = [word for word in words if word not in self.stopwords]
            if len(filtered_words)>20:
                return filtered_words[:20]
            else:
                return filtered_words

        
    def get_sentiment(self, text):
        # 使用 SnowNLP 进行情感分析
        s = SnowNLP(text)
        
        # 获取情感极性得分（范围在0到1之间，越接近1表示正面情感，越接近0表示负面情感）
        sentiment_score = s.sentiments
        
        # 判断情感极性
        if sentiment_score > 0.6:
            sentiment = 1
        elif sentiment_score < 0.4:
            sentiment = -1
        else:
            sentiment = 0
        
        return sentiment
    
    