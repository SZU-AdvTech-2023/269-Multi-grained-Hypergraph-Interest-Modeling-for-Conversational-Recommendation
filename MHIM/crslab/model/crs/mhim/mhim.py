# -*- encoding: utf-8 -*-
# @Time    :   2021/5/26
# @Author  :   Chenzhan Shang
# @email   :   czshang@outlook.com

r"""
PCR
====
References:
    Chen, Qibin, et al. `"Towards Knowledge-Based Recommender Dialog System."`_ in EMNLP 2019.

.. _`"Towards Knowledge-Based Recommender Dialog System."`:
   https://www.aclweb.org/anthology/D19-1189/

"""

import json
import os.path
import random

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from tqdm import tqdm
from torch_geometric.nn import RGCNConv, HypergraphConv

from crslab.config import DATASET_PATH
from crslab.model.base import BaseModel
from crslab.model.crs.mhim.attention import MHItemAttention
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.attention import SelfAttentionBatch
from crslab.model.utils.modules.transformer import TransformerEncoder
from crslab.model.crs.mhim.decoder import TransformerDecoderKG


class MHIMModel(BaseModel):
    """

    Attributes:
        vocab_size: A integer indicating the vocabulary size.
        pad_token_idx: A integer indicating the id of padding token.
        start_token_idx: A integer indicating the id of start token.
        end_token_idx: A integer indicating the id of end token.
        token_emb_dim: A integer indicating the dimension of token embedding layer.
        pretrain_embedding: A string indicating the path of pretrained embedding.
        n_entity: A integer indicating the number of entities.
        n_relation: A integer indicating the number of relation in KG.
        num_bases: A integer indicating the number of bases.
        kg_emb_dim: A integer indicating the dimension of kg embedding.
        user_emb_dim: A integer indicating the dimension of user embedding.
        n_heads: A integer indicating the number of heads.
        n_layers: A integer indicating the number of layer.
        ffn_size: A integer indicating the size of ffn hidden.
        dropout: A float indicating the dropout rate.
        attention_dropout: A integer indicating the dropout rate of attention layer.
        relu_dropout: A integer indicating the dropout rate of relu layer.
        learn_positional_embeddings: A boolean indicating if we learn the positional embedding.
        embeddings_scale: A boolean indicating if we use the embeddings scale.
        reduction: A boolean indicating if we use the reduction.
        n_positions: A integer indicating the number of position.
        longest_label: A integer indicating the longest length for response generation.
        user_proj_dim: A integer indicating dim to project for user embedding.

        vocab_size: 一个表示词汇表大小的整数。
        pad_token_idx: 一个表示填充令牌的ID的整数。
        start_token_idx: 一个表示起始令牌的ID的整数。
        end_token_idx: 一个表示结束令牌的ID的整数。
        token_emb_dim: 一个表示令牌嵌入层维度的整数。
        pretrain_embedding: 一个表示预训练嵌入路径的字符串。
        n_entity: 一个表示实体数量的整数。
        n_relation: 一个表示知识图中关系数量的整数。
        num_bases: 一个表示基数数量的整数。
        kg_emb_dim: 一个表示知识图嵌入维度的整数。
        user_emb_dim: 一个表示用户嵌入维度的整数。
        n_heads: 一个表示头的数量的整数。
        n_layers: 一个表示层数的整数。
        ffn_size: 一个表示FFN隐藏层大小的整数。
        dropout: 一个表示丢弃率的浮点数。
        attention_dropout: 一个表示注意力层丢弃率的整数。
        relu_dropout: 一个表示ReLU层丢弃率的整数。
        learn_positional_embeddings: 一个布尔值，表示是否学习位置嵌入。
        embeddings_scale: 一个布尔值，表示是否使用嵌入缩放。
        reduction: 一个布尔值，表示是否使用缩减。
        n_positions: 一个表示位置数量的整数。
        longest_label: 一个表示生成响应的最大长度的整数。
        user_proj_dim: 一个表示用户嵌入投影维度的整数。


    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.device = device
        self.gpu = opt.get("gpu", -1)
        self.dataset = opt.get("dataset", None)
        assert self.dataset in ['HReDial', 'HTGReDial']
        # vocab
        self.pad_token_idx = vocab['tok2ind']['__pad__']
        self.start_token_idx = vocab['tok2ind']['__start__']
        self.end_token_idx = vocab['tok2ind']['__end__']
        self.vocab_size = vocab['vocab_size']
        self.token_emb_dim = opt.get('token_emb_dim', 300)
        self.pretrain_embedding = side_data.get('embedding', None)
        # kg
        self.n_entity = vocab['n_entity']
        self.entity_kg = side_data['entity_kg']
        self.n_relation = self.entity_kg['n_relation']
        self.edge_idx, self.edge_type = edge_to_pyg_format(self.entity_kg['edge'], 'RGCN')
        self.edge_idx = self.edge_idx.to(device)
        self.edge_type = self.edge_type.to(device)
        self.num_bases = opt.get('num_bases', 8)
        self.kg_emb_dim = opt.get('kg_emb_dim', 300)
        self.user_emb_dim = self.kg_emb_dim
        # transformer
        self.n_heads = opt.get('n_heads', 2)
        self.n_layers = opt.get('n_layers', 2)
        self.ffn_size = opt.get('ffn_size', 300)
        self.dropout = opt.get('dropout', 0.1)
        self.attention_dropout = opt.get('attention_dropout', 0.0)
        self.relu_dropout = opt.get('relu_dropout', 0.1)
        self.embeddings_scale = opt.get('embedding_scale', True)
        self.learn_positional_embeddings = opt.get('learn_positional_embeddings', False)
        self.reduction = opt.get('reduction', False)
        self.n_positions = opt.get('n_positions', 1024)
        self.longest_label = opt.get('longest_label', 30)
        self.user_proj_dim = opt.get('user_proj_dim', 512)
        # pooling
        self.pooling = opt.get('pooling', None)
        assert self.pooling == 'Attn' or self.pooling == 'Mean'
        # MHA
        self.mha_n_heads = opt.get('mha_n_heads', 4)
        self.extension_strategy = opt.get('extension_strategy', None)
        self.pretrain = opt.get('pretrain', False)
        self.pretrain_data = None
        self.pretrain_epoch = opt.get('pretrain_epoch', 9999)

        super(MHIMModel, self).__init__(opt, device)

    def build_model(self, *args, **kwargs):
        if self.pretrain:
            pretrain_file = os.path.join('pretrain', self.dataset, str(self.pretrain_epoch) + '-epoch.pth')
            self.pretrain_data = torch.load(pretrain_file, map_location=torch.device('cuda:' + str(self.gpu[0])))
            logger.info(f"[Load Pretrain Weights from {pretrain_file}]")
        self._build_copy_mask()
        self._build_adjacent_matrix()
        self._build_embedding()
        self._build_kg_layer()
        self._build_recommendation_layer()
        self._build_conversation_layer()

    def _build_copy_mask(self):
        # 定义 token 文件的路径
        token_filename = os.path.join(DATASET_PATH, "hredial", "nltk", "token2id.json")
        # 打开 token 文件
        token_file = open(token_filename, 'r')       
        # 从文件中加载 token 到 id 的映射关系
        token2id = json.load(token_file)       
        # 创建 id 到 token 的映射关系
        id2token = {token2id[token]: token for token in token2id}       
        # 初始化一个用于指示是否进行复制的掩码列表
        self.copy_mask = list()       
        # 遍历所有的 id
        for i in range(len(id2token)):
            # 获取对应的 token
            token = id2token[i]           
            # 如果 token 以 '@' 开头，表示可以进行复制
            if token[0] == '@':
                self.copy_mask.append(True)
            else:
                # 否则不能进行复制
                self.copy_mask.append(False)        
        # 将掩码列表转换为 PyTorch 的张量，并将其移到指定的设备上
        self.copy_mask = torch.as_tensor(self.copy_mask).to(self.device)


    def _build_adjacent_matrix(self):
        # 创建一个空字典用于表示图的结构，其中键为实体头部，值为与之相连的尾部列表
        graph = dict()

        # 将起始实体 head 作为键，目标实体 tail 添加到以 head 为键的值中。如果 head 已经在 graph 字典中，
        # 就将 tail 添加到现有的值列表中；如果 head 不在 graph 字典中，则创建一个新的键值对，值是包含 tail 的列表。
        for head, tail, relation in tqdm(self.entity_kg['edge']):
            graph[head] = graph.get(head, []) + [tail]
        
        # graph： 对于每一个头实体作为键，遍历与它连接的邻居实体并作为值
        # {
        #     entity1: {neighbor1, neighbor2, ...},
        #     entity2: {neighbor3, neighbor4, ...},
        #     ...
        # }
        # graph示例
        # {
        #     0: {1, 2},  # 实体索引为 0 对应 "<http://dbpedia.org/resource/Agua_Dulce,_California>"
        #     1: {2},     # 实体索引为 1 对应 "<http://dbpedia.org/resource/Ganesh_(actor)>"
        #     2: {},      # 实体索引为 2 对应 "<http://dbpedia.org/resource/The_Absent-Minded_Professor>"
        # }


        # 创建一个空字典用于表示邻接矩阵
        adj = dict()

        # 遍历所有实体
        for entity in tqdm(range(self.n_entity)):
            adj[entity] = set()  # 初始化实体的邻接集合

            # 如果实体不在图中，跳过
            if entity not in graph:
                continue

            last_hop = {entity}  # 初始化上一跳的实体集合

            # 遍历一定次数（这里是1次）其实就是包括几跳邻居
            for _ in range(1):
                buffer = set()  # 用于存储当前跳的实体集合
                for source in last_hop:
                    adj[entity].update(graph[source])  # 将与上一跳实体相连的实体添加到邻接集合中
                    buffer.update(graph[source])  # 将与上一跳实体相连的实体添加到缓冲集合中
                last_hop = buffer  # 更新上一跳的实体集合

        # 将邻接矩阵设置为类的属性
        self.adj = adj
        
        # 打印日志，表示邻接矩阵已构建完成
        logger.info(f"[Adjacent Matrix built.]")


    def _build_embedding(self):
        # 如果预训练的嵌入矩阵存在，则使用预训练的嵌入矩阵创建 Token 嵌入层
        if self.pretrain_embedding is not None:
            # 使用预训练的嵌入矩阵创建 Token 嵌入层，并设置为可更新
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrain_embedding, dtype=torch.float), freeze=False,
                padding_idx=self.pad_token_idx)
        else:
            # 如果没有预训练的嵌入矩阵，则随机初始化 Token 嵌入层
            self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
            # 使用正态分布初始化权重，mean=0，std=kg_emb_dim ** -0.5
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
            # 将 padding token 的嵌入向量设置为零
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)

        # 创建实体的嵌入层，并随机初始化权重，mean=0，std=kg_emb_dim ** -0.5
        self.kg_embedding = nn.Embedding(self.n_entity, self.kg_emb_dim, 0)
        nn.init.normal_(self.kg_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        # 将零实体的嵌入向量设置为零
        nn.init.constant_(self.kg_embedding.weight[0], 0)

        # 打印调试信息
        logger.debug('[Build embedding]')


    def _build_kg_layer(self):
        # 创建知识图编码器（graph encoder），使用 RGCNConv 进行关系图卷积
        self.kg_encoder = RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)

        # 如果有预训练权重，加载预训练的知识图编码器权重
        if self.pretrain:
            self.kg_encoder.load_state_dict(self.pretrain_data['encoder'])

        # 创建超图卷积层，用于处理超图结构中的实体和关系
        self.hyper_conv_session = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
        self.hyper_conv_knowledge = HypergraphConv(self.kg_emb_dim, self.kg_emb_dim)

        # 创建注意力机制类型，这里使用了 MHItemAttention
        self.item_attn = MHItemAttention(self.kg_emb_dim, self.mha_n_heads)

        # 如果池化方式是 'Attn'，创建自注意力池化层
        if self.pooling == 'Attn':
            self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
            self.kg_attn_his = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)

        # 打印调试信息
        logger.debug('[Build kg layer]')


    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()
        logger.debug('[Build recommendation layer]')

    def _build_conversation_layer(self):
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))
        self.entity_to_token = nn.Linear(self.kg_emb_dim, self.token_emb_dim, bias=True)
        self.related_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions
        )
        self.context_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions
        )
        self.decoder = TransformerDecoderKG(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.embeddings_scale,
            self.learn_positional_embeddings,
            self.pad_token_idx,
            self.n_positions
        )
        self.user_proj_1 = nn.Linear(self.user_emb_dim, self.user_proj_dim)
        self.user_proj_2 = nn.Linear(self.user_proj_dim, self.vocab_size)
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)

        self.copy_proj_1 = nn.Linear(2 * self.token_emb_dim, self.token_emb_dim)
        self.copy_proj_2 = nn.Linear(self.token_emb_dim, self.vocab_size)
        logger.debug('[Build conversation layer]')

    def _get_session_hypergraph(self, session_related_entities):
        # 初始化超图的节点、边以及边的计数器
        hypergraph_nodes, hypergraph_edges, hyper_edge_counter = list(), list(), 0
        
        # 遍历每个 session 中的相关实体
        for related_entities in session_related_entities:
            # 如果相关实体列表为空，则跳过
            if len(related_entities) == 0:
                continue
            # 将相关实体添加到超图节点列表中
            hypergraph_nodes += related_entities
            # 为每个相关实体分配一个边，将边的标识添加到超图边列表中
            hypergraph_edges += [hyper_edge_counter] * len(related_entities)
            # 更新边的计数器
            hyper_edge_counter += 1
        
        # 构建超图的边索引张量
        hyper_edge_index = torch.tensor([hypergraph_nodes, hypergraph_edges], device=self.device)
        
        # 返回超图节点和边索引
        return list(set(hypergraph_nodes)), hyper_edge_index


    def _get_knowledge_hypergraph(self, session_related_items):
        # 将 session_related_items 中的所有相关项放入一个集合中
        related_items_set = set()
        for related_items in session_related_items:
            related_items_set.update(related_items)
        # 将集合转为列表，即为 session_related_items
        session_related_items = list(related_items_set)

        # 初始化超图的节点、边以及边的计数器
        hypergraph_nodes, hypergraph_edges, hyper_edge_counter = list(), list(), 0

        # 遍历 session_related_items 中的每个项
        for item in session_related_items:
            # 将当前项添加到超图节点列表中
            hypergraph_nodes.append(item)
            # 为当前项分配一个边，将边的标识添加到超图边列表中
            hypergraph_edges.append(hyper_edge_counter)
            # 获取当前项的邻居（在知识图中与当前项相连的其他实体）
            neighbors = list(self.adj[item])
            # 将邻居添加到超图节点列表中
            hypergraph_nodes += neighbors
            # 将边的标识添加到超图边列表中，与邻居的数量相同
            hypergraph_edges += [hyper_edge_counter] * len(neighbors)
            # 更新边的计数器
            hyper_edge_counter += 1

        # 构建超图的边索引张量
        hyper_edge_index = torch.tensor([hypergraph_nodes, hypergraph_edges], device=self.device)

        # 返回超图节点和边索引
        return list(set(hypergraph_nodes)), hyper_edge_index


    def _get_knowledge_embedding(self, hypergraph_items, raw_knowledge_embedding):
        # 初始化存储知识图嵌入的列表
        knowledge_embedding_list = []

        # 遍历超图的节点
        for item in hypergraph_items:
            # 获取当前节点及其邻居的子图
            sub_graph = [item] + list(self.adj[item])
            # 从原始的知识图嵌入中提取子图的嵌入
            sub_graph_embedding = raw_knowledge_embedding[sub_graph]
            # 对子图嵌入进行平均，得到当前节点的嵌入
            sub_graph_embedding = torch.mean(sub_graph_embedding, dim=0)
            # 将当前节点的嵌入添加到列表中
            knowledge_embedding_list.append(sub_graph_embedding)

        # 将节点的嵌入列表堆叠为张量
        return torch.stack(knowledge_embedding_list, dim=0)


    @staticmethod
    def flatten(inputs):
        outputs = set()
        for li in inputs:
            for i in li:
                outputs.add(i)
        return list(outputs)

    def _attention_and_gating(self, session_embedding, knowledge_embedding, context_embedding):
        related_embedding = torch.cat((session_embedding, knowledge_embedding), dim=0)
        if context_embedding is None:
            if self.pooling == 'Attn':
                user_repr = self.kg_attn_his(related_embedding)
            else:
                assert self.pooling == 'Mean'
                user_repr = torch.mean(related_embedding, dim=0)
        elif self.pooling == 'Attn':
            attentive_related_embedding = self.item_attn(related_embedding, context_embedding)
            user_repr = self.kg_attn_his(attentive_related_embedding)
            user_repr = torch.unsqueeze(user_repr, dim=0)
            user_repr = torch.cat((context_embedding, user_repr), dim=0)
            user_repr = self.kg_attn(user_repr)
        else:
            assert self.pooling == 'Mean'
            attentive_related_embedding = self.item_attn(related_embedding, context_embedding)
            user_repr = torch.mean(attentive_related_embedding, dim=0)
            user_repr = torch.unsqueeze(user_repr, dim=0)
            user_repr = torch.cat((context_embedding, user_repr), dim=0)
            user_repr = torch.mean(user_repr, dim=0)
        return user_repr

    def encode_user(self, batch_related_entities, batch_related_items, batch_context_entities, kg_embedding):
        # 初始化一个空列表，用于存储每个批次的用户表示
        user_repr_list = []
        
        # 遍历相关实体、物品和上下文实体的批次
        for session_related_entities, session_related_items, context_entities in zip(batch_related_entities, batch_related_items, batch_context_entities):
            # 将session_related_items扁平化以简化处理
            flattened_session_related_items = self.flatten(session_related_items)

            # 冷启动: 处理没有与会话相关的物品的情况
            if len(flattened_session_related_items) == 0:
                if len(context_entities) == 0:
                    # 如果没有上下文实体存在，则将用户表示初始化为零向量
                    user_repr = torch.zeros(self.user_emb_dim, device=self.device)
                elif self.pooling == 'Attn':
                    # 如果使用注意力池化，则对知识图嵌入应用注意力机制
                    user_repr = kg_embedding[context_entities]
                    user_repr = self.kg_attn(user_repr)
                else:
                    # 如果不使用注意力池化，则默认使用平均池化
                    assert self.pooling == 'Mean'
                    user_repr = kg_embedding[context_entities]
                    user_repr = torch.mean(user_repr, dim=0)
                user_repr_list.append(user_repr)
                continue

            # 从与会话相关的物品中提取超图信息
            hypergraph_items, session_hyper_edge_index = self._get_session_hypergraph(session_related_items)
            
            # 在知识图中对与会话相关的物品应用超图卷积
            session_embedding = self.hyper_conv_session(kg_embedding, session_hyper_edge_index)
            session_embedding = session_embedding[hypergraph_items]
            
            # 从知识图中提取超图信息
            _, knowledge_hyper_edge_index = self._get_knowledge_hypergraph(session_related_items)
            raw_knowledge_embedding = self.hyper_conv_knowledge(kg_embedding, knowledge_hyper_edge_index)
            
            # 基于超图项提取知识嵌入
            knowledge_embedding = self._get_knowledge_embedding(hypergraph_items, raw_knowledge_embedding)
            
            # 使用注意力和门控机制计算用户表示
            if len(context_entities) == 0:
                user_repr = self._attention_and_gating(session_embedding, knowledge_embedding, None)
            else:
                context_embedding = kg_embedding[context_entities]
                user_repr = self._attention_and_gating(session_embedding, knowledge_embedding, context_embedding)
            
            # 将计算得到的用户表示添加到列表中
            user_repr_list.append(user_repr)
        
        # 沿着批次维度堆叠用户表示并返回结果
        return torch.stack(user_repr_list, dim=0)


    def recommend(self, batch, mode):
        # 从批次中获取相关实体、相关物品、上下文实体和目标物品
        related_entities, related_items = batch['related_entities'], batch['related_items']
        context_entities, item = batch['context_entities'], batch['item']
        
        # 获取知识图嵌入，使用知识图编码器对知识图嵌入进行处理
        kg_embedding = self.kg_encoder(self.kg_embedding.weight, self.edge_idx, self.edge_type)  # (n_entity, emb_dim)
        
        # 获取扩展物品
        extended_items = batch['extended_items']
        
        # 对每个批次中的相关物品进行处理
        for i in range(len(related_items)):
            # 根据扩展策略截断或添加扩展物品到相关物品中
            truncate = min(int(max(2, int(len(related_items[i]) / 4))), len(extended_items[i]))
            if self.extension_strategy == 'Adaptive':
                # 自适应策略：截断或添加扩展物品
                related_items[i] = related_items[i] + extended_items[i][:truncate]
            else:
                # 默认为随机策略：随机选择一部分扩展物品添加到相关物品中
                assert self.extension_strategy == 'Random'
                extended_items_sample = random.sample(extended_items[i], truncate)
                related_items[i] = related_items[i] + extended_items_sample

        # 编码用户，得到用户嵌入
        user_embedding = self.encode_user(
            related_entities,
            related_items,
            context_entities,
            kg_embedding
        )  # (batch_size, emb_dim)
        
        # 计算用户与知识图中各实体的分数
        scores = F.linear(user_embedding, kg_embedding, self.rec_bias.bias)  # (batch_size, n_entity)
        
        # 计算推荐损失，用于训练模型
        loss = self.rec_loss(scores, item)
        
        # 返回损失和分数
        return loss, scores


    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)

    def freeze_parameters(self):
        freeze_models = [
            self.kg_embedding,
            self.kg_encoder,
            self.hyper_conv_session,
            self.hyper_conv_knowledge,
            self.item_attn,
            self.rec_bias
        ]
        if self.pooling == "Attn":
            freeze_models.append(self.kg_attn)
            freeze_models.append(self.kg_attn_his)
        for model in freeze_models:
            for p in model.parameters():
                p.requires_grad = False

    def encode_session(self, batch_related_items, batch_context_entities, kg_embedding):
        """
            Return: session_repr (batch_size, batch_seq_len, token_emb_dim), mask (batch_size, batch_seq_len)
        """
        session_repr_list = []
        for session_related_items, context_entities in zip(batch_related_items, batch_context_entities):
            flattened_session_related_items = self.flatten(session_related_items)

            # COLD START
            if len(flattened_session_related_items) == 0:
                if len(context_entities) == 0:
                    session_repr_list.append(None)
                else:
                    session_repr = kg_embedding[context_entities]
                    session_repr_list.append(session_repr)
                continue

            # TOTAL
            hypergraph_items, session_hyper_edge_index = self._get_session_hypergraph(session_related_items)
            session_embedding = self.hyper_conv_session(kg_embedding, session_hyper_edge_index)
            session_embedding = session_embedding[hypergraph_items]
            _, knowledge_hyper_edge_index = self._get_knowledge_hypergraph(session_related_items)
            raw_knowledge_embedding = self.hyper_conv_knowledge(kg_embedding, knowledge_hyper_edge_index)
            knowledge_embedding = self._get_knowledge_embedding(hypergraph_items, raw_knowledge_embedding)
            if len(context_entities) == 0:
                session_repr = torch.cat((session_embedding, knowledge_embedding), dim=0)
                session_repr_list.append(session_repr)
            else:
                context_embedding = kg_embedding[context_entities]
                session_repr = torch.cat((session_embedding, knowledge_embedding, context_embedding), dim=0)
                session_repr_list.append(session_repr)

        batch_seq_len = max([session_repr.size(0) for session_repr in session_repr_list if session_repr is not None])
        mask_list = []
        for i in range(len(session_repr_list)):
            if session_repr_list[i] is None:
                mask_list.append([False] * batch_seq_len)
                zero_repr = torch.zeros((batch_seq_len, self.kg_emb_dim), device=self.device, dtype=torch.float)
                session_repr_list[i] = zero_repr
            else:
                mask_list.append([False] * (batch_seq_len - session_repr_list[i].size(0)) + [True] * session_repr_list[i].size(0))
                zero_repr = torch.zeros((batch_seq_len - session_repr_list[i].size(0), self.kg_emb_dim),
                                        device=self.device, dtype=torch.float)
                session_repr_list[i] = torch.cat((zero_repr, session_repr_list[i]), dim=0)

        session_repr_embedding = torch.stack(session_repr_list, dim=0)
        session_repr_embedding = self.entity_to_token(session_repr_embedding)
        return session_repr_embedding, torch.tensor(mask_list, device=self.device, dtype=torch.bool)

    def decode_forced(self, related_encoder_state, context_encoder_state, session_state, user_embedding, resp):
        bsz = resp.size(0)
        seqlen = resp.size(1)
        inputs = resp.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, related_encoder_state, context_encoder_state, session_state)
        token_logits = F.linear(latent, self.token_embedding.weight)
        user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)

        user_latent = self.entity_to_token(user_embedding)
        user_latent = user_latent.unsqueeze(1).expand(-1, seqlen, -1)
        copy_latent = torch.cat((user_latent, latent), dim=-1)
        copy_logits = self.copy_proj_2(torch.relu(self.copy_proj_1(copy_latent)))
        if self.dataset == 'HReDial':
            copy_logits = copy_logits * self.copy_mask.unsqueeze(0).unsqueeze(0)  # not for tg-redial
        sum_logits = token_logits + user_logits + copy_logits
        _, preds = sum_logits.max(dim=-1)
        return sum_logits, preds

    def decode_greedy(self, related_encoder_state, context_encoder_state, session_state, user_embedding):
        bsz = context_encoder_state[0].shape[0]
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(self.longest_label):
            scores, incr_state = self.decoder(xs, related_encoder_state, context_encoder_state, session_state, incr_state)  # incr_state is always None
            scores = scores[:, -1:, :]
            token_logits = F.linear(scores, self.token_embedding.weight)
            user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)

            user_latent = self.entity_to_token(user_embedding)
            user_latent = user_latent.unsqueeze(1).expand(-1, 1, -1)
            copy_latent = torch.cat((user_latent, scores), dim=-1)
            copy_logits = self.copy_proj_2(torch.relu(self.copy_proj_1(copy_latent)))
            if self.dataset == 'HReDial':
                copy_logits = copy_logits * self.copy_mask.unsqueeze(0).unsqueeze(0)  # not for tg-redial
            sum_logits = token_logits + user_logits + copy_logits
            probs, preds = sum_logits.max(dim=-1)
            logits.append(scores)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.end_token_idx).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def converse(self, batch, mode):
        related_tokens = batch['related_tokens']
        context_tokens = batch['context_tokens']
        related_items = batch['related_items']
        related_entities = batch['related_entities']
        context_entities = batch['context_entities']
        response = batch['response']
        kg_embedding = self.kg_encoder(self.kg_embedding.weight, self.edge_idx, self.edge_type)  # (n_entity, emb_dim)
        session_state = self.encode_session(
            related_items,
            context_entities,
            kg_embedding
        )  # (batch_size, batch_seq_len, token_emb_dim)
        user_embedding = self.encode_user(
            related_entities,
            related_items,
            context_entities,
            kg_embedding
        )  # (batch_size, emb_dim)
        related_encoder_state = self.related_encoder(related_tokens)
        context_encoder_state = self.context_encoder(context_tokens)
        if mode != 'test':
            self.longest_label = max(self.longest_label, response.shape[1])
            logits, preds = self.decode_forced(related_encoder_state, context_encoder_state, session_state, user_embedding, response)
            logits = logits.view(-1, logits.shape[-1])
            labels = response.view(-1)
            return self.conv_loss(logits, labels), preds
        else:
            _, preds = self.decode_greedy(related_encoder_state, context_encoder_state, session_state, user_embedding)
            return preds

    def forward(self, batch, mode, stage):
        if len(self.gpu) >= 2:
            self.edge_idx = self.edge_idx.cuda(torch.cuda.current_device())
            self.edge_type = self.edge_type.cuda(torch.cuda.current_device())
        if stage == "conv":
            return self.converse(batch, mode)
        if stage == "rec":
            return self.recommend(batch, mode)