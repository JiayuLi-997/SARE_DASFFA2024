# -*- coding: UTF-8 -*-

""" AFM
Reference:
	'Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks', Xiao et al, 2017. Arxiv.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseModel import ImpressionContextModel
from utils.layers import AttLayer

class AFM(ImpressionContextModel):
	reader = 'ImpressionContextReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'attention_size', 'loss_n', 'reg_weight']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--attention_size', type=int, default=64,
							help='Size of attention embedding vectors.')
		parser.add_argument('--dropout_prob',type=float, default=0.3)
		parser.add_argument('--reg_weight',type=float, default=2.0)
		return ImpressionContextModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.context_feature_dim = sum(corpus.feature_max.values()) 
		self.context_feature_num = len(corpus.feature_max)

		self.vec_size = args.emb_size
		self.attention_size = args.attention_size
		self.dropout_prob = args.dropout_prob
		self.reg_weight = args.reg_weight

		self._define_params()
		self.apply(self.init_weights)
	
	@staticmethod
	def init_weights(m):
		if 'Linear' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)
			if m.bias is not None:
				nn.init.normal_(m.bias, mean=0.0, std=0.01)
		elif 'Embedding' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)

	def _define_params(self):
		self.context_embedding = nn.Embedding(self.context_feature_dim, self.vec_size)
		self.linear_embedding = nn.Embedding(self.context_feature_dim, 1)
		self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)
		self.dropout_layer = nn.Dropout(p=self.dropout_prob)
		self.attlayer = AttLayer(self.vec_size, self.attention_size)
		self.p = torch.nn.Parameter(torch.randn(self.vec_size),requires_grad=True)

	def build_cross(self, feat_emb):
		row = []
		col = []
		for i in range(self.context_feature_num-1):
			for j in range(i+1, self.context_feature_num):
				row.append(i)
				col.append(j)
		p = feat_emb[:,:,row]
		q = feat_emb[:,:,col]
		return p, q

	def afm_layer(self, infeature):
		"""Reference:
			RecBole: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/afm.py
		"""
		p, q = self.build_cross(infeature)
		pair_wise_inter = torch.mul(p,q) # batch_size * num_items * num_pairs * emb_dim

		att_signal = self.attlayer(pair_wise_inter).unsqueeze(dim=-1) # attention weights for each pair
		att_inter = torch.mul(
			att_signal, pair_wise_inter
		)  # [batch_size, num_items, num_pairs, emb_dim]
		att_pooling = torch.sum(att_inter, dim=-2)  # [batch_size, num_items, emb_dim]
		att_pooling = self.dropout_layer(att_pooling)  # [batch_size, num_items, emb_dim]

		att_pooling = torch.mul(att_pooling, self.p)  # [batch_size, num_items, emb_dim]
		att_pooling = torch.sum(att_pooling, dim=-1, keepdim=True)  # [batch_size, num_items, 1]

		return att_pooling
	
	def forward(self, feed_dict):
		context_features = feed_dict['context_mh']

		linear_value = self.overall_bias + self.linear_embedding(context_features).squeeze(dim=-1).sum(dim=-1)
		fm_vectors = self.context_embedding(context_features) # batch_size * num_items * num_feature * emb_dim
		afm_vectors = self.afm_layer(fm_vectors)
		predictions = linear_value + afm_vectors.squeeze(dim=-1)

		return {'prediction':predictions,'feed_dict':feed_dict}
	
	def loss(self, out_dict: dict, target=None):
		l2_loss = self.reg_weight * torch.norm(self.attlayer.w.weight, p=2)
		loss = super().loss(out_dict,target)
		return loss + l2_loss