# -*- coding: UTF-8 -*-

""" FM
Reference:
	Factorization Machines. Steffen Rendle.
 	2010 IEEE International conference on data mining. IEEE, 2010.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseModel import ImpressionContextModel

class FM(ImpressionContextModel):
	reader = 'ImpressionContextReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size','loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		return ImpressionContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.context_feature_dim = sum(corpus.feature_max.values()) 
		self.context_feature_num = len(corpus.feature_max)
		self.vec_size = args.emb_size
		self._define_params()
		self.apply(self.init_weights)

	def _define_params(self):
		self.context_embedding = nn.Embedding(self.context_feature_dim, self.vec_size)
		self.linear_embedding = nn.Embedding(self.context_feature_dim, 1)
		self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)
	
	def forward(self, feed_dict):
		context_features = feed_dict['context_mh']

		linear_value = self.overall_bias + self.linear_embedding(context_features).squeeze(dim=-1)
		linear_value = linear_value.sum(dim=-1)
		fm_vectors = self.context_embedding(context_features)
		fm_vectors = 0.5 * (fm_vectors.sum(dim=-2).pow(2) - fm_vectors.pow(2).sum(dim=-2))
		predictions = linear_value + fm_vectors.sum(dim=-1)

		return {'prediction':predictions}
