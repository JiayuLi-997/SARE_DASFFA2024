# -*- coding: UTF-8 -*-

""" DIN
Reference:
 	Deep interest network for click-through rate prediction. 
	Zhou, Guorui, et al. 
  	Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseModel import ImpressionContextSeqModel

class DIN(ImpressionContextSeqModel):
	reader = 'ImpressionContextSeqReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--att_layers', type=str, default='[64]',
							help="Size of each layer.")
		parser.add_argument('--dnn_layers', type=str, default='[64]',
							help="Size of each layer.")
		parser.add_argument('--softmax_stag',type=int,default=0)
		return ImpressionContextSeqModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)

		self.softmax_stag = args.softmax_stag
		self.include_id = 1
		self.user_feature_dim = sum([corpus.feature_max[f] for f in corpus.user_feature_names+['user_id']])
		self.situ_feature_dim = sum([corpus.feature_max[f] for f in corpus.context_feature_names])
		self.item_feature_dim = sum([corpus.feature_max[f] for f in corpus.item_feature_names+['item_id']])
		self.item_feature_num = len(corpus.item_feature_names) + 1
		self.user_feature_num = len(corpus.user_feature_names) + 1
		self.situ_feature_num = len(corpus.context_feature_names)

		self.vec_size = args.emb_size
		self.att_layers = eval(args.att_layers)
		self.dnn_layers = eval(args.dnn_layers)

		self._define_params()
		self.apply(self.init_weights)
	
	def _define_params(self):
		self.user_embedding = nn.Embedding(self.user_feature_dim, self.vec_size)
		self.situ_embedding = nn.Embedding(self.situ_feature_dim, self.vec_size)
		self.item_embedding = nn.Embedding(self.item_feature_dim, self.vec_size)

		self.user_transfer = nn.Linear(self.user_feature_num*self.vec_size, self.item_feature_num * self.vec_size)

		self.att_mlp_layers = torch.nn.ModuleList()
		pre_size = 4 * self.item_feature_num * self.vec_size 
		for size in self.att_layers:
			self.att_mlp_layers.append(torch.nn.Linear(pre_size, size))
			# self.att_mlp_layers.append(nn.ReLU())
			self.att_mlp_layers.append(nn.Sigmoid())
			# self.att_mlp_layers.append(Dice(size,dim=3))
			self.att_mlp_layers.append(torch.nn.Dropout(self.dropout))
			pre_size = size
		self.dense = nn.Linear(pre_size, 1)

		self.dnn_mlp_layers = torch.nn.ModuleList()
		pre_size = 3 * self.item_feature_num * self.vec_size + self.user_feature_num * self.vec_size + self.situ_feature_num * self.vec_size
		# pre_size = 2 * self.item_feature_num * self.vec_size + self.item_feature_num * self.vec_size + self.situ_feature_num * self.vec_size
		# pre_size = 3 * self.item_feature_num * self.vec_size + self.item_feature_num * self.vec_size + self.situ_feature_num * self.vec_size
		for size in self.dnn_layers:
			self.dnn_mlp_layers.append(torch.nn.Linear(pre_size, size))
			# self.dnn_mlp_layers.append(torch.nn.BatchNorm1d(num_features=size))
			self.dnn_mlp_layers.append(Dice(size))
			self.dnn_mlp_layers.append(torch.nn.Dropout(self.dropout))
			pre_size = size
		self.dnn_mlp_layers.append(torch.nn.Linear(pre_size, 1))

	def attention(self, queries, keys, keys_length, softmax_stag=False, return_seq_weight=False):
		'''Reference:
			RecBole layers: SequenceAttLayer
			queries: batch * (if*vecsize)
		'''
		embedding_size = queries.shape[-1]  # H
		hist_len = keys.shape[1]  # T
		queries = queries.repeat(1, hist_len)
		queries = queries.view(-1, hist_len, embedding_size)
		# MLP Layer
		input_tensor = torch.cat(
			[queries, keys, queries - keys, queries * keys], dim=-1
		)
		output = input_tensor
		for layer in self.att_mlp_layers:
			output = layer(output)
		output = torch.transpose(self.dense(output), -1, -2)
		# get mask
		output = output.squeeze(1)
		mask = self.mask_mat.repeat(output.size(0), 1)
		mask = mask >= keys_length.unsqueeze(1)
		# mask
		if softmax_stag:
			mask_value = -np.inf
		else:
			mask_value = 0.0
		output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))
		output = output.unsqueeze(1)
		output = output / (embedding_size**0.5)
		# get the weight of each user's history list about the target item
		if softmax_stag:
			output = fn.softmax(output, dim=2)  # [B, 1, T]
		if not return_seq_weight:
			output = torch.matmul(output, keys)  # [B, 1, H]
		torch.cuda.empty_cache()
		return output.squeeze()

	def attention_and_dnn(self, item_feats_emb, history_feats_emb, hislens, user_feats_emb, situ_feats_emb):
		batch_size, item_num, feats_emb = item_feats_emb.shape
		_, max_len, his_emb = history_feats_emb.shape

		item_feats_emb2d = item_feats_emb.view(-1, feats_emb) # 每个sample的item在一块
		history_feats_emb2d = history_feats_emb.unsqueeze(1).repeat(1,item_num,1,1).view(-1,max_len,his_emb)
		hislens2d = hislens.unsqueeze(1).repeat(1,item_num).view(-1)
		user_feats_emb2d = user_feats_emb.repeat(1,item_num,1).view(-1, user_feats_emb.shape[-1])
		situ_feats_emb2d = situ_feats_emb.repeat(1,item_num,1).view(-1, situ_feats_emb.shape[-1])
		user_his_emb = self.attention(item_feats_emb2d, history_feats_emb2d, hislens2d,softmax_stag=self.softmax_stag)
		din = torch.cat([user_his_emb, item_feats_emb2d, user_his_emb*item_feats_emb2d, user_feats_emb2d,situ_feats_emb2d], dim=-1)
		# din = torch.cat([user_his_emb, item_feats_emb2d, user_feats_emb2d,situ_feats_emb2d], dim=-1)
		for layer in self.dnn_mlp_layers:
			din = layer(din)
		predictions = din
		return predictions.view(batch_size, item_num)

	def forward(self, feed_dict):
		u_ids = feed_dict['user_id']  # B
		i_ids = feed_dict['item_id']  # B * -1
		hislens = feed_dict['lengths'] # B
		user_feats = feed_dict['user_features'] # B * 1 * user features(at least user_id)
		item_feats = feed_dict['item_features'] # B * item num * item features(at least item_id)
		# reverse=False
		# if np.random.rand()>0.5:
		# 	item_feats = torch.cat([item_feats[:,50:,:],item_feats[:,:50,:]],dim=1)
		# 	reverse=True
		situ_feats = feed_dict['context_features'] # B * 1 * situ features
		history_item_feats = feed_dict['history_item_features'] # B * hislens * item features

		# embedding
		user_feats_emb = self.user_embedding(user_feats).flatten(start_dim=-2) # B * 1 * (uf*vecsize)
		# user_feats_squeeze = self.user_transfer(user_feats_emb)
		situ_feats_emb = self.situ_embedding(situ_feats).flatten(start_dim=-2) # B * 1 * (sf*vecsize)
		history_feats_emb = self.item_embedding(history_item_feats).flatten(start_dim=-2) # B * hislens * (if*vecsize)
		item_feats_emb = self.item_embedding(item_feats).flatten(start_dim=-2) # B * item num * (if*vecsize)

		# since storage is not supported for all neg items to predict at once, we need to predict one by one
		self.mask_mat = (torch.arange(history_item_feats.shape[1]).view(1,-1)).to(self.device)
		# predictions = self.attention_and_dnn(item_feats_emb, history_feats_emb, hislens, user_feats_emb, situ_feats_emb)
		predictions = self.attention_and_dnn(item_feats_emb, history_feats_emb, hislens, user_feats_emb, situ_feats_emb)

		# if reverse:
		# predictions = torch.cat([predictions[:,-50:],predictions[:,:-50]],dim=1)
      
		return {'prediction':predictions}	
	
	# class Dataset(ImpressionContextSeqModel.Dataset):
	# 	def _get_feed_dict(self, index):
	# 		feed_dict = super()._get_feed_dict(index)
	# 		for key in feed_dict:
	# 			if 'history' in key:
	# 				feed_dict[key] = feed_dict[key][::-1].copy()
	# 		return feed_dict

class Dice(nn.Module):
	"""The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

	Input shape:
		- 2 dims: [batch_size, embedding_size(features)]
		- 3 dims: [batch_size, num_features, embedding_size(features)]

	Output shape:
		- Same shape as input.

	References
		- [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
		- https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
	"""

	def __init__(self, emb_size, dim=2, epsilon=1e-8, device='cpu'):
		super(Dice, self).__init__()
		assert dim == 2 or dim == 3

		self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
		self.sigmoid = nn.Sigmoid()
		self.dim = dim

		# wrap alpha in nn.Parameter to make it trainable
		if self.dim == 2:
			self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
		else:
			self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

	def forward(self, x):
		assert x.dim() == self.dim
		if self.dim == 2:
			x_p = self.sigmoid(self.bn(x))
			out = self.alpha * (1 - x_p) * x + x_p * x
		else:
			x = torch.transpose(x, 1, 2)
			x_p = self.sigmoid(self.bn(x))
			out = self.alpha * (1 - x_p) * x + x_p * x
			out = torch.transpose(out, 1, 2)
		return out
