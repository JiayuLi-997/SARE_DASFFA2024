import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as fn

from models.BaseModel import ImpressionContextSeqModel
from models.SARE.BaseSARE import *
from models.context.DIN import Dice

class DIN_SARE(ImpressionContextSeqModel,BaseSARE):
	reader = 'ImpressionContextSeqReader'
	runner = 'ImpressionSituRunner'
	extra_log_args = ['prob_weights','prob_loss_n','situ_weights','situ_lr','situ_l2','topk']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--att_layers', type=str, default='[64]',
							help="Size of each layer.")
		parser.add_argument('--dnn_layers', type=str, default='[64]',
							help="Size of each layer.")
		parser = BaseSARE.parse_model_args(parser)
		return ImpressionContextSeqModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionContextSeqModel.__init__(self,args, corpus)
		self.vec_size = args.emb_size
		self.att_layers = eval(args.att_layers)
		self.dnn_layers = eval(args.dnn_layers)
		self.get_generalSARE_init(args, corpus)

		self._define_params()
		self.apply(self.init_weights)
		nn.init.eye_(self.situ_i_transform.weight)
		nn.init.eye_(self.situ_his_transform.weight)
		nn.init.eye_(self.situ_u_transform.weight)
	
	def _define_params(self):
		self.user_embedding = nn.Embedding(self.user_feature_dim, self.vec_size)
		self.item_embedding = nn.Embedding(self.item_feature_dim, self.vec_size)

		self.att_mlp_layers = nn.ModuleList()
		pre_size = 4 * self.item_feature_num * self.vec_size 
		for size in self.att_layers:
			self.att_mlp_layers.append(nn.Linear(pre_size, size))
			self.att_mlp_layers.append(nn.Sigmoid())
			self.att_mlp_layers.append(nn.Dropout(self.dropout))
			pre_size = size
		self.dense = nn.Linear(pre_size, 1)

		self.dnn_mlp_layers = nn.ModuleList()
		pre_size = 3 * self.item_feature_num * self.vec_size + self.user_feature_num * self.vec_size 
		for size in self.dnn_layers:
			self.dnn_mlp_layers.append(nn.Linear(pre_size, size))
			self.dnn_mlp_layers.append(nn.BatchNorm1d(num_features=size))
			self.dnn_mlp_layers.append(Dice(size))
			self.dnn_mlp_layers.append(nn.Dropout(self.dropout))
			pre_size = size
		self.dnn_mlp_layers.append(torch.nn.Linear(pre_size, 1))

		self.get_generalSARE_params()
		self.situ_i_transform = nn.Linear(self.vec_size*self.item_feature_num,int(self.situ_embedding_size/2))
		self.situ_his_transform = nn.Linear(self.att_layers[-1]*self.item_feature_num,int(self.situ_embedding_size/2))
		self.la_weights2 = nn.Linear(self.situ_embedding_size,len(self.activations))
	
	def customize_parameters(self) -> list:
		# customize optimizer settings for different parameters
		weight_p, bias_p = [], []
		situ_p, situ_bias_p = [], []
		for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
			if 'situ' in name or 'la_weights' in name:
				if 'bias' in name:
					situ_bias_p.append(p)
				else:
					situ_p.append(p)
			elif 'bias' in name:
				bias_p.append(p)
			else:
				weight_p.append(p)
		optimize_dict = [{'params':situ_p,'lr':self.situ_lr,'weight_decay':self.situ_l2},
                   {'params':situ_bias_p,'lr':self.situ_lr,'weight_decay':0},
                   {'params': weight_p,}, {'params': bias_p, 'weight_decay': 0}]
		return optimize_dict
	
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

	def attention_and_dnn(self, item_feats_emb, history_feats_emb, hislens, user_feats_emb,):
                    #    situ_feats_emb):
		batch_size, item_num, feats_emb = item_feats_emb.shape
		_, max_len, his_emb = history_feats_emb.shape

		item_feats_emb2d = item_feats_emb.view(-1, feats_emb) # 每个sample的item在一块
		history_feats_emb2d = history_feats_emb.unsqueeze(1).repeat(1,item_num,1,1).view(-1,max_len,his_emb)
		hislens2d = hislens.unsqueeze(1).repeat(1,item_num).view(-1)
		user_feats_emb2d = user_feats_emb.repeat(1,item_num,1).view(-1, user_feats_emb.shape[-1])
		
		user_his_emb = self.attention(item_feats_emb2d, history_feats_emb2d, hislens2d)
		din = torch.cat([user_his_emb, item_feats_emb2d, user_his_emb*item_feats_emb2d, user_feats_emb2d,], dim=-1)
		for layer in self.dnn_mlp_layers:
			din = layer(din)
		predictions = din
		return predictions.view(batch_size, item_num), (user_his_emb*item_feats_emb2d).view(batch_size, item_num, -1)

	def forward(self, feed_dict):
		hislens = feed_dict['lengths'] # B
		user_feats = feed_dict['user_features'] # B * 1 * user features(at least user_id)
		item_feats = feed_dict['item_features'] # B * item num * item features(at least item_id)
		history_item_feats = feed_dict['history_item_features'] # B * hislens * item features

		# embedding
		user_feats_emb = self.user_embedding(user_feats).flatten(start_dim=-2) # B * 1 * (uf*vecsize)
		history_feats_emb = self.item_embedding(history_item_feats).flatten(start_dim=-2) # B * hislens * (if*vecsize)
		item_feats_emb = self.item_embedding(item_feats).flatten(start_dim=-2) # B * item num * (if*vecsize)

		self.mask_mat = (torch.arange(history_item_feats.shape[1]).view(1,-1)).to(self.device)
		predictions, user_his_emb = self.attention_and_dnn(item_feats_emb, history_feats_emb, hislens, user_feats_emb,)# situ_feats_emb)

		situ_u_vectors = self.situ_u_transform(user_feats_emb.squeeze(1))
		situ_i_vectors = self.situ_i_transform(item_feats_emb)
		situ_his_vectors = self.situ_his_transform(user_his_emb)
		situ_predictions, pred_situ, situ_embed = self.situation_predict(situ_u_vectors, situ_i_vectors,situ_his_vectors, [feed_dict[situ] for situ in self.situ_feature_cnts])

		out_dict = {'prediction': predictions, 'situ_prediction':situ_predictions,
		}
		torch.cuda.empty_cache()
		return out_dict
	
	def situation_predict(self, u_embeddings, i_embeddings, his_embeddings, situ_target):
		s = self.la_weights(u_embeddings) # batch * activation layers
		i_activated = []
		for i,f in enumerate(self.activations):
			i_activated.append(s[:,i][:,None,None]*f(i_embeddings))
		pred_situ = torch.stack(i_activated,dim=-1).sum(dim=-1)

		s_his = self.la_weights2(u_embeddings)
		h_activated = []
		for i,f in enumerate(self.activations):
			h_activated.append(s_his[:,i][:,None,None]*f(his_embeddings))
		pred_situ_his = torch.stack(h_activated,dim=-1).sum(dim=-1)

		pred_situ_sum = torch.cat([pred_situ, pred_situ_his],dim=-1)

		situ_embeds = [situ_embedding(situ_id) for situ_id,situ_embedding in zip(situ_target,self.situ_embeddings)]
		situ_embed = torch.stack(situ_embeds,dim=-1) # batch * context embed * context num
		situ_embed_weights = self.get_situ_fusion_weights(u_embeddings, situ_embed) # batch * context num
		situ_embed = (situ_embed_weights[:,None,:] * situ_embed).sum(dim=-1) # / torch.norm(situ_embed,p=2,dim=1)[:,None,:] ).sum(dim=-1)

		pred_situ_prob = (pred_situ_sum*situ_embed[:,None,:]).sum(dim=-1) / torch.norm(pred_situ_sum,p=2,dim=2
						) / torch.norm(situ_embed,p=2,dim=1)[:,None]
		if pred_situ_prob.isnan().max() or pred_situ_prob.isinf().max():
			print('Situation probability error!')
		return pred_situ_prob, pred_situ, situ_embed

	
	def loss(self, out_dict, target):
		return self.SARE_loss(out_dict,target)

	class Dataset(ImpressionContextSeqModel.Dataset):
		def __init__(self, model, corpus, phase: str):
			super().__init__(model,corpus,phase)
			self.include_id = model.include_id
		
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			torch.cuda.empty_cache()
			return feed_dict


	