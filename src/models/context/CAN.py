'''
Reference:
 	CAN: feature co-action network for click-through rate prediction.
	Bian, Weijie, et al. 
  	Proceedings of the fifteenth ACM international conference on web search and data mining. 2022.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as fn
import pandas as pd
from models.context.DIN import Dice

from models.BaseModel import ImpressionContextSeqModel

class CAN(ImpressionContextSeqModel):
	reader = 'ImpressionContextSeqReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['induce_vec_size', 'feed_vec_size', 'orders']
 
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--feed_vec_size',type=int,default=8)
		parser.add_argument('--induce_vec_size',type=int,default=256)
		parser.add_argument('--orders',type=int,default=3)
		parser.add_argument('--co_action_layers',type=str,default='[8,4]')
		parser.add_argument('--dnn_layers',type=str,default='[200,80]')
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--att_layers', type=str, default='[64]',
							help="Size of each layer.")
		parser.add_argument('--din_dnn_layers', type=str, default='[64]',
							help="Size of each layer.")
		return ImpressionContextSeqModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.include_id = 1
		self.user_feature_dim = sum([corpus.feature_max[f] for f in corpus.user_feature_names+['user_id']])
		self.situ_feature_dim = sum([corpus.feature_max[f] for f in corpus.context_feature_names])
		self.item_feature_dim = sum([corpus.feature_max[f] for f in corpus.item_feature_names+['item_id']])
		self.item_feature_num = len(corpus.item_feature_names) + 1
		self.user_feature_num = len(corpus.user_feature_names) + 1
		self.situ_feature_num = len(corpus.context_feature_names)

		self.feed_vec_size = args.feed_vec_size		
		self.induce_vec_size = args.induce_vec_size

		self.orders = args.orders
		self.co_action_layers = eval(args.co_action_layers)
		pre_size = self.feed_vec_size*self.orders
		co_action_nums = 0
		for layer_size in self.co_action_layers:
			co_action_nums += pre_size*layer_size + layer_size
			pre_size = layer_size
		assert self.induce_vec_size>=co_action_nums

		self.dnn_layers = eval(args.dnn_layers)
		self.use_softmax = False
		
		self.softmax_stag = 0 #args.softmax_stag
		self.vec_size = args.emb_size
		self.att_layers = eval(args.att_layers)
		self.din_dnn_layers = eval(args.din_dnn_layers)

		self._define_params()
		self.apply(self.init_weights)


	def _define_params(self):
		self.user_embedding_feed = nn.Embedding(self.user_num, self.feed_vec_size)
		self.situ_embedding_feed = nn.Embedding(self.situ_feature_dim, self.feed_vec_size)
		self.item_embedding_feed = nn.Embedding(self.item_feature_dim, self.feed_vec_size)
		self.item_embedding_induce = nn.Embedding(self.item_num, self.induce_vec_size)

		self.activation = nn.Tanh()
		DIN_OUTPUT_DIM = self.din_dnn_layers[-1]
		inp_shape = sum(self.co_action_layers) * ((self.situ_feature_num+1)+ 1)
		self.bn1 = torch.nn.BatchNorm1d(num_features=inp_shape+DIN_OUTPUT_DIM)#+self.item_feature_num*self.vec_size)
		self.dnn_mlp_layers = torch.nn.ModuleList()
		pre_size = inp_shape+DIN_OUTPUT_DIM
		for size in self.dnn_layers:
			self.dnn_mlp_layers.append(torch.nn.Linear(pre_size, size))	
			self.dnn_mlp_layers.append(torch.nn.PReLU())
			self.dnn_mlp_layers.append(torch.nn.Dropout(self.dropout))
			pre_size = size
		self.dnn_mlp_layers.append(torch.nn.Linear(pre_size, 2 if self.use_softmax else 1))
		
		self.user_embedding = nn.Embedding(self.user_feature_dim, self.vec_size)
		self.situ_embedding = nn.Embedding(self.situ_feature_dim, self.vec_size)
		self.item_embedding = nn.Embedding(self.item_feature_dim, self.vec_size)

		self.att_mlp_layers = torch.nn.ModuleList()
		pre_size = 4 * self.item_feature_num * self.vec_size 
		for size in self.att_layers:
			self.att_mlp_layers.append(torch.nn.Linear(pre_size, size))
			self.att_mlp_layers.append(nn.Sigmoid())
			self.att_mlp_layers.append(torch.nn.Dropout(self.dropout))
			pre_size = size
		self.dense = nn.Linear(pre_size, 1)

		self.din_dnn_mlp_layers = torch.nn.ModuleList()
		pre_size = 3 * self.item_feature_num * self.vec_size + self.user_feature_num * self.vec_size + self.situ_feature_num * self.vec_size
		for size in self.din_dnn_layers:
			self.din_dnn_mlp_layers.append(torch.nn.Linear(pre_size, size))
			self.din_dnn_mlp_layers.append(Dice(size))
			self.din_dnn_mlp_layers.append(torch.nn.Dropout(self.dropout))
			pre_size = size
	
	def forward(self, feed_dict):
		user_feats = feed_dict['user_features'] # B * user features(at least user_id)
		situ_feats = feed_dict['context_features']
		item_feats = feed_dict['item_features'] # B * item num * item features(at least item_id)
		item_ids = feed_dict['item_id'] # B * item num
		user_ids = feed_dict['user_id'] # B * item num
		history_item_feats = feed_dict['history_item_features'] # B * hislens * item features
		history_item_ids = feed_dict['history_items'] # B * hislen

		hislens = feed_dict['lengths'] # B
		mask = 	torch.arange(history_item_ids.shape[1])[None,:].to(self.device) < hislens[:,None]

		# embedding
		user_ids_emb = self.user_embedding_feed(user_ids).unsqueeze(1) #.flatten(start_dim=-2) # B * 1 * vecsize
		situ_feats_emb = self.situ_embedding_feed(situ_feats) #.flatten(start_dim=-2) # B * 1 * sf*vecsize
		item_ids_emb = self.item_embedding_induce(item_ids) #.flatten(start_dim=-2) # B * item num * (if*vecsize)
		item_his_eb = self.item_embedding_feed(history_item_ids) #.flatten(start_dim=-2) # B * hislens * (if*vecsize)

		ui_coaction = self.gen_coaction(item_ids_emb, user_ids_emb,)

		si_coaction = []
		for s_feature in range(self.situ_feature_num):
			si_coaction.append(self.gen_coaction(item_ids_emb, situ_feats_emb[:,:,s_feature,:]))
		si_coaction = torch.cat(si_coaction,dim=-1)

		his_coaction = self.gen_his_coation(item_ids_emb, item_his_eb, mask)
 	
		# embedding2
		user_feats_emb_din = self.user_embedding(user_feats).flatten(start_dim=-2) # B * 1 * (uf*vecsize)
		situ_feats_emb_din = self.situ_embedding(situ_feats).flatten(start_dim=-2) # B * 1 * (sf*vecsize)
		history_feats_emb_din = self.item_embedding(history_item_feats).flatten(start_dim=-2) # B * hislens * (if*vecsize)
		item_feats_emb_din = self.item_embedding(item_feats).flatten(start_dim=-2) # B * item num * (if*vecsize)

		# since storage is not supported for all neg items to predict at once, we need to predict one by one
		self.mask_mat = (torch.arange(history_item_feats.shape[1]).view(1,-1)).to(self.device)
		din_output = self.attention_and_dnn(item_feats_emb_din, history_feats_emb_din, hislens, 
                                       user_feats_emb_din, situ_feats_emb_din)
		
		all_coaction = torch.cat([ui_coaction,si_coaction,his_coaction,din_output,],dim=-1)
		logit = self.fcn_net(all_coaction)
		return {'prediction':logit,'feed_dict':feed_dict}

	def gen_coaction(self, induction, feed):
		# induction: B * item num * induce vec size; feed: B * 1 * feed vec size
		B, item_num, _ = induction.shape

		feed_orders = []
		for i in range(self.orders):
			feed_orders.append(feed**(i+1))
		feed_orders = torch.cat(feed_orders,dim=-1) # B * 1 * (feed vec size * order)

		weight, bias = [], []
		pre_size = feed_orders.shape[-1]
		start_dim = 0
		for layer in self.co_action_layers:
			weight.append(induction[:,:,start_dim:start_dim+pre_size*layer].view(B,item_num,pre_size,layer))
			start_dim += pre_size*layer
			bias.append(induction[:,:,start_dim:start_dim+layer]) # B * item num * layer
			start_dim += layer
			pre_size = layer

		outputs = []
		hidden_state = feed_orders.repeat(1,item_num,1).unsqueeze(2)
		for layer_idx in range(len(self.co_action_layers)):
			hidden_state = self.activation(torch.matmul(hidden_state, weight[layer_idx]) + bias[layer_idx].unsqueeze(2))
			outputs.append(hidden_state.squeeze(2))
		outputs = torch.cat(outputs,dim=-1)
		return outputs
			
	def gen_his_coation(self, induction, feed, mask):
		# induction: B * item num * induce vec size; feed_his: B * his * feed vec size
		B, item_num, _ = induction.shape
		max_len = feed.shape[1]
		
		feed_orders = []
		for i in range(self.orders):
			feed_orders.append(feed**(i+1))
		feed_orders = torch.cat(feed_orders,dim=-1) # B * his * (feed vec size * order)

		weight, bias = [], []
		pre_size = feed_orders.shape[-1]
		start_dim = 0
		for layer in self.co_action_layers:
			weight.append(induction[:,:,start_dim:start_dim+pre_size*layer].view(B,item_num,pre_size,layer))
			start_dim += pre_size*layer
			bias.append(induction[:,:,start_dim:start_dim+layer]) # B * item num * layer
			start_dim += layer
			pre_size = layer
	
		outputs = []
		hidden_state = feed_orders.unsqueeze(2).repeat(1,1,item_num,1).unsqueeze(3)
		for layer_idx in range(len(self.co_action_layers)):
			# weight: B * item num * pre size * size, hidden: B * his len * item num * 1 * pre size
			hidden_state = self.activation(torch.matmul(hidden_state, 
                                weight[layer_idx].unsqueeze(1)) + 
                   				bias[layer_idx].unsqueeze(1).unsqueeze(3)) # B * his len * item num * 1 * size
			outputs.append((hidden_state.squeeze(3)*mask[:,:,None,None]).sum(dim=1)/mask.sum(dim=-1)[:,None,None]) # B * item num * size
		outputs = torch.cat(outputs,dim=-1)
		return outputs
 	
	def fcn_net(self, inp, ):
		bn1 = self.bn1(inp.transpose(-1,-2)).transpose(-1,-2)
		output = bn1
		for layer in self.dnn_mlp_layers:
			output = layer(output)
		output = output.squeeze(-1)
		return output
	
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
		for layer in self.din_dnn_mlp_layers:
			din = layer(din)
		predictions = din
		return predictions.view(batch_size, item_num, self.din_dnn_layers[-1])

	
	class Dataset(ImpressionContextSeqModel.Dataset):
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			return feed_dict
