'''
Reference:
Rashed A, Elsayed S, Schmidt-Thieme L. 
Context and attribute-aware sequential recommendation via cross-attention.
Proceedings of the 16th ACM Conference on Recommender Systems. 2022.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from models.BaseModel import ImpressionContextSeqModel
# from utils.layers import MultiHeadAttention


def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
	'''Applies layer normalization.

	Args:
		inputs: A tensor with 2 or more dimensions, where the first dimension has
		`batch_size`.
		epsilon: A floating number. A very small number for preventing ZeroDivision Error.
		scope: Optional scope for `variable_scope`.
		reuse: Boolean, whether to reuse the weights of a previous layer
		by the same name.

	Returns:
		A tensor with the same shape and data dtype as `inputs`.
	'''
	# with torch.no_grad():
	inputs_shape = inputs.size()
	params_shape = inputs_shape[-1:]

	mean = torch.mean(inputs, dim=-1, keepdim=True)
	variance = torch.var(inputs, dim=-1, keepdim=True)
	# beta = nn.Parameter(torch.zeros(params_shape))
	# gamma = nn.Parameter(torch.ones(params_shape))
	normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
	# outputs = gamma * normalized + beta

	return normalized

class FeedForward(nn.Module):
	def __init__(self, num_units=[2048, 512], dropout_rate=0.2):
		super(FeedForward, self).__init__()
		# self.conv1 = nn.Conv1d(num_units[0], num_units[0], kernel_size=1)
		# self.conv2 = nn.Conv1d(num_units[0], num_units[1], kernel_size=1)
		self.conv1 = nn.Linear(num_units[0],num_units[0])
		self.conv2 = nn.Linear(num_units[0],num_units[1])
		self.dropout = nn.Dropout(dropout_rate)
		self.leaky_relu = nn.LeakyReLU()

	def forward(self, inputs):
		# Inner layer
		outputs = self.conv1(inputs)
		outputs = self.dropout(outputs)
		outputs = self.leaky_relu(outputs)
		
		# Readout layer
		outputs = self.conv2(outputs)
		outputs = self.dropout(outputs)
		
		# Residual connection
		outputs += inputs
		
		return outputs

class CARCA(ImpressionContextSeqModel,):
	reader = 'ImpressionContextSeqSituReader'
	# runner = 'ImpressionSituRunner'
	runner = 'ImpressionRunner'
	extra_log_args = ['loss_n','dropout','output_sigmoid','num_heads','num_blocks']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--situ_emb_size', type=int, default=64,
							help='Size of situation embedding vectors.')
		parser.add_argument('--g_vec_size', type=int, default=64,
							help='Size of history attr&context vectors.')
		parser.add_argument('--output_sigmoid',type=int,default=0)
		parser.add_argument('--use_res',type=int,default=0)
		parser.add_argument('--num_heads',type=int,default=2)
		parser.add_argument('--num_blocks',type=int,default=3)
		parser.add_argument('--final_layer_size',type=str,default='[64,64]')
		parser.add_argument('--user_linear',type=int,default=1)
		parser.add_argument('--final_use_seq',type=int,default=1)
		return ImpressionContextSeqModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionContextSeqModel.__init__(self, args, corpus)
		self.user_linear = args.user_linear
		self.user_feature_dim = sum([corpus.feature_max[f] for f in corpus.user_feature_names+['user_id']])
		self.user_feature_num = len(corpus.user_feature_names) + 1
		# self.item_feature_dim = sum([corpus.feature_max[f] for f in corpus.item_feature_names+['item_id']])
		self.item_feature_dim = sum([corpus.feature_max[f] for f in corpus.item_feature_names])
		self.situ_feature_dim = sum([corpus.feature_max[f] for f in corpus.context_feature_names])
		self.item_feature_num = len(corpus.item_feature_names) # + 1
		self.situ_feature_num = len(corpus.context_feature_names)
 	
		self.vec_size = args.emb_size # d
		self.meta_vec_size = args.situ_emb_size # g / (item f num + situ f num)
		# self.g_vec_size = args.g_vec_size

		self.num_heads = args.num_heads
		self.num_blocks = args.num_blocks	

		self.sigmoid = args.output_sigmoid
		self.use_res = args.use_res
		self.final_layer_size = eval(args.final_layer_size)
		self.include_id = False
		self.final_use_seq = args.final_use_seq
  
		self._define_params()
		self.apply(self.init_weights)
		for i in range(self.num_blocks):
			nn.init.eye_(self.attention_block[i].q_linear.weight)
			nn.init.eye_(self.attention_block[i].k_linear.weight)
			nn.init.eye_(self.attention_block[i].v_linear.weight)

		nn.init.eye_(self.attention2.q_linear.weight)
		nn.init.eye_(self.attention2.k_linear.weight)
		nn.init.eye_(self.attention2.v_linear.weight)

	def _define_params(self):
		self.item_id_embedding = nn.Embedding(self.item_num, self.vec_size)
		# self.item_id_bias = nn.Parameter(torch.randn(self.vec_size),requires_grad=True)
		self.item_feat_embedding = nn.Embedding(self.item_feature_dim, self.meta_vec_size)
		self.situ_feat_embedding = nn.Embedding(self.situ_feature_dim, self.meta_vec_size)
		# self.seq_feat_linear = nn.Linear(self.meta_vec_size*(self.item_feature_num+self.situ_feature_num), 
		# 						   self.g_vec_size, bias=True)
		self.g_vec_size = self.meta_vec_size*(self.item_feature_num+self.situ_feature_num)
		# self.seq_linear = nn.Linear(self.vec_size+self.g_vec_size, self.vec_size,bias=False)

		self.user_feature_embedding = nn.Embedding(self.user_feature_dim, self.vec_size)
		if self.user_linear:
			self.user_feat_linear = nn.Linear(self.vec_size*self.user_feature_num, self.vec_size)
			self.u_vec_size = self.vec_size
		else:
			self.u_vec_size = self.vec_size*self.user_feature_num
			self.user_feat_linear = nn.Identity()

		self.dropout_layer = nn.Dropout(p=self.dropout)

		self.attention_block = nn.ModuleList()
		for i in range(self.num_blocks):
			# self.attention_block.append(MultiHeadAttention(self.vec_size, n_heads=self.num_heads,kq_same=False,bias=False))
			self.attention_block.append(MultiHeadAttention(self.vec_size+self.g_vec_size, n_heads=self.num_heads,kq_same=False,bias=False))

		self.FFN = nn.ModuleList()
		for i in range(self.num_blocks):
			# self.FFN.append(FeedForward(num_units=[self.vec_size,self.vec_size],dropout_rate=self.dropout))
			self.FFN.append(FeedForward(num_units=[self.vec_size+self.g_vec_size,self.vec_size+self.g_vec_size],dropout_rate=self.dropout))

		# self.attention2 = MultiHeadAttention(self.vec_size, n_heads=self.num_heads, kq_same=False,bias=False)
		self.attention2 = MultiHeadAttention(self.vec_size+self.g_vec_size, n_heads=self.num_heads, kq_same=False,bias=False)
		self.final_layer = nn.ModuleList()# nn.Linear(self.vec_size, 1)
		# pre_size = self.vec_size*4
		if self.final_use_seq:
			pre_size = self.vec_size*3+self.g_vec_size*3+self.u_vec_size
		else:
			pre_size = self.vec_size*2+self.g_vec_size*2+self.u_vec_size
		for size in self.final_layer_size:
			self.final_layer.append(torch.nn.Linear(pre_size, size))	
			self.final_layer.append(torch.nn.LeakyReLU())
			self.final_layer.append(torch.nn.Dropout(self.dropout))
			pre_size = size
		self.final_layer.append(torch.nn.Linear(pre_size, 1))
	
	def customize_parameters(self) -> list:
		# customize optimizer settings for different parameters
		weight_p, bias_p = [], []
		for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
			if 'bias' in name and name not in ['item_id_bias','seq_feat_linear','seq_linear']:
				bias_p.append(p)
			else:
				weight_p.append(p)
		optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
		return optimize_dict

	def forward(self, feed_dict):
		hislens = feed_dict['lengths'] # B
		user_feats = feed_dict['user_features'].squeeze(1) # B * user features(at least user_id)
		item_feats = feed_dict['item_features'] # B * item num * item features(at least item_id)
		situ_feats = feed_dict['context_features'].squeeze(1) # B * 1 * situ features
		history_item_feats = feed_dict['history_item_features'] # B * hislens * item features
		history_situ_feats = feed_dict['history_context_features']
		input_item_ids = feed_dict['item_id'] # B * hislen
		input_seq = feed_dict['history_items'] # B * hislen
		mask = torch.unsqueeze(torch.tensor(input_seq != 0, dtype=torch.float32).to(self.device), -1)
		item_num = item_feats.shape[1]

		seq_in = self.item_id_embedding(input_seq) # +self.item_id_bias[None,None,:] # ignore the scale?
		# seq_in = self.item_id_embedding(input_seq)+self.item_id_bias[None,None,:] # ignore the scale?
		seq_feat = self.item_feat_embedding(history_item_feats).flatten(start_dim=-2)
		seq_cxt = self.situ_feat_embedding(history_situ_feats).flatten(start_dim=-2)
		# seq_feat_in = torch.cat([seq_feat,seq_cxt],dim=-1) # + self.meta_bias[None,None,:] # B * len * [(item f + situ f) * situ_emb_size]
		seq_feat_in = torch.cat([seq_feat,seq_cxt],dim=-1)#+ self.meta_bias[None,None,:] # B * len * [(item f + situ f) * situ_emb_size]
		# seq_feat_emb = self.seq_feat_linear(seq_feat_in)
		seq_feat_emb = seq_feat_in 
		seq_concat = torch.cat([seq_in, seq_feat_emb],dim=-1) # B * hislen * (6*vecsize)
  
		# seq = self.seq_linear(seq_concat)
		seq = seq_concat
		seq = self.dropout_layer(seq)
		seq *= mask # B * hislen * vec size

		for i in range(self.num_blocks):
			seq_norm = normalize(seq)
			att_mask = mask.squeeze(-1).unsqueeze(dim=1).repeat(1,seq.shape[1],1)
			seq = self.attention_block[i](seq_norm,seq_norm,seq_norm,
									   mask=att_mask) # * seq_norm # B * len * emb size
			seq = self.FFN[i](normalize(seq)) * mask# B * emb * len
		seq = normalize(seq)
  
		target_emb_in = self.item_id_embedding(input_item_ids)
		# target_emb_in = self.item_id_embedding(input_item_ids)+self.item_id_bias[None,None,:]
		target_feat_in = self.item_feat_embedding(item_feats).flatten(start_dim=-2) # B * item num * (item f * vec_size)
		target_cxt = self.situ_feat_embedding(situ_feats).flatten(start_dim=-2).unsqueeze(1).repeat(1,item_num,1)
		target_feat = torch.cat([target_feat_in,target_cxt],dim=-1)
		# target_feat_emb = self.seq_feat_linear(target_feat)
		target_feat_emb = target_feat 
		target_emb_con = torch.cat([target_emb_in,target_feat_emb],dim=-1)
		# target_emb = normalize(self.seq_linear(target_emb_con))
		target_emb = normalize((target_emb_con))

		# seq_trans = torch.transpose(seq,dim0=0,dim1=1)
		# target_logits = self.attention2(torch.transpose(target_emb,dim0=0,dim1=1), seq_trans, seq_trans, 
		#						   mask=(1-mask).squeeze().bool())
		# target_logits = torch.transpose(target_logits,dim0=0,dim1=1)
		att_mask = mask.squeeze(-1).unsqueeze(dim=1).repeat(1,target_emb.shape[1],1)
		target_logits = self.attention2(target_emb, seq, seq, 
								  mask=att_mask)
		if self.use_res:
			target_logits += target_emb

		# predictions = self.final_layer(target_logits).squeeze()
		user_emb = self.user_feature_embedding(user_feats).flatten(start_dim=-2).unsqueeze(1)
		user_f = self.user_feat_linear(user_emb).repeat(1,target_logits.shape[1],1)

		seq = (seq*mask).sum(dim=1) / mask.sum(dim=1)

		if self.final_use_seq:
			predictions = torch.cat([target_logits, target_emb,
                           seq.unsqueeze(1).repeat(1,target_logits.shape[1],1), user_f],dim=-1)
		else:
			predictions = torch.cat([target_logits,target_emb,user_f],dim=-1)
		for layer in self.final_layer:
			predictions = layer(predictions)
		predictions = predictions.squeeze(-1)

		if self.sigmoid:
			predictions = predictions.sigmoid()

		if predictions.isnan().max() or predictions.isinf().max():
			print("prediction error!")

		return {'prediction': predictions, 'feed_dict':feed_dict,}
	
	class Dataset(ImpressionContextSeqModel.Dataset):
		def _get_feed_dict(self, index):
			# get item features, user features, and context features separately
			feed_dict = super()._get_feed_dict(index)
			pos = self.data['position'][index]
			user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]

			if len(self.corpus.user_feature_names):
				user_fnames = self.corpus.user_feature_names
				user_features = [self.corpus.user_features[feed_dict['user_id']][c] for c in self.corpus.user_feature_names] 
				user_fnames = ['user_id'] + user_fnames
				user_features = [feed_dict['user_id']] + user_features
				feed_dict['user_features'] = self._convert_multihot(user_fnames, np.array(user_features).reshape(1,-1))
			else:# self.include_id:
				feed_dict['user_features'] = feed_dict['user_id'].reshape(-1,1)
	
			if len(self.corpus.item_feature_names):
				item_fnames = self.corpus.item_feature_names
				items_features = np.array([[self.corpus.item_features[iid][c] if iid in self.corpus.item_features else 0
										for c in self.corpus.item_feature_names] for iid in feed_dict['item_id'] ])
				items_features_history = np.array([[self.corpus.item_features[iid][c] if iid in self.corpus.item_features else 0
										for c in self.corpus.item_feature_names] for iid in feed_dict['history_items'] ])
				if self.include_id:
					item_fnames = ['item_id']+item_fnames
					item_ids = feed_dict['item_id'].reshape(-1,1)
					items_features = np.concatenate([item_ids, items_features],axis=1)
					his_item_ids = feed_dict['history_items'].reshape(-1,1)
					items_features_history = np.concatenate([his_item_ids,items_features_history],axis=1)
				feed_dict['item_features'] = self._convert_multihot(item_fnames, np.array(items_features))
				feed_dict['history_item_features'] = self._convert_multihot(item_fnames,np.array(items_features_history))

			if len(self.corpus.context_feature_names):
				his_context_features = np.array([ x[-1] for x in user_seq ])
				feed_dict['history_context_features'] = self._convert_multihot(self.corpus.context_feature_names, his_context_features)
			
			for key in feed_dict:
				if 'history' in key:
					feed_dict[key] = feed_dict[key][::-1].copy()
			return feed_dict
	
class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, n_heads, kq_same=False, bias=True, attention_d=-1):
		super().__init__()
		"""
		It has projection layer for getting keys, queries and values. Followed by attention.
		"""
		self.d_model = d_model
		self.h = n_heads
		if attention_d<0:
			self.attention_d = self.d_model
		else:
			self.attention_d = attention_d

		self.d_k = self.attention_d // self.h
		self.kq_same = kq_same

		if not kq_same:
			self.q_linear = nn.Linear(d_model, self.attention_d, bias=bias)
		self.k_linear = nn.Linear(d_model, self.attention_d, bias=bias)
		self.v_linear = nn.Linear(d_model, self.attention_d, bias=bias)

	def head_split(self, x):  # get dimensions bs * h * seq_len * d_k
		new_x_shape = x.size()[:-1] + (self.h, self.d_k)
		return x.view(*new_x_shape).transpose(-2, -3)

	def forward(self, q, k, v, mask=None):
		origin_shape = q.size()

		# perform linear operation and split into h heads
		if not self.kq_same:
			q = self.head_split(self.q_linear(q))
		else:
			q = self.head_split(self.k_linear(q))
		k = self.head_split(self.k_linear(k))
		v = self.head_split(self.v_linear(v))
		if len(mask.shape) < len(v.shape):
			mask = mask.unsqueeze(dim=1).repeat(1,self.h,1,1)

		# calculate attention using function we will define next
		output = self.scaled_dot_product_attention(q, k, v, self.d_k, mask)

		# concatenate heads and put through final linear layer
		output = output.transpose(-2, -3).reshape(origin_shape)
		return output

	@staticmethod
	def scaled_dot_product_attention(q, k, v, d_k, mask=None):
		"""
		This is called by Multi-head attention object to find the values.
		"""
		scores = torch.matmul(q, k.transpose(-2, -1)) / d_k ** 0.5  # bs, head, q_len, k_len
		if mask is not None:
			scores = scores.masked_fill(mask == 0, -np.inf)
		scores = ((scores - scores.max(dim=-1,keepdim=True)[0])).softmax(dim=-1)
		output = torch.matmul(scores, v)  # bs, head, q_len, d_k
		return output

		scores - scores.max(dim=-1,keepdim=True)[0] - ((scores-scores.max(dim=-1,keepdim=True)[0]).exp()*mask).sum(dim=-1,keepdim=True).log()