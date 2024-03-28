'''
Reference:
 	MEANTIME: Mixture of attention mechanisms with multi-temporal embeddings for sequential recommendation. 
	Cho, Sung Min, Eunhyeok Park, and Sungjoo Yoo. 
  	Proceedings of the 14th ACM Conference on recommender systems. 2020.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils.layers import TransformerMeantimeBlock, GELU

from models.BaseModel import ImpressionContextSeqModel

class ConstantEmbedding(nn.Module):
	def __init__(self, args, item_feature_num):
		super().__init__()
		hidden = args.hidden_units * item_feature_num
		self.emb = nn.Embedding(1, hidden)
		self.emb_num = 1

	def forward(self, d):
		batch_size, T = d['history_items'].shape
		return self.emb.weight.unsqueeze(0).repeat(batch_size, T, 1)  # B x T x H

class PositionalEmbedding(nn.Module):
	def __init__(self, args, item_feature_num):
		super().__init__()
		max_len = args.history_max+1
		hidden = args.hidden_units*item_feature_num
		self.emb = nn.Embedding(max_len, hidden)
		self.emb_num = 1

	def forward(self, d):
		x = d['history_items']
		batch_size, length = x.size(0),x.size(1)
		return self.emb.weight.unsqueeze(0).repeat(batch_size, 1, 1)[:,:length,:]  # B x T x H

class DayEmbedding(nn.Module):
	def __init__(self, args, situ_dim, situ_num, item_feature_num):
		super().__init__()
		self.emb_num = situ_num
		self.emb = nn.Embedding(situ_dim,args.hidden_units*item_feature_num)
		self.hidden_units = args.hidden_units * item_feature_num
		# self.linear_layer = nn.Linear(args.hidden_units*situ_num,args.hidden_units*item_feature_num)
	
	def forward(self, d):
		e = self.emb(d['history_context_features']).flatten(start_dim=-2)
		# e_linear = self.linear_layer(e)
		e_split = torch.split(e,self.hidden_units,dim=-1)
		return list(e_split)

class SinusoidTimeDiffEmbedding(nn.Module):
	def __init__(self, args, item_feature_num):
		super().__init__()
		self.time_unit_divide = args.time_unit_divide
		self.hidden = args.hidden_units * item_feature_num
		self.freq = args.freq

	def forward(self, d):
		# t : B x T
		# time_diff : B x T x T  (value range: -time_range ~ time_range)
		t = d['history_times']
		time_diff = t.unsqueeze(2) - t.unsqueeze(1)
		time_diff = time_diff.to(torch.float)
		time_diff = time_diff / self.time_unit_divide

		freq_seq = torch.arange(0, self.hidden, 2.0, dtype=torch.float)  # [0, 2, ..., H-2]
		freq_seq = freq_seq.to(time_diff)  # device
		inv_freq = 1 / torch.pow(self.freq, (freq_seq / self.hidden))  # 1 / 10^(4 * [0, 2/H, 4/H, (H-2)/H])

		sinusoid_inp = torch.einsum('bij,d->bijd', time_diff, inv_freq)  # B x T x T x (H/2)
		pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)  # B x T x T x H

		return pos_emb

class ExponentialTimeDiffEmbedding(nn.Module):
	def __init__(self, args, item_feature_num):
		super().__init__()
		self.time_unit_divide = args.time_unit_divide
		self.hidden = args.hidden_units * item_feature_num
		self.freq = args.freq

	def forward(self, d):
		# t : B x T
		# time_diff : B x T x T  (value range: -time_range ~ time_range)
		t = d['history_times']
		time_diff = t.unsqueeze(2) - t.unsqueeze(1)
		time_diff = time_diff.to(torch.float)
		time_diff = time_diff / self.time_unit_divide
		time_diff.abs_()  # absolute to only use the positive part of exp(-x)

		freq_seq = torch.arange(0, self.hidden, 1.0, dtype=torch.float)  # [0, 1, ..., H-1]
		freq_seq = freq_seq.to(time_diff)  # device
		inv_freq = 1 / torch.pow(self.freq, (freq_seq / self.hidden))  # 1 / 10^(4 * [0, 1/H, 2/H, (H-1)/H])

		exponential_inp = torch.einsum('bij,d->bijd', time_diff, inv_freq)  # B x T x T x H
		pos_emb = (-exponential_inp).exp()  # B x T x T x H
		return pos_emb

class Log1pTimeDiffEmbedding(nn.Module):
	def __init__(self, args, item_feature_num):
		super().__init__()
		self.time_unit_divide = args.time_unit_divide
		self.hidden = args.hidden_units * item_feature_num
		self.freq = args.freq

	def forward(self, d):
		# t : B x T
		# time_diff : B x T x T  (value range: -time_range ~ time_range)
		t = d['history_times']
		time_diff = t.unsqueeze(2) - t.unsqueeze(1)
		time_diff = time_diff.to(torch.float)
		time_diff = time_diff / self.time_unit_divide
		time_diff.abs_()  # absolute to only use the positive part of log(1+x)

		freq_seq = torch.arange(0, self.hidden, 1.0, dtype=torch.float)  # [0, 1, ..., H-1]
		freq_seq = freq_seq.to(time_diff)  # device
		inv_freq = 1 / torch.pow(self.freq, (freq_seq / self.hidden))  # 1 / 10^(4 * [0, 1/H, 2/H, (H-1)/H])

		log1p_inp = torch.einsum('bij,d->bijd', time_diff, inv_freq)  # B x T x T x H
		pos_emb = log1p_inp.log1p()  # B x T x T x H
		return pos_emb

class MeantimeBody(nn.Module):
	def __init__(self, args, La, Lr, item_feature_num):
		super().__init__()

		n_layers = args.num_blocks

		self.transformer_blocks = nn.ModuleList(
			[TransformerMeantimeBlock(args, La, Lr, item_feature_num) for _ in range(n_layers)])

	def forward(self, x, attn_mask, abs_kernel, rel_kernel, info):
		# x : B x T x H
		# abs_kernel : La of [B x T x H]
		# rel_kernel : Lr of [B x T x T x H]
		for layer, transformer in enumerate(self.transformer_blocks):
			x = transformer.forward(x, attn_mask, abs_kernel, rel_kernel, layer=layer, info=info)
		return x

class MeanTime(ImpressionContextSeqModel):
	reader = 'ImpressionContextSeqSituReader'
	# runner = 'ImpressionSituRunner'
	runner = 'ImpressionRunner'
	extra_log_args = ['loss_n','dropout']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--hidden_units',type=int,default=60,)
		parser.add_argument('--time_unit_divide', type=int, default=1,help='The timestamp difference is divided by this value')
		parser.add_argument('--freq', type=int, default=10000, help='freq hyperparameter used in temporal embeddings')
		# MEANTIME
		parser.add_argument('--absolute_kernel_types', type=str, default='p-d-c',help="Absolute kernel types separated by'-'(e.g. d-c). p=Pos, d=Day, c=Con")
		parser.add_argument('--relative_kernel_types', type=str, default='s-e-l',help="Relative kernel types separated by'-'(e.g. e-l). s=Sin, e=Exp, l=Log")

		parser.add_argument('--num_blocks', type=int, default=1, help='Number of transformer layers')
		parser.add_argument('--residual_ln_type', type=str, choices=['post', 'pre'], default='pre', help='Two variations exist regarding the order of layer normalization in residual connections in transformers')
		parser.add_argument('--headtype', type=str, choices=['linear', 'dot'], default='dot', help='Two types of prediction heads on top of transformers')
		parser.add_argument('--head_use_ln', type=int, default=0, help='If true, the prediction head also uses layer normalization')

		parser.add_argument('--model_init_range', type=float,default=0.01, help='Range used to initialize the model parameters')
	
		return ImpressionContextSeqModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionContextSeqModel.__init__(self, args, corpus)
		self.item_num = corpus.n_items +1 # add mask
		self.args = args
		self.model_init_range = args.model_init_range

		args.num_items = self.item_num
		hidden = args.hidden_units
		absolute_kernel_types = args.absolute_kernel_types
		relative_kernel_types = args.relative_kernel_types
		##### Footers
		# Token Embeddings
		self.item_feature_dim = sum([corpus.feature_max[f] for f in corpus.item_feature_names]+[self.item_num])
		self.situ_feature_dim = sum([corpus.feature_max[f] for f in corpus.context_feature_names])
		self.item_feature_num = len(corpus.item_feature_names)+1
		self.situ_feature_num = len(corpus.context_feature_names)
		self.item_embedding = nn.Embedding(self.item_feature_dim, hidden)
		# self.item_embedding = nn.Embedding(self.item_num, hidden)
		# Absolute Embeddings
		self.absolute_kernel_embeddings_list = nn.ModuleList()
		if absolute_kernel_types is not None and len(absolute_kernel_types) > 0:
			for kernel_type in absolute_kernel_types.split('-'):
				if kernel_type == 'p':  # position
					emb = PositionalEmbedding(args, self.item_feature_num)
				elif kernel_type == 'd':  # day and all other
					emb = DayEmbedding(args,self.situ_feature_dim, self.situ_feature_num, self.item_feature_num)
				elif kernel_type == 'c':  # constant
					emb = ConstantEmbedding(args, self.item_feature_num)
				else:
					raise ValueError
				if type(emb) == list:
					self.absolute_kernel_embeddings_list += emb
				else:
					self.absolute_kernel_embeddings_list.append(emb)
		self.La = sum([x.emb_num for x in self.absolute_kernel_embeddings_list])
		# Relative Embeddings
		self.relative_kernel_embeddings_list = nn.ModuleList()
		if relative_kernel_types is not None and len(relative_kernel_types) > 0:
			for kernel_type in relative_kernel_types.split('-'):
				if kernel_type == 's':  # time difference
					emb = SinusoidTimeDiffEmbedding(args, self.item_feature_num)
				elif kernel_type == 'e':
					emb = ExponentialTimeDiffEmbedding(args, self.item_feature_num)
				elif kernel_type == 'l':
					emb = Log1pTimeDiffEmbedding(args, self.item_feature_num)
				else:
					raise ValueError
				self.relative_kernel_embeddings_list.append(emb)
		# Lengths
		self.Lr = len(self.relative_kernel_embeddings_list)
		self.L = self.La + self.Lr
		# Sanity check
		print(self.L)
		assert (hidden*self.item_feature_num) % self.L == 0, 'multi-head has to be possible'
		assert len(self.absolute_kernel_embeddings_list) > 0 or len(self.relative_kernel_embeddings_list) > 0
		##### BODY
		self.body = MeantimeBody(args, self.La, self.Lr, self.item_feature_num)
		##### Heads
		# self.bert_head = BertDotProductPredictionHead(args)
		self.headtype = args.headtype
		if args.headtype == 'dot':
			# self.head = BertDotProductPredictionHead(args, self.item_feat_embedding)
			self.head = BertDotProductPredictionHead(args, self.item_embedding, item_feature_num=self.item_feature_num)
		elif args.headtype == 'linear':
			self.head = BertLinearPredictionHead(args,item_feature_num=self.item_feature_num)
		else:
			raise ValueError
		##### dropout
		self.dropout = nn.Dropout(p=args.dropout)
		##### Weight Initialization
		self.self_init_weights()
		##### MISC
		self.ce = nn.CrossEntropyLoss()
		self.softmax = nn.Softmax(dim=-1)

	def self_init_weights(self):
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=self.model_init_range)
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		elif isinstance(module, BertDotProductPredictionHead):
			for param in [module.bias]:
				param.data.zero_()
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	@classmethod
	def code(cls):
		return 'meantime'

	def get_logits(self, d):
		x = d['history_item_features']
		his_items = d['history_items']
		attn_mask = (his_items > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # B x 1 x T x T
		# token_embeddings = self.dropout(self.token_embedding(d)) # B x T x H
		token_embeddings = self.dropout(self.item_embedding(x).flatten(start_dim=-2)) # B x T x H

		# token_embeddings = token_embeddings.unsqueeze(0).expand(self.L, -1, -1, -1)  # L x B x T x H
		# token_embeddings = token_embeddings.chunk(self.L, dim=0)  # L of [1 x B x T x H]
		# token_embeddings = [x.squeeze(0) for x in token_embeddings]  # L of [B x T x H]

		absolute_kernel_embeddings = []
		for emb in self.absolute_kernel_embeddings_list:
			abs_emb = emb(d)
			if type(abs_emb) == list:
				absolute_kernel_embeddings += [self.dropout(a) for a in abs_emb]
			else:
				absolute_kernel_embeddings.append(self.dropout(abs_emb))

		# absolute_kernel_embeddings = [self.dropout(emb(d)) for emb in self.absolute_kernel_embeddings_list]  # La of [B x T x H]
		relative_kernel_embeddings = [self.dropout(emb(d)) for emb in self.relative_kernel_embeddings_list]  # Lr of [B x T x T x H]

		# info = {} if self.output_info else None
		info = None 

		# last_hidden = L of [B x T x H]
		last_hidden = self.body(token_embeddings, attn_mask,
								absolute_kernel_embeddings,
								relative_kernel_embeddings,
								info=info)
		# last_hidden = torch.cat(last_hidden, dim=-1)  # B x T x LH

		return last_hidden, info

	def get_scores(self, d, logits):
		# logits : B x H or M x H
		# if self.training:  # logits : M x H, returns M x V
		# 	h = self.head(logits)  # M x V
		# else:  # logits : B x H,  returns B x C
		# candidates = d['item_id']  # B x C
		if self.headtype == 'dot':
			candidates = d['item_features']  # B x C
			candidates_id = d['item_id']  # B x C
		else:
			candidates = d['item_id']  # B x C
			candidates_id = d['item_id']  # B x C
		h = self.head(logits, candidates, candidates_id)  # B x C
		return h
	
	def forward(self, d):
		logits, info = self.get_logits(d)
		last_logits = logits[:,0,:] # all history is reversed
		predictions = self.get_scores(d,last_logits)
		return {'prediction': predictions, 'feed_dict':d,}
		# ret = {'logits':logits, 'info':info}
		# if self.training:
		#	 labels = d['labels']
		#	 loss, loss_cnt = self.get_loss(d, logits, labels)
		#	 ret['loss'] = loss
		#	 ret['loss_cnt'] = loss_cnt
		# else:
		#	 # get scores (B x V) for validation
		#	 last_logits = logits[:, -1, :]  # B x H
		#	 ret['scores'] = self.get_scores(d, last_logits)  # B x C
		# return ret
	
	class Dataset(ImpressionContextSeqModel.Dataset):
		def _get_feed_dict(self, index):
			# get item features, user features, and context features separately
			feed_dict = super()._get_feed_dict(index)
			pos = self.data['position'][index]
			user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
			feed_dict['history_times'] = np.array([x[1] for x in user_seq]+[self.data['time'][index]])
			feed_dict['history_items'] = np.array([x[0] for x in user_seq]+[self.corpus.n_items])

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
				his_context_features = self._convert_multihot(self.corpus.context_feature_names, his_context_features)
				feed_dict['history_context_features'] = np.concatenate((his_context_features,feed_dict['context_features']),axis=0) 
			
			for key in feed_dict:
				if 'history' in key:
					feed_dict[key] = feed_dict[key][::-1].copy()
			# zero_padding = [0,0,np.array([[0]*len(user_seq[0][-1])]),np.array([[0]*len(item_fnames)])]	
			# while len(feed_dict['history_items']) < self.model.history_max:
			# 	feed_dict['history_items'] = np.append(feed_dict['history_items'],0)
			# 	feed_dict['history_times'] = np.append(feed_dict['history_times'],0)
			# 	feed_dict['history_item_features'] = np.append(feed_dict['history_item_features'],zero_padding[-1],axis=0)
			# 	feed_dict['history_context_features'] = np.append(feed_dict['history_context_features'],zero_padding[-2],axis=0)
   
			return feed_dict

class BertDotProductPredictionHead(nn.Module):
	def __init__(self, args, token_embeddings, input_size=None, item_feature_num=1):
		super().__init__()
		self.token_embeddings = token_embeddings
		hidden = args.hidden_units * item_feature_num
		if input_size is None:
			input_size = hidden
		self.vocab_size = args.num_items + 1
		if args.head_use_ln:
			self.out = nn.Sequential(
				nn.Linear(input_size, hidden),
				GELU(),
				nn.LayerNorm(hidden)
			)
		else:
			self.out = nn.Sequential(
				nn.Linear(input_size, hidden),
				GELU(),
			)
		self.bias = nn.Parameter(torch.zeros(1, self.vocab_size))
		
	def forward(self, x, candidates, candidates_id=None):
		x = self.out(x)  # B x H or M x H
		if candidates is not None:  # x : B x H
			# emb = self.token_embeddings(candidates)  # B x C x H
			emb = self.token_embeddings(candidates).flatten(start_dim=-2)  # B x C x H
			logits = (x.unsqueeze(1) * emb).sum(-1)  # B x C
			bias = self.bias.expand(logits.size(0), -1).gather(1, candidates_id)  # B x C
			logits += bias
		else:  # x : M x H
			emb = self.token_embeddings.weight[:self.vocab_size]  # V x H
			logits = torch.matmul(x, emb.transpose(0, 1))  # M x V
			logits += self.bias
		return logits

class BertLinearPredictionHead(nn.Module):
	def __init__(self, args, input_size=None, item_feature_num=1):
		super().__init__()
		self.vocab_size = args.num_items + 1
		hidden = input_size if input_size is not None else args.hidden_units*item_feature_num
		if args.head_use_ln:
			self.out = nn.Sequential(
				nn.Linear(hidden, hidden),
				GELU(),
				nn.LayerNorm(hidden),
				nn.Linear(hidden, self.vocab_size)
			)
		else:
			self.out = nn.Sequential(
				nn.Linear(hidden, hidden),
				GELU(),
				nn.Linear(hidden, self.vocab_size)
			)

	def forward(self, x, candidates=None, candidates_id=None):
		x = self.out(x)  # B x V or M x V
		if candidates is not None:
			x = x.gather(1, candidates)  # B x C or M x C
		return x



