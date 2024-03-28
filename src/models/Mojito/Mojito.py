'''
Reference:
 	Attention Mixtures for Time-Aware Sequential Recommendation.
	Tran, Viet Anh, et al. 
  	Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2023.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from models.BaseModel import ImpressionContextSeqModel

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


class Mojito(ImpressionContextSeqModel,):
	reader = 'ImpressionContextSeqSituReader'
	# runner = 'ImpressionSituRunner'
	runner = 'ImpressionRunner'

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--num_heads',type=int,default=2)
		parser.add_argument('--num_blocks',type=int,default=3)
		parser.add_argument('--tempo_linspace',type=int,default=8)
		parser.add_argument('--tempo_embedding_dim',type=int,default=8)
		parser.add_argument('--input_scale',type=int,default=0)
		parser.add_argument('--lambda_trans_seq',type=float,default=0.8)
		parser.add_argument('--lambda_glob',type=float,default=0.1)
		parser.add_argument('--residual_type',type=str,default='add')
		parser.add_argument('--fism_history_max',type=int,default=20)
		return ImpressionContextSeqModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionContextSeqModel.__init__(self, args, corpus)
		self.user_feature_dim = sum([corpus.feature_max[f] for f in corpus.user_feature_names+['user_id']])
		# self.item_feature_dim = sum([corpus.feature_max[f] for f in corpus.item_feature_names+['item_id']])
		self.item_feature_dim = sum([corpus.feature_max[f] for f in corpus.item_feature_names])
		self.situ_feature_dim = sum([corpus.feature_max[f] for f in corpus.context_feature_names])
		self.item_feature_num = len(corpus.item_feature_names) # + 1
		self.user_feature_num = len(corpus.user_feature_names) + 1
		self.situ_feature_num = len(corpus.context_feature_names)
 	
		self.vec_size = args.emb_size # d
		self.embedding_dim = args.emb_size
		self.num_heads = args.num_heads
		self.num_blocks = args.num_blocks	
		self.tempo_linspace = args.tempo_linspace
		self.tempo_embedding_dim = args.tempo_embedding_dim
		self.expand_dim = 1
		self.input_scale = args.input_scale
		self.num_global_heads = 2

		# self.local_output_dim = self.vec_size*2
		self.local_output_dim = -1 
		self.residual_type = args.residual_type
 
		self.lambda_trans_seq = args.lambda_trans_seq
		self.lambda_glob = args.lambda_glob

		self.beta = 1.0
		self.fism_history_max = args.fism_history_max #int(self.history_max/2)
 
		self._define_params()
		self.apply(self.init_weights)

	def _define_params(self):
		# self.item_feat_embedding = nn.Embedding(self.item_feature_dim, self.meta_vec_size)
		# self.situ_feat_embedding = nn.Embedding(self.situ_feature_dim, self.meta_vec_size)
		self.basis_time_encodes = nn.ModuleList()
		for i in range(self.situ_feature_num):
			self.basis_time_encodes.append(BasisTimeEncode(self.tempo_linspace,time_dim=self.tempo_embedding_dim,
												  expand_dim=self.expand_dim,device=self.device))
		self.ctx_linear = nn.Linear(self.tempo_embedding_dim*self.situ_feature_num, self.embedding_dim)

		# self.user_embedding = nn.Embedding(self.user_num, self.vec_size)
		self.user_embedding = nn.Embedding(self.user_num, self.vec_size*2)
		self.user_embedding_fism = nn.Embedding(self.user_num, self.vec_size)
		self.item_id_embedding = nn.Embedding(self.item_num, self.vec_size)

		self.position_embedding = nn.Embedding(self.history_max+1, self.vec_size)

		self.dropout_layer = nn.Dropout(p=self.dropout)

		self.dim_head = int(self.vec_size*2 / self.num_heads)
		self.admix_multi_head_attention_blocks = Multihead_Attention_Blocks(self.vec_size, 
								num_blocks=self.num_blocks,dim_head=self.dim_head,num_heads=self.num_heads,
								dropout_rate=self.dropout, output_dim=self.local_output_dim,
								residual_type=self.residual_type)
		self.sigma_noise = nn.Parameter(torch.ones(self.num_global_heads) * 0.1, requires_grad=True)
		# self.sigma_noise = torch.zeros(self.num_global_heads).to(self.device)


	def forward(self, feed_dict):
		hislens = feed_dict['lengths'] # B
		# user_feats = feed_dict['user_features'] # B * 1 * user features(at least user_id)
		item_feats = feed_dict['item_features'] # B * item num * item features(at least item_id)
		situ_feats = feed_dict['context_features'] # B * 1 * situ features
		history_item_feats = feed_dict['history_item_features'] # B * hislens * item features
		history_situ_feats = feed_dict['history_context_features']
		input_item_ids = feed_dict['item_id'] # B * hislen
		input_seq = feed_dict['history_items'] # B * hislen
		input_fism_ids = feed_dict['history_fism_items'] # B * hislen
		# input_fism_ids = feed_dict['history_items'][:,:-1] # B * hislen
		max_len = input_seq.shape[1]
		user_ids = feed_dict['user_id']
		reverse=False
		if np.random.rand()>0.5:
			item_feats = torch.cat([item_feats[:,50:,:],item_feats[:,:50,:]],dim=1)
			input_item_ids = torch.cat([input_item_ids[:,50:],input_item_ids[:,:50]],dim=1)
			reverse=True

		users = self.user_embedding(user_ids)
		seq = self.item_id_embedding(input_seq)
		if self.input_scale:
			seq = seq * (self.vec_size**0.5)
		nonscale_input_seq = self.item_id_embedding(input_seq) # B * hislen * vec size

		# absolute position sequence representation
		positions_mask = torch.arange(max_len)[None,:].to(self.device) < hislens[:,None]
		positions = (torch.arange(max_len).to(self.device)[None,:]+1) * positions_mask
		abs_position = self.position_embedding(positions) # B * hislen * vec size
		seq = seq + abs_position # B * seqlen * embedding_dim

		ctx_seq_concat = []
		for i in range(self.situ_feature_num):
			ctx_seq_concat.append(self.basis_time_encodes[i](history_situ_feats[:,:,i]))
		ctx_seq_concat = torch.cat(ctx_seq_concat,dim=-1)
		self_ctx_seq = self.ctx_linear(ctx_seq_concat)
		self_ctx_seq = self.dropout_layer(self_ctx_seq) # B * seqlen * embedding_dim

		ctx_seq = self_ctx_seq.clone()
		if self.input_scale:
			ctx_seq = ctx_seq * (self.vec_size**0.5)
		loc_ctx_seq = ctx_seq + abs_position
  
		mask = torch.tensor(input_seq != 0, dtype=torch.float32).to(self.device)
		loc_seq = self._seq_representation(seq, loc_ctx_seq, self_ctx_seq, 
									 sigma_noise=self.sigma_noise, mask=mask) # attention, B * hislen * 2*embedding
  
		test_item_emb = self.item_id_embedding(input_item_ids) # B * item num * embedding
		test_ctx_concat = []
		for i in range(self.situ_feature_num):
			test_ctx_concat.append(self.basis_time_encodes[i](situ_feats[:,:,i]))
		test_ctx_concat = torch.cat(test_ctx_concat,dim=-1)
		test_ctx = self.ctx_linear(test_ctx_concat) # B * 1 * embedding
		test_ctx = self.dropout_layer(test_ctx).repeat(1, test_item_emb.shape[1],1)
		fused_test_item_emb = torch.cat([test_item_emb, test_ctx],dim=-1) # B * item num * 2*embedding
		
		loc_seq_att = self._fism_attentive_vectors(users,loc_seq, fused_test_item_emb, mask)
		# loc_test_logits = torch.matmul(loc_seq, fused_test_item_emb.transpose(dim0=-1,dim1=-2)) # B * hislen * item num
		loc_test_logits = (loc_seq_att*fused_test_item_emb).sum(dim=-1)

		fism_items = self.item_id_embedding(input_fism_ids) # B * fismlen * vec
		# ctx_fism_concat = []
		# for i in range(self.situ_feature_num):
		# 	ctx_fism_concat.append(self.basis_time_encodes[i](history_situ_feats[:,:,i]))
		# ctx_fism_concat = torch.cat(ctx_fism_concat,dim=-1)
		# self_ctx_fism = self.ctx_linear(ctx_fism_concat)
		# self_ctx_fism = self.dropout_layer(self_ctx_fism) # B * fismlen * embedding_dim
		# fused_fism_items = torch.cat([fism_items, self_ctx_fism],dim=-1)
		# user_fism_items = torch.cat([users.unsqueeze(1),fism_items],dim=1)
		fism_mask = torch.tensor(input_fism_ids!=0, dtype=torch.float32).to(self.device)
		# fism_mask = torch.cat([torch.ones(fism_mask.shape[0],1).to(self.device),fism_mask],dim=-1)

		# # # 接下来这部分是paper中没有的
		# att_seq = self._fism_attentive_vectors(users, fism_items, nonscale_input_seq, fism_mask=fism_mask)
		# # nonscale_input_seq: B * hislen * embedding_dim
		# # att_seq: B * hislen * emb
		# glob_seq_vecs = nonscale_input_seq * (1-self.lambda_trans_seq) + (nonscale_input_seq * att_seq) * self.lambda_trans_seq
		# # glob_seq_vecs = glob_seq_vecs[:,1:,:].sum(dim=1,keepdims=True) # why from the 1st?
		# glob_seq_vecs = (glob_seq_vecs*mask.unsqueeze(-1)).sum(dim=1,keepdims=True) # sum over the recent history 
		# loc_test_atts = self._fism_attentive_vectors(users, nonscale_input_seq, test_item_emb, fism_mask=mask) # weighted aggregation of item representations
		# 以下是paper中有的
		users_fism = self.user_embedding_fism(user_ids)
		glob_test_atts = self._fism_attentive_vectors(users_fism, fism_items, test_item_emb, fism_mask=fism_mask) # weighted aggregation of item representations
		
		# code implementation, not reasonable? glob_test_logits = test_item_emb * (1-self.lambda_trans_seq) + (test_item_emb * glob_test_atts) * self.lambda_trans_seq
		glob_test_logits = glob_test_atts
		
		# glob_test_logits = (glob_test_logits + glob_seq_vecs) / hislens[:,None,None]
		# glob_test_logits = (glob_test_logits + loc_test_atts) / hislens[:,None,None]
		glob_test_logits = (glob_test_logits) / hislens[:,None,None]
		glob_test_logits = (glob_test_logits*test_item_emb).sum(dim=-1)

		# loc_test_logits = loc_test_logits[:,0,:]
		test_logits = (1-self.lambda_glob) * loc_test_logits + (self.lambda_glob) * glob_test_logits
		# test_logits = (1-self.lambda_glob) * loc_test_logits.sigmoid() + (self.lambda_glob) * glob_test_logits.sigmoid()

		if test_logits.isnan().max() or test_logits.isinf().max():
			print("Error!")
		if reverse:
			test_logits = torch.cat([test_logits[:,-50:],test_logits[:,:-50]],dim=1)

		return {'prediction': test_logits, 'feed_dict':feed_dict,}

	def _seq_representation(self, seq, ctx_seq, self_ctx_seq, sigma_noise, mask):
		concat_seq = torch.cat([seq,ctx_seq],dim=-1) # B * seqlen * (vec*2)
		return self._adix_sas_representation(seq=concat_seq, context_seq=self_ctx_seq,sigma_noise=sigma_noise,
									   mask=mask)

	def _adix_sas_representation(self, seq, context_seq, sigma_noise, mask):
		# seq: B * seqlen * (vec*2), context_seq: B * seqlen * vec
		sigma_noise = self.sigma_noise.unsqueeze(0).repeat(seq.shape[0],1) # B * 2
		seq = F.dropout(seq, p=self.dropout)
		seq = seq * mask.unsqueeze(-1)
		seq = self.admix_multi_head_attention_blocks(seq=seq,context_seq=context_seq,
							 sigma_noise=sigma_noise, mask=mask)
		seq = normalize(seq)
		return seq

	def _fism_attentive_vectors(self, users, fism_items, seq, fism_mask):
		# seq: B * len * emb, fism: B * (flen) * emb, users: B * emb
		w_ij = torch.matmul(seq, fism_items.transpose(1, 2)) # B * len * (flen)
		exp_wij = torch.exp(w_ij-w_ij.max())
		exp_sum = torch.sum(exp_wij*fism_mask[:,None,:], dim=-1, keepdim=True)
		if self.beta != 1.0:
			exp_sum = torch.pow(exp_sum, self.beta)
		att = exp_wij / exp_sum * fism_mask.unsqueeze(1)
		att_vecs = torch.matmul(att, fism_items) # B * len * emb
	
		if att_vecs.isnan().max() or att_vecs.isinf().max():
			print("Error!")
		if att.isnan().max() or att.isinf().max():
			print("Error!")
		if exp_wij.isnan().max() or exp_wij.isinf().max():
			print("Error!")
		att_u_vecs = att_vecs + users[:,None,:]

		return att_u_vecs
	
	class Dataset(ImpressionContextSeqModel.Dataset):
		def __init__(self, model, corpus, phase: str):
			super().__init__(model,corpus,phase)
			self.include_id = False

		def _get_feed_dict(self, index):
			# get item features, user features, and context features separately
			feed_dict = super()._get_feed_dict(index)
			pos = self.data['position'][index]
			user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
			# input_fism_ids = feed_dict['history_fism_items'] # B * hislen
			fism_idx = np.random.choice(len(user_seq),size=min(self.model.fism_history_max,pos),replace=False)
			feed_dict['history_fism_items'] = np.array([user_seq[idx][0] for idx in fism_idx])
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
	
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
			# zero_padding = [0,0,np.array([[0]*len(user_seq[0][-1])]),np.array([[0]*len(item_fnames)])]	
			# while len(feed_dict['history_items']) < self.model.history_max:
			# 	feed_dict['history_items'] = np.append(feed_dict['history_items'],0)
			# 	feed_dict['history_times'] = np.append(feed_dict['history_times'],0)
			# 	feed_dict['history_item_features'] = np.append(feed_dict['history_item_features'],zero_padding[-1],axis=0)
			# 	feed_dict['history_context_features'] = np.append(feed_dict['history_context_features'],zero_padding[-2],axis=0)
   
			return feed_dict
	

class DenseLayer(nn.Module):
	def __init__(self, input_dim, output_dim, activation='ReLU'):
		super(DenseLayer, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim)
		self.activation = getattr(nn, activation)()

	def forward(self, x):
		return self.activation(self.linear(x)) # 存疑：没有归一化，activation funciton是否要加？

class Multihead_Attention(nn.Module):
	def __init__(self, vec_size, num_heads=8, dim_head=16, dropout_rate=0, 
			causality=False, residual_type='add'):
		super(Multihead_Attention,self).__init__()
		self.num_heads = num_heads
		self.dim_head = dim_head
		self.dropout_rate = dropout_rate
		self.causality = causality
		self.residual_type = residual_type
		self.dropout_layer = nn.Dropout(p=self.dropout_rate)
	
		self.q_glob_it_linear = nn.Linear(vec_size*2, dim_head,bias=False)
		self.k_glob_it_linear = nn.Linear(vec_size*2, dim_head,bias=False)
		self.q_glob_ctx_linear = nn.Linear(vec_size, dim_head,bias=False)
		self.k_glob_ctx_linear = nn.Linear(vec_size, dim_head,bias=False)

		self.v_linear = nn.Linear(vec_size*2, self.num_heads*self.dim_head, bias=False)
		self.hdp_net = DenseLayer(2, self.num_heads,activation='Identity')
	
	def head_split(self, x):  # get dimensions bs * h * seq_len * d_k
		new_x_shape = x.size()[:-1] + (self.num_heads, self.dim_head)
		return x.view(*new_x_shape).transpose(-2, -3)
  
	def forward(self, queries_list, keys_list, sigma_noise, key_mask=None, query_mask=None):
		num_global_heads = len(queries_list)
		queries_it, queries_ctx = queries_list
		keys_it, keys_ctx = keys_list
		dim_glob_head = self.dim_head
		
		Q_glob_it = self.q_glob_it_linear(queries_it)
		K_glob_it = self.k_glob_it_linear(keys_it)
		Q_glob_ctx = self.q_glob_ctx_linear(queries_ctx)
		K_glob_ctx = self.k_glob_ctx_linear(keys_ctx)

		V = self.v_linear(keys_it) # B * T_k * (num heads*dim_heads)

		# Q_glob_it_, K_glob_it_ = self.head_split(Q_glob_it), self.head_split(K_glob_it)
		# Q_glob_ctx_, K_glob_ctx_ = self.head_split(Q_glob_ctx), self.head_split(K_glob_ctx)
		Q_glob_it_, K_glob_it_ = Q_glob_it, K_glob_it
		Q_glob_ctx_, K_glob_ctx_ = Q_glob_ctx, K_glob_ctx
		V_ = self.head_split(V) # B * h * T_k * C/h
  
		mean_att_scores_it = torch.matmul( Q_glob_it_, K_glob_it_.transpose(-2,-1) )
		mean_att_scores_ctx = torch.matmul( Q_glob_ctx_, K_glob_ctx_.transpose(-2,-1) )
		mean_att_scores_list = [mean_att_scores_it, mean_att_scores_ctx]
		mean_att_scores = torch.stack(mean_att_scores_list,dim=1) # B * 2 * T_q * T_k

		att_scores = mean_att_scores + (sigma_noise**2)[:,:,None,None]*torch.randn_like(mean_att_scores)
		att_scores = att_scores.transpose(1,-1) # B * T_k * T_q * 2
		att_scores = self.hdp_net(att_scores) # B * T_k * T_q * h
		att_scores = att_scores.transpose(1,-1) # B * h * T_q * T_k

		# scale
		att_scores = att_scores / self.dim_head**0.5
		if key_mask is not None: # B * T_k
			key_mask = key_mask.unsqueeze(1).unsqueeze(1).repeat(1,self.num_heads,att_scores.shape[2],1)
			att_scores = att_scores.masked_fill(key_mask == 0, -np.inf)
		att_scores = (att_scores-att_scores.max()).softmax(dim=-1)

		if query_mask is not None: # B * T_q
			query_mask = query_mask.unsqueeze(1).unsqueeze(-1).repeat(1, self.num_heads, 1, att_scores.shape[-1])
			att_scores = att_scores * query_mask
		att_scores = self.dropout_layer(att_scores) # B * h * T_q * T_k

		outputs = torch.matmul(att_scores, V_) # B * h * T_q * C/h

		outputs = torch.transpose(outputs,dim0=1,dim1=2).flatten(start_dim=-2) # B * T_q * C
		# C = vec*2
		if self.residual_type == 'add':
			outputs = queries_it + outputs
		elif self.residual_type == 'mult':
			outputs = queries_it + outputs
		elif self.residual_type == 'none':
			pass

		return outputs
  

class Multihead_Attention_Blocks(nn.Module):
	def __init__(self, vec_size, num_blocks, dim_head, num_heads, dropout_rate,output_dim=-1,
			  residual_type='add'):
		super(Multihead_Attention_Blocks,self).__init__()
		self.embedding_dim = num_heads * dim_head
		self.num_blocks = num_blocks
		self.num_heads = num_heads
		self.dim_head = dim_head
		self.dropout_rate = dropout_rate
		self.output_dim = output_dim
		self.mh_attention = nn.ModuleList()
		self.ffn_layers = nn.ModuleList()
		for i in range(num_blocks):
			self.mh_attention.append(Multihead_Attention(vec_size,num_heads=num_heads,
								dim_head=dim_head,dropout_rate=dropout_rate,residual_type=residual_type))
			if i == num_blocks-1 and output_dim>0:
				num_units = [self.embedding_dim, output_dim]
			else:
				num_units = [self.embedding_dim, self.embedding_dim]
			self.ffn_layers.append(FeedForward(num_units,dropout_rate=dropout_rate))

	def forward(self, seq, context_seq, sigma_noise, mask):
		# seq: B * hislen * (vec*2), context_seq: B * hislen * vec
		for i in range(self.num_blocks):
			queries_list=[normalize(seq),normalize(context_seq)]
			keys_list = [seq,context_seq]
			seq = self.mh_attention[i](queries_list,keys_list,sigma_noise,mask,mask) # same size as seq
			# if i == self.num_blocks-1 and self.output_dim>0:
				# num_
			# seq = self.ffn_layers[i](seq.transpose(-1,-2)).transpose(-1,-2)
			seq = self.ffn_layers[i](seq)
			seq = seq * mask.unsqueeze(-1)
			# context_seq = seq[:, :, self.dim_head:]
			vec_size = int(seq.shape[-1]/2)
			context_seq = seq[:, :, :vec_size]
		return seq

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
		outputs = outputs + inputs
		
		return outputs
	
class BasisTimeEncode(nn.Module):
	def __init__(self, tempo_linspace, time_dim, expand_dim, device):
		super(BasisTimeEncode, self).__init__()
		self.device = device
		self.tempo_linspace = tempo_linspace
		self.time_dim = time_dim
		self.expand_dim = expand_dim
		init_period_base = torch.linspace(0, self.tempo_linspace, self.time_dim).to(self.device)
		self.period_var = nn.Parameter(torch.pow(10.0, init_period_base), requires_grad=False).unsqueeze(dim=1).repeat(1,expand_dim)
		self.expand_coef = torch.arange(1, self.expand_dim + 1).float().unsqueeze(0).to(self.device)
		self.freq_var = 1 / self.period_var * self.expand_coef
		self.basis_expan_var = nn.Parameter(torch.randn(self.time_dim, 2 * self.expand_dim), requires_grad=True)
		self.basis_expan_var_bias = nn.Parameter(torch.zeros(self.time_dim), requires_grad=True)

	def forward(self, inputs):
		inputs = inputs.unsqueeze(2).expand(-1, -1, self.time_dim)
		sin_enc = torch.sin(inputs[:,:,:,None] * self.freq_var[None,None,:,:])
		cos_enc = torch.cos(inputs[:,:,:,None] * self.freq_var[None,None,:,:])
		time_enc = torch.cat([sin_enc, cos_enc], -1) * self.basis_expan_var[None,None,:,:] 
		enc = time_enc.sum(-1)+ self.basis_expan_var_bias[None,None,:]
		return enc
