import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as fn

from models.BaseModel import ImpressionContextSeqModel
from models.SARE.BaseSARE import *
from models.context.DIN import Dice

class CAN_SARE(ImpressionContextSeqModel,BaseSARE):
	reader = 'ImpressionContextSeqReader'
	runner = 'ImpressionSituRunner'
	extra_log_args = ['prob_weights','prob_loss_n','situ_weights','situ_lr','situ_l2','topk']

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
		parser = BaseSARE.parse_model_args(parser)
		return ImpressionContextSeqModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionContextSeqModel.__init__(self,args, corpus)
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

		self.get_generalSARE_init(args, corpus)

		self._define_params()
		self.apply(self.init_weights)
		nn.init.eye_(self.situ_i_transform.weight)
		nn.init.eye_(self.situ_his_transform.weight)
		nn.init.eye_(self.situ_u_transform.weight)
	
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


		self.get_generalSARE_params()
		self.situ_u_transform = nn.Linear(self.vec_size*self.user_feature_num+self.feed_vec_size,self.situ_embedding_size)
		self.situ_i_transform = nn.Linear(self.vec_size*self.item_feature_num+sum(self.co_action_layers),
                                    int(self.situ_embedding_size/2))
		self.situ_his_transform = nn.Linear(self.att_layers[-1]*self.item_feature_num+sum(self.co_action_layers),
                                      int(self.situ_embedding_size/2))
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
	
	def situation_predict(self, u_embeddings, i_embeddings, his_embeddings, situ_target):
		if self.ablation_ucfe:
			s = self.la_weights.unsqueeze(0)#.repeat(i_embeddings.shape[0],1)
		else:
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
		din_output, user_his_emb = self.attention_and_dnn(item_feats_emb_din, history_feats_emb_din, hislens, 
                                       user_feats_emb_din, situ_feats_emb_din)
		
		all_coaction = torch.cat([ui_coaction,si_coaction,his_coaction,din_output,],dim=-1)
		logit = self.fcn_net(all_coaction)
		
		user_SARE = torch.cat([user_ids_emb,user_feats_emb_din],dim=-1)
		situ_u_vectors = self.situ_u_transform(user_SARE).squeeze(1)
		item_SARE = torch.cat([ui_coaction, item_feats_emb_din],dim=-1)
		situ_i_vectors = self.situ_i_transform(item_SARE)
		his_SARE = torch.cat([his_coaction,user_his_emb],dim=-1)
		situ_his_vectors = self.situ_his_transform(his_SARE)
		situ_predictions, pred_situ, situ_embed = self.situation_predict(situ_u_vectors, situ_i_vectors,situ_his_vectors, [feed_dict[situ] for situ in self.situ_feature_cnts])
		return {'prediction':logit,'feed_dict':feed_dict, 'situ_prediction':situ_predictions}

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
		his_rep = (user_his_emb*item_feats_emb2d).view(batch_size, item_num, -1)
		return predictions.view(batch_size, item_num, self.din_dnn_layers[-1]), his_rep


	class Dataset(ImpressionContextSeqModel.Dataset):
		def __init__(self, model, corpus, phase: str):
			super().__init__(model,corpus,phase)
			self.include_id = model.include_id
		
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			torch.cuda.empty_cache()
			return feed_dict
