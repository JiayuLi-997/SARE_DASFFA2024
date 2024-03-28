import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseModel import ImpressionContextModel
from models.SARE.BaseSARE import *

class xDeepFM_SARE(ImpressionContextModel,BaseSARE):
	reader = 'ImpressionContextReader'
	runner = 'ImpressionSituRunner'
	extra_log_args = ['prob_weights','prob_loss_n','situ_weights','situ_lr','situ_l2','topk']
	
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--layers', type=str, default='[64,64]',
							help="Size of each layer.")
		parser.add_argument('--cin_layers',type=str,default='[8,8]')
		parser.add_argument('--direct', type=int, default=0,
							help="Whether utilize the output of current network for the next layer.")
		parser.add_argument('--reg_weight',type=float, default=0.2)
		parser = BaseSARE.parse_model_args(parser)
		return ImpressionContextModel.parse_model_args(parser)
	
	@staticmethod
	def init_weights(m):
		# if ''
		if 'Linear' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)
			if m.bias is not None:
				nn.init.normal_(m.bias, mean=0.0, std=0.01)
		elif 'Embedding' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)

	def __init__(self, args, corpus):
		ImpressionContextModel.__init__(self,args, corpus)
		
		self.vec_size = args.emb_size
		self.layers = eval(args.layers)
		self.reg_weight = args.reg_weight

		# reference: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/xdeepfm.py
		self.direct = args.direct
		self.cin_layer_size = temp_cin_size = eval(args.cin_layers)
		# Check whether the size of the CIN layer is legal.
		if not self.direct:
			self.cin_layer_size = list(map(lambda x: int(x // 2 * 2), temp_cin_size))
			if self.cin_layer_size[:-1] != temp_cin_size[:-1]:
				self.logger.warning(
					"Layer size of CIN should be even except for the last layer when direct is True."
					"It is changed to {}".format(self.cin_layer_size)
				)
		self.get_generalSARE_init(args, corpus)
		# self.context_feature_dim = sum(corpus.feature_max.values()) 
		# self.context_feature_num = len(corpus.feature_max)

		self._define_params()
		self.apply(self.init_weights)
		nn.init.eye_(self.situ_i_transform.weight)
		nn.init.eye_(self.situ_u_transform.weight)

	def _define_params(self):
		# FM
		self.context_embedding = nn.Embedding(self.context_feature_dim, self.vec_size)
		self.linear_embedding = nn.Embedding(self.context_feature_dim, 1)
		self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)
		
		# CIN
		# Create a convolutional layer for each CIN layer
		self.conv1d_list = nn.ModuleList()
		self.field_nums = [self.context_feature_num]
		for i, layer_size in enumerate(self.cin_layer_size):
			conv1d = nn.Conv1d(self.field_nums[-1] * self.field_nums[0], layer_size, 1) # in channels, out channels, kernel_size (卷积大小是kernel_size*in channels), 沿非in channel的方向扫
			self.conv1d_list.append(conv1d)
			if self.direct:
				self.field_nums.append(layer_size)
			else:
				self.field_nums.append(layer_size // 2)
		# Get the output size of CIN
		if self.direct:
			self.final_len = sum(self.cin_layer_size)
		else:
			self.final_len = (
				sum(self.cin_layer_size[:-1]) // 2 + self.cin_layer_size[-1]
			)
		self.cin_linear = nn.Linear(self.final_len, 1)

		# Deep
		pre_size = self.context_feature_num * self.vec_size
		self.deep_layers = torch.nn.ModuleList()
		for size in self.layers:
			self.deep_layers.append(torch.nn.Linear(pre_size, size))
			# self.deep_layers.append(torch.nn.BatchNorm1d(size))
			self.deep_layers.append(torch.nn.ReLU())
			self.deep_layers.append(torch.nn.Dropout(self.dropout))
			pre_size = size
		if len(self.layers):
			self.deep_layers.append(torch.nn.Linear(pre_size, 1))
		self.get_generalSARE_params()
  
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

	def forward(self, feed_dict):
		context_features = feed_dict['context_mh']
		context_vectors = self.context_embedding(context_features)
		# FM
		fm_prediction = self.overall_bias + self.linear_embedding(context_features).squeeze(dim=-1).sum(dim=-1)
		# deep
		deep_vectors = context_vectors.flatten(start_dim=-2) # batch size * item num * (feature num * emb size)
		for layer in self.deep_layers:
			deep_vectors = layer(deep_vectors)
		deep_prediction = deep_vectors.squeeze(dim=-1)
		# CIN
		cin_output = self.compreseed_interaction_network(context_vectors, feed_dict['item_id'], activation="nn.ELU")
		cin_prediction = self.cin_linear(cin_output).squeeze(dim=-1)

		predictions = fm_prediction + deep_prediction + cin_prediction
		
		situ_u_vectors = self.situ_u_transform(context_vectors[:,0,feed_dict['umh_idx'][0],:].flatten(start_dim=-2))
		# situ_u_vectors = (context_vectors[:,0,feed_dict['umh_idx'][0],:].flatten(start_dim=-2))
		# situ_u_vectors = (context_vectors[:,0,feed_dict['umh_idx'][0],:].flatten(start_dim=-2))
		situ_i_vectors = self.situ_i_transform(context_vectors[:,:,feed_dict['imh_idx'][0],:].flatten(start_dim=-2))
		situ_predictions, pred_situ, situ_embed = self.situation_predict(situ_u_vectors, situ_i_vectors, [feed_dict[situ] for situ in self.situ_feature_cnts])

		out_dict = {'prediction': predictions, 'situ_prediction':situ_predictions,
		}
		return out_dict

	def loss(self, out_dict, target):
		l2_loss = self.reg_weight * self.reg_loss() 
		all_loss, prob_loss, situ_loss, rec_loss = self.SARE_loss(out_dict,target)
		all_loss += l2_loss
		return all_loss, prob_loss, situ_loss, rec_loss

	def l2_reg(self, parameters, include_bias=False):
		"""
		Reference: RecBole
		Calculate the L2 normalization loss of parameters in a certain layer.
		Returns:
			loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
		"""
		reg_loss = 0
		for name, parm in parameters:
			if include_bias:
				if name.endswith("bias"):
					reg_loss = reg_loss + parm.norm(2)
			if name.endswith("weight"):
				reg_loss = reg_loss + parm.norm(2)
		return reg_loss

	def reg_loss(self):
		l2_reg_loss = self.l2_reg(self.deep_layers.named_parameters()) + self.l2_reg(self.linear_embedding.named_parameters())
		for conv1d in self.conv1d_list:
			l2_reg_loss += self.l2_reg(conv1d.named_parameters(),include_bias=True)
		return l2_reg_loss

	def compreseed_interaction_network(self, input_features, item_ids, activation="nn.ReLU"):
		"""Reference:
			RecBole - https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/xdeepfm.py
		"""
		batch_size, item_num, feature_num, embedding_size = input_features.shape
		all_item_result = []
		for item_idx in range(item_num):
			if (item_ids[:,item_idx]>0).sum() == 0:
				all_item_result.append(torch.zeros(batch_size, self.final_len).to(self.device))
				continue
			hidden_nn_layers = [input_features[:,item_idx,:,:]]
			final_result = []
			for i, layer_size in enumerate(self.cin_layer_size):
				z_i = torch.einsum( "bhd,bmd->bhmd", hidden_nn_layers[-1], hidden_nn_layers[0])
				z_i = z_i.view(
					batch_size, self.field_nums[0] * self.field_nums[i], embedding_size
				)
				z_i = self.conv1d_list[i](z_i) # 每个embedding位置的所有feature加权求和 

				# Pass the CIN intermediate result through the activation function.
				if activation.lower() == "identity" or activation == "None":
					output = z_i
				else:
					activate_func = eval(activation)()
					output = activate_func(z_i)

				# Get the output of the hidden layer.
				if self.direct:
					direct_connect = output
					next_hidden = output
				else:
					if i != len(self.cin_layer_size) - 1:
						next_hidden, direct_connect = torch.split(
							output, 2 * [layer_size // 2], 1
						)
					else:
						direct_connect = output
						next_hidden = 0
				final_result.append(direct_connect) # direct_connect: batch size *  * 
				hidden_nn_layers.append(next_hidden)
			result = torch.cat(final_result, dim=1)
			result = torch.sum(result, dim=-1)
			all_item_result.append(result)
			# all_item_result.append(result.unsqueeze(1))
		all_item_result = torch.stack(all_item_result,dim=1)
		return all_item_result

	class Dataset(ImpressionModel.Dataset):
		def __init__(self, model, corpus, phase: str):
			super().__init__(model,corpus,phase)
			self.include_id = model.include_id
		
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			for feature in self.data.keys():
				if feature[:2] == 'c_':
					feed_dict[feature] = self.data[feature][index]
			All_context, user_features, items_features = [],[],[] # concatenate all context information
			if len(self.corpus.user_feature_names):
				user_features = np.array([self.corpus.user_features[feed_dict['user_id']][c] 
							for c in self.corpus.user_feature_names]).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				All_context = user_features.copy()
			if len(self.corpus.item_feature_names):
				items_features = np.array([[self.corpus.item_features[iid][c] if iid>0 else 0 for c in self.corpus.item_feature_names] 
											for iid in feed_dict['item_id'] ])
				All_context = np.concatenate([All_context, items_features.copy()],axis=-1) if len(All_context) else items_features.copy()
			id_names = []
			if self.include_id:
				user_id = np.array([feed_dict['user_id']]).reshape(1,-1).repeat(feed_dict['item_id'].shape[0],axis=0)
				item_ids = feed_dict['item_id'].reshape(-1,1)
				All_context = np.concatenate([All_context, user_id, item_ids],axis=-1) if len(All_context) else np.concatenate([user_id,item_ids],axis=-1)
				id_names = ['user_id','item_id']
				user_features = np.concatenate([user_features,user_id], axis=-1) if len(user_features) else user_id
				items_features = np.concatenate([items_features,item_ids], axis=-1) if len(items_features) else item_ids
			# if len(self.corpus.context_feature_names):
			# 	context_features = np.array([self.data[f][index] for f in self.corpus.context_feature_names]).reshape(1,-1).repeat(
			# 				feed_dict['item_id'].shape[0],axis=0)
			# 	All_context = np.concatenate([All_context, context_features], axis=-1) if len(All_context) else context_features
			# transfer into multi-hot embedding
			base, base_i, base_u = 0, 0, 0
			# i_idx, u_idx = 0,0
			u_idx, i_idx = [], []	
			# for i, feature in enumerate(self.corpus.user_feature_names + self.corpus.item_feature_names + id_names + self.corpus.context_feature_names):
			for i, feature in enumerate(self.corpus.user_feature_names + self.corpus.item_feature_names + id_names ):
				All_context[:,i] += base
				base += self.corpus.feature_max[feature]
				if feature in self.corpus.user_feature_names+['user_id']:
					u_idx.append(i)
				if feature in self.corpus.item_feature_names+['item_id']:
					i_idx.append(i)
			feed_dict['context_mh'] = All_context # item num * feature num (will repeat 'item num' times for user and context features)
			feed_dict['umh_idx'] = u_idx
			feed_dict['imh_idx'] = i_idx
			return feed_dict
