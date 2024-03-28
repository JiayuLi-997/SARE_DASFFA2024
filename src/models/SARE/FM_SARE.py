import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseModel import ImpressionContextModel
from models.SARE.BaseSARE import *

class FM_SARE(ImpressionContextModel,BaseSARE):
	reader = 'ImpressionContextReader'
	runner = 'ImpressionSituRunner'
	extra_log_args = ['prob_weights','prob_loss_n','situ_weights','situ_lr','situ_l2','topk']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser = BaseSARE.parse_model_args(parser)
		return ImpressionContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionContextModel.__init__(self,args, corpus)
		self.vec_size = args.emb_size
		self.get_generalSARE_init(args, corpus)

		self._define_params()
		self.apply(self.init_weights)
		nn.init.eye_(self.situ_i_transform.weight)
		nn.init.eye_(self.situ_u_transform.weight)

	def _define_params(self):
		self.context_embedding = nn.Embedding(self.context_feature_dim, self.vec_size)
		self.linear_embedding = nn.Embedding(self.context_feature_dim, 1)
		self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)
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

		linear_value = self.overall_bias + self.linear_embedding(context_features).squeeze(dim=-1)
		linear_value = linear_value.sum(dim=-1)
		context_vectors = self.context_embedding(context_features)
		fm_vectors = 0.5 * (context_vectors.sum(dim=-2).pow(2) - context_vectors.pow(2).sum(dim=-2))
		predictions = linear_value + fm_vectors.sum(dim=-1)
		
		situ_u_vectors = self.situ_u_transform(context_vectors[:,0,feed_dict['umh_idx'][0],:].flatten(start_dim=-2))
		situ_i_vectors = self.situ_i_transform(context_vectors[:,:,feed_dict['imh_idx'][0],:].flatten(start_dim=-2))
		situ_predictions, pred_situ, situ_embed = self.situation_predict(situ_u_vectors, situ_i_vectors, [feed_dict[situ] for situ in self.situ_feature_cnts])

		out_dict = {'prediction': predictions, 'situ_prediction':situ_predictions,
		}
		return out_dict
	
	def loss(self, out_dict, target):
		return self.SARE_loss(out_dict,target)

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
			# transfer into multi-hot embedding
			base = 0
			u_idx, i_idx = [], []	
			for i, feature in enumerate(self.corpus.user_feature_names + self.corpus.item_feature_names + id_names):
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
